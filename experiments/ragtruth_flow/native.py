"""Teacher-forced observer interventions for the actually generated target token."""

import torch

from experiments.path_conflict.operators import grouped_values, source_write, vocabulary_logits


def token_lens(model, state, token):
    normalized = model.model.norm(state)
    weight = model.lm_head.weight[token].float()
    value = (normalized.float() * weight).sum(dim=-1)
    if model.lm_head.bias is not None:
        value += model.lm_head.bias[token].float()
    return value


class TokenRun:
    def __init__(self, model, prefix_length, target, groups, intervention=None):
        self.model = model
        self.query = prefix_length - 1
        self.target = int(target)
        self.selected_heads = tuple(groups["selected_heads"])
        self.groups = {name: value for name, value in groups.items() if name != "selected_heads"}
        self.intervention = intervention
        self.values = {}
        self.residuals = {}
        self.trajectory = []
        self.local = []
        self.final_normalized = None
        self.handles = []

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(block.input_layernorm.register_forward_pre_hook(self.save_residual(layer)))
            self.handles.append(block.self_attn.v_proj.register_forward_hook(self.save_values(layer)))
            self.handles.append(block.self_attn.register_forward_hook(self.attention_hook(layer)))
            self.handles.append(block.register_forward_hook(self.layer_hook(layer)))
        self.handles.append(self.model.lm_head.register_forward_pre_hook(self.save_final))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def save_residual(self, layer):
        def hook(module, inputs):
            self.residuals[layer] = inputs[0][0, self.query].detach().clone()
        return hook

    def save_values(self, layer):
        def hook(module, inputs, output):
            self.values[layer] = output.detach()
        return hook

    def save_final(self, module, inputs):
        self.final_normalized = inputs[0][0, -1].detach()

    def layer_hook(self, layer):
        def hook(module, inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            value = token_lens(self.model, state[0, self.query], self.target)
            self.trajectory.append(dict(layer=layer, site="after_mlp", target_logit=float(value)))
        return hook

    def attention_hook(self, layer):
        def hook(module, inputs, output):
            attention_output, attention = output[:2]
            config = self.model.config
            values = grouped_values(
                self.values.pop(layer), config.num_attention_heads, config.num_key_value_heads
            )
            base_residual = self.residuals.pop(layer)
            residual = base_residual + attention_output[0, self.query]
            baseline = token_lens(self.model, residual, self.target)

            for name, sources in self.groups.items():
                for head_layer, head in self.selected_heads:
                    if head_layer != layer:
                        continue
                    _, write, mass = source_write(
                        attention, values, module.o_proj.weight,
                        [self.query], list(sources), [head],
                    )
                    removed = token_lens(self.model, residual - write[0], self.target)
                    self.local.append(dict(
                        layer=layer, head=head, source_group=name,
                        attention_mass=float(mass[head, 0]),
                        local_support=float(baseline - removed),
                    ))

            changed = attention_output
            item = self.intervention
            if item is not None and item["layer"] == layer:
                _, write, _ = source_write(
                    attention, values, module.o_proj.weight,
                    [self.query], list(self.groups[item["source_group"]]), [item["head"]],
                )
                changed = changed.clone()
                changed[0, self.query] -= write[0]

            after = base_residual + changed[0, self.query]
            value = token_lens(self.model, after, self.target)
            self.trajectory.append(dict(layer=layer, site="after_attention", target_logit=float(value)))
            return (changed, *output[1:])
        return hook


def forward_target(model, prefix_ids, target, groups, intervention=None):
    ids = torch.tensor([prefix_ids], device=next(model.parameters()).device)
    with torch.inference_mode(), TokenRun(
        model, len(prefix_ids), target, groups, intervention
    ) as run:
        model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        logits = vocabulary_logits(model, run.final_normalized)
        log_prob = logits.double().log_softmax(-1)[target]
    return float(log_prob), run
