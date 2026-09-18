"""Memory-bounded teacher-forced interventions for one generated target token."""

from dataclasses import dataclass
import copy

import torch

from experiments.path_conflict.operators import grouped_values, source_write, vocabulary_logits


def token_lens(model, state, token):
    normalized = model.model.norm(state)
    weight = model.lm_head.weight[token].float()
    value = (normalized.float() * weight).sum(dim=-1)
    if model.lm_head.bias is not None:
        value += model.lm_head.bias[token].float()
    return value


@dataclass
class PreparedTarget:
    query_id: int
    prefix_length: int
    cache: object


def cached_values(cache, layer, heads, kv_heads):
    """Return cached V as [batch, query_head, source, head_dim]."""
    values = cache.layers[layer].values
    return values.repeat_interleave(heads // kv_heads, dim=1)


def prepare_target(model, prefix_ids):
    """Prefill every token before the decision query with memory-efficient SDPA."""
    history = prefix_ids[:-1]
    if not history:
        raise ValueError("RAGTruth target prefix must contain history before the decision query")

    device = next(model.parameters()).device
    ids = torch.tensor([history], device=device)
    mask = torch.ones_like(ids)

    model.set_attn_implementation("sdpa")
    try:
        with torch.inference_mode():
            output = model.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            cache = output.past_key_values
        del output
    finally:
        model.set_attn_implementation("eager")

    return PreparedTarget(
        query_id=int(prefix_ids[-1]),
        prefix_length=len(prefix_ids),
        cache=cache,
    )


class TokenRun:
    """Run only the final query token against a prefilled history cache."""

    def __init__(self, model, target, groups, cache, intervention=None):
        self.model = model
        self.query = 0
        self.target = int(target)
        self.cache = cache
        self.selected_heads = tuple(groups["selected_heads"])
        self.groups = {
            name: value for name, value in groups.items()
            if name != "selected_heads"
        }
        self.intervention = intervention
        self.residuals = {}
        self.trajectory = []
        self.local = []
        self.handles = []

    def __enter__(self):
        for layer, block in enumerate(self.model.model.layers):
            self.handles.append(
                block.input_layernorm.register_forward_pre_hook(
                    self.save_residual(layer)
                )
            )
            self.handles.append(
                block.self_attn.register_forward_hook(
                    self.attention_hook(layer)
                )
            )
            self.handles.append(
                block.register_forward_hook(self.layer_hook(layer))
            )
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def save_residual(self, layer):
        def hook(module, inputs):
            self.residuals[layer] = inputs[0][0, 0].detach().clone()
        return hook

    def layer_hook(self, layer):
        def hook(module, inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            value = token_lens(self.model, state[0, 0], self.target)
            self.trajectory.append(
                dict(layer=layer, site="after_mlp", target_logit=float(value))
            )
        return hook

    def selected_source_effects(self, layer, module, attention, values, residual):
        baseline = token_lens(self.model, residual, self.target)
        for name, sources in self.groups.items():
            for head_layer, head in self.selected_heads:
                if head_layer != layer:
                    continue
                _, write, mass = source_write(
                    attention,
                    values,
                    module.o_proj.weight,
                    [0],
                    list(sources),
                    [head],
                )
                removed = token_lens(
                    self.model, residual - write[0], self.target
                )
                self.local.append(dict(
                    layer=layer,
                    head=head,
                    source_group=name,
                    attention_mass=float(mass[head, 0]),
                    local_support=float(baseline - removed),
                ))

    def attention_hook(self, layer):
        def hook(module, inputs, output):
            attention_output, attention = output[:2]
            config = self.model.config
            values = cached_values(
                self.cache,
                layer,
                config.num_attention_heads,
                config.num_key_value_heads,
            )
            base_residual = self.residuals.pop(layer)
            residual = base_residual + attention_output[0, 0]

            self.selected_source_effects(
                layer, module, attention, values, residual
            )

            changed = attention_output
            item = self.intervention
            if item is not None and item["layer"] == layer:
                _, write, _ = source_write(
                    attention,
                    values,
                    module.o_proj.weight,
                    [0],
                    list(self.groups[item["source_group"]]),
                    [item["head"]],
                )
                changed = changed.clone()
                changed[0, 0] -= write[0]

            after = base_residual + changed[0, 0]
            value = token_lens(self.model, after, self.target)
            self.trajectory.append(dict(
                layer=layer,
                site="after_attention",
                target_logit=float(value),
            ))

            # Attention weights have already been reduced to Python scalars.
            # Do not retain even the query-row tensor in the model output.
            if len(output) == 1:
                return (changed,)
            return (changed, None, *output[2:])

        return hook


def forward_target(model, prepared, target, groups, intervention=None):
    """Decode one query token; eager attention is only [heads,1,prefix]."""
    device = next(model.parameters()).device
    cache = copy.deepcopy(prepared.cache)
    ids = torch.tensor([[prepared.query_id]], device=device)
    mask = torch.ones((1, prepared.prefix_length), dtype=torch.long, device=device)

    model.set_attn_implementation("eager")
    with torch.inference_mode(), TokenRun(
        model, target, groups, cache, intervention
    ) as run:
        output = model.model(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        final_state = output.last_hidden_state[0, 0]
        logits = vocabulary_logits(model, final_state)
        log_prob = logits.double().log_softmax(-1)[target]

    del output, cache
    return float(log_prob), run
