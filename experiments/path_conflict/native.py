"""Llama eager-attention interventions using its own A, V and output projection.

Remove actual source-conditioned writes, without changing or renormalizing A.
All interventions precede the candidate token; subsequent layers run normally.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Intervention:
    layer: int
    groups: tuple
    scope: str = 'query'
    heads: tuple = ()
    operation: str = 'cut'
    seed: int = 0


def grouped_values(projected_values, heads, kv_heads):
    batch, length, width = projected_values.shape
    values = projected_values.view(batch, length, kv_heads, width // kv_heads).transpose(1, 2)
    return values.repeat_interleave(heads // kv_heads, dim=1)


def source_write(attention, values, output_weight, queries, sources, heads):
    """Return head-resolved weighted V and its summed W_O write. Batch size is one."""
    selected_attention = attention[0][:, queries][:, :, sources]
    weighted_values = selected_attention @ values[0][:, sources]
    selected = torch.zeros_like(weighted_values)
    selected[list(heads)] = weighted_values[list(heads)]
    joined = selected.transpose(0, 1).reshape(len(queries), -1)
    return selected, F.linear(joined, output_weight), selected_attention.sum(dim=-1)


def equal_norm_change(write, seed):
    """A residual-space random direction with the SAME L2 change per query."""
    generator = torch.Generator(device=write.device).manual_seed(seed)
    random = torch.randn(write.shape, generator=generator, device=write.device, dtype=torch.float32)
    random /= random.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (random * write.float().norm(dim=-1, keepdim=True)).to(write.dtype)


def lens_margin(model, states, correct, wrong):
    normalized = model.model.norm(states)
    direction = model.lm_head.weight[correct].float() - model.lm_head.weight[wrong].float()
    return (normalized.float() * direction).sum(dim=-1)


class NativeRun:
    """One forward's hooks. Handles are removed even when the forward raises."""
    def __init__(self, model, prefix_length, groups, candidate_ids, interventions=(), restore=None):
        self.model = model
        self.prefix_length = prefix_length
        self.query = prefix_length - 1
        self.groups = groups
        self.correct, self.wrong = candidate_ids
        self.interventions = {item.layer: item for item in interventions}
        self.restore = restore
        self.values, self.head_states, self.residuals = {}, {}, {}
        self.baseline_heads, self.trajectory, self.writes, self.changes = {}, [], [], []
        self.handles = []
        self.reconstruction = []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.input_layernorm.register_forward_pre_hook(self.save_residual(index)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self.save_values(index)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self.save_heads(index)))
            self.handles.append(layer.self_attn.register_forward_hook(self.attention_hook(index)))
            self.handles.append(layer.mlp.register_forward_hook(self.mlp_hook(index)))
            self.handles.append(layer.register_forward_hook(self.layer_hook(index)))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def save_residual(self, index):
        def hook(module, inputs):
            self.residuals[index] = inputs[0][0, self.query].detach().clone()
        return hook

    def save_values(self, index):
        def hook(module, inputs, output):
            self.values[index] = output.detach()
        return hook

    def save_heads(self, index):
        def hook(module, inputs):
            self.head_states[index] = inputs[0].detach()
        return hook

    def layer_hook(self, index):
        def hook(module, inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            margin = lens_margin(self.model, state[0, self.query], self.correct, self.wrong)
            self.trajectory.append(dict(layer=index, site='after_mlp', margin=float(margin)))
        return hook

    def mlp_hook(self, index):
        def hook(module, inputs, output):
            item = self.interventions.get(index)
            if item is None or item.groups != ('mlp',):
                return output
            queries = [self.query] if item.scope == 'query' else list(range(self.prefix_length))
            changed = output.clone()
            write = output[0, queries]
            changed[0, queries] = 0
            self.changes.append(dict(layer=index, scope=item.scope, operation='cut_mlp', groups='mlp',
                query_change_norm=float(write[-1].float().norm()), prefix_change_norm=float(write.float().norm())))
            return changed
        return hook

    def attention_hook(self, index):
        def hook(module, inputs, output):
            attention_output, attention = output[:2]
            config = self.model.config
            values = grouped_values(self.values.pop(index), config.num_attention_heads, config.num_key_value_heads)
            heads = self.head_states.pop(index)
            self.baseline_heads[index] = heads[:, :self.prefix_length].clone()
            if not self.interventions and self.restore is None:
                self.observe(index, module, attention, values, heads, attention_output)
            changed = attention_output
            if index in self.interventions and self.interventions[index].groups != ('mlp',):
                changed = self.apply(index, module, attention, values, changed)
            if self.restore is not None and index == self.restore['layer']:
                changed = self.restore_heads(module, heads, changed)
            residual = self.residuals.pop(index) + changed[0, self.query]
            margin = lens_margin(self.model, residual, self.correct, self.wrong)
            self.trajectory.append(dict(layer=index, site='after_attention', margin=float(margin)))
            return (changed, *output[1:])
        return hook

    def apply(self, index, module, attention, values, output):
        item = self.interventions[index]
        queries = [self.query] if item.scope == 'query' else list(range(self.prefix_length))
        sources = sorted({int(source) for name in item.groups for source in self.groups[name]})
        heads = item.heads or tuple(range(self.model.config.num_attention_heads))
        _, write, _ = source_write(attention, values, module.o_proj.weight, queries, sources, heads)
        if item.operation == 'random':
            write = equal_norm_change(write, item.seed)
        changed = output.clone()
        changed[0, queries] -= write
        self.changes.append(dict(layer=index, scope=item.scope, operation=item.operation,
            groups='+'.join(item.groups), query_change_norm=float(write[-1].float().norm()),
            prefix_change_norm=float(write.float().norm()), heads=list(heads)))
        return changed

    def restore_heads(self, module, current, output):
        """After an upstream cut, restore only named downstream heads at the decision query."""
        head_count = self.model.config.num_attention_heads
        present = current[0, self.query].view(head_count, -1)
        baseline = self.restore['states'][0, self.query].view(head_count, -1)
        delta = torch.zeros_like(present)
        selected = self.restore['heads'] or tuple(range(head_count))
        delta[list(selected)] = baseline[list(selected)] - present[list(selected)]
        result = output.clone()
        result[0, self.query] += F.linear(delta.flatten(), module.o_proj.weight)
        return result

    def observe(self, index, module, attention, values, heads, output):
        count = self.model.config.num_attention_heads
        head_width = heads.shape[-1] // count
        residual = self.residuals[index] + output[0, self.query]
        base_margin = lens_margin(self.model, residual, self.correct, self.wrong)
        all_sources = list(range(self.prefix_length))
        _, total, _ = source_write(attention, values, module.o_proj.weight, [self.query], all_sources, range(count))
        expected = output[0, self.query].float()
        actual = total[0].float()
        if module.o_proj.bias is not None:
            actual += module.o_proj.bias.float()
        error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.reconstruction.append(float(error))
        for name, sources in self.groups.items():
            components, _, mass = source_write(attention, values, module.o_proj.weight, [self.query], list(sources), range(count))
            blocks = module.o_proj.weight.view(-1, count, head_width).permute(1, 0, 2)
            messages = torch.einsum('hod,hd->ho', blocks, components[:, 0])
            margins = lens_margin(self.model, residual[None] - messages, self.correct, self.wrong)
            for head in range(count):
                self.writes.append(dict(layer=index, head=head, source_group=name,
                    attention_mass=float(mass[head, 0]), value_norm=float(components[head, 0].float().norm()),
                    write_norm=float(messages[head].float().norm()),
                    local_lens_support=float(base_margin - margins[head])))


def forward(model, ids, probe, interventions=(), restore=None):
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    first_ids = [candidate[0] for candidate in probe['candidates']]
    with torch.inference_mode(), NativeRun(model, len(probe['prefix_ids']), probe['groups'],
                                          first_ids, interventions, restore) as run:
        output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                       use_cache=False, output_attentions=True, return_dict=True)
        logits = output.logits[0].float()
    return logits, run


def evaluate(model, probe, interventions=(), restore=None):
    """Score complete fixed alternatives. No temperature, top-p or newly generated gold."""
    prefix = probe['prefix_ids']
    record = {}
    trajectory, writes, changes = [], [], []
    for name, candidate in zip(('correct', 'wrong'), probe['candidates']):
        logits, run = forward(model, prefix + candidate, probe, interventions, restore)
        positions = torch.arange(len(prefix) - 1, len(prefix) + len(candidate) - 1, device=logits.device)
        target = torch.tensor(candidate, device=logits.device)
        log_probability = logits[positions].log_softmax(-1)[torch.arange(len(candidate), device=logits.device), target]
        record[name + '_logp'] = float(log_probability.sum())
        record[name + '_mean_logp'] = float(log_probability.mean())
        record[name + '_tokens'] = len(candidate)
        record[name + '_token_logps'] = log_probability.cpu().tolist()
        record[name + '_first_logp'] = float(log_probability[0])
        if name == 'correct':
            first = logits[len(prefix) - 1]
            record['next_margin'] = float(first[probe['candidates'][0][0]] - first[probe['candidates'][1][0]])
            record['next_top1'] = int(first.argmax())
            trajectory, writes, changes = run.trajectory, run.writes, run.changes
    record['sequence_margin'] = record['correct_logp'] - record['wrong_logp']
    record['mean_margin'] = record['correct_mean_logp'] - record['wrong_mean_logp']
    return record, trajectory, writes, changes
