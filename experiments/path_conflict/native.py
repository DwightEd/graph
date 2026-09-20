"""Llama eager-attention interventions using its own A, V and output projection.

Remove actual source-conditioned writes, without changing or renormalizing A.
All interventions precede the candidate token; subsequent layers run normally.
"""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .operators import (
    grouped_values, source_write, equal_norm_change, lens_margin,
    local_readout_direction,
)


@dataclass(frozen=True)
class Intervention:
    layer: int
    groups: tuple
    scope: str = 'query'
    heads: tuple = ()
    operation: str = 'cut'
    seed: int = 0
    dose: float = 1.0
    reference_norm: float | None = None
    queries: tuple = ()
    sources: tuple = ()
    replacement: torch.Tensor | dict | None = None


def request_layer_attention(module, args, kwargs):
    """Expose each layer's A to hooks without retaining the full attention stack."""
    return args, dict(kwargs, output_attentions=True)


class NativeRun:
    """One forward's hooks. Handles are removed even when the forward raises."""
    def __init__(self, model, prefix_length, groups, candidate_ids, interventions=(), restore=None, trace_heads=(), trace_window=10, record_writes=True, capture_units=()):
        self.model = model
        self.prefix_length = prefix_length
        self.query = prefix_length - 1
        self.groups = groups
        self.correct, self.wrong = candidate_ids
        self.interventions = {}
        for item in interventions:
            self.interventions.setdefault(item.layer, []).append(item)
        self.restore = restore
        self.trace_heads = trace_heads
        self.trace_window = trace_window
        self.record_writes = record_writes
        self.capture_units = capture_units
        self.message_writes = {}
        self.routes = {}
        self.baseline_mlps = {}
        self.final_normalized = None
        self.values, self.head_states, self.residuals = {}, {}, {}
        self.baseline_heads, self.trajectory, self.writes, self.changes = {}, [], [], []
        self.handles = []
        self.reconstruction = []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.register_forward_pre_hook(request_layer_attention, with_kwargs=True))
            self.handles.append(layer.input_layernorm.register_forward_pre_hook(self.save_residual(index)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self.save_values(index)))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self.save_heads(index)))
            self.handles.append(layer.self_attn.register_forward_hook(self.attention_hook(index)))
            self.handles.append(layer.mlp.register_forward_hook(self.mlp_hook(index)))
            self.handles.append(layer.register_forward_hook(self.layer_hook(index)))
        self.handles.append(self.model.lm_head.register_forward_pre_hook(self.save_normalized))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def save_normalized(self, module, inputs):
        self.final_normalized = inputs[0][0].detach()

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
            if self.record_writes and not self.interventions and self.restore is None:
                self.baseline_mlps[index] = output[0, self.query].detach().cpu().clone()
            if not self.interventions:
                for unit in self.capture_units:
                    if unit['layer'] == index:
                        self.message_writes['mlp_' + unit['unit']] = output[0, [unit['receiver']]].detach().cpu()
            changed = output
            for item in self.interventions.get(index, ()):
                if item.groups == ('mlp',):
                    queries = self.query_positions(item)
                    write = self.intervention_change(item, output[0, queries])
                    changed = changed.clone()
                    changed[0, queries] -= write
                    self.changes.append(dict(layer=index, scope=item.scope, operation=item.operation + '_mlp', groups='mlp',
                        dose=item.dose, query_change_norm=float(write[-1].float().norm()),
                        prefix_change_norm=float(write.float().norm()), receiver_positions=queries))
            restore = self.restore
            if restore is not None and restore['layer'] == index and restore.get('site', 'heads') == 'mlp':
                changed = changed.clone()
                changed[0, self.query] = restore['states'].to(changed)
            return changed
        return hook

    def attention_hook(self, index):
        def hook(module, inputs, output):
            attention_output, attention = output[:2]
            config = self.model.config
            values = grouped_values(self.values.pop(index), config.num_attention_heads, config.num_key_value_heads)
            heads = self.head_states.pop(index)
            if not self.interventions:
                self.capture_messages(index, module, attention, values)
            if self.record_writes and not self.interventions and self.restore is None:
                self.baseline_heads[index] = heads[:, :self.prefix_length].cpu().clone()
                self.save_routes(index, module, attention, values, attention_output)
                self.observe(index, module, attention, values, heads, attention_output)
            changed = attention_output
            for item in self.interventions.get(index, ()):
                if item.groups != ('mlp',):
                    changed = self.apply(item, module, attention, values, changed)
            if (self.restore is not None and index == self.restore['layer']
                    and self.restore.get('site', 'heads') == 'heads'):
                changed = self.restore_heads(module, heads, changed)
            residual = self.residuals.pop(index) + changed[0, self.query]
            margin = lens_margin(self.model, residual, self.correct, self.wrong)
            self.trajectory.append(dict(layer=index, site='after_attention', margin=float(margin)))
            return (changed, *output[1:])
        return hook

    def capture_messages(self, index, module, attention, values):
        for unit in self.capture_units:
            if unit['layer'] == index:
                sources = unit['sources'] if 'sources' in unit else [unit['source']]
                _, write, _ = source_write(attention, values, module.o_proj.weight,
                                          [unit['receiver']], sources, [unit['head']])
                self.message_writes[unit['unit']] = write.detach().cpu()

    def apply(self, item, module, attention, values, output):
        queries = self.query_positions(item)
        sources = list(item.sources) if item.sources else sorted(
            {int(source) for name in item.groups for source in self.groups[name]})
        heads = item.heads or tuple(range(self.model.config.num_attention_heads))
        _, write, _ = source_write(attention, values, module.o_proj.weight, queries, sources, heads)
        write = self.intervention_change(item, write)
        changed = output.clone()
        changed[0, queries] -= write
        self.changes.append(dict(layer=item.layer, scope=item.scope, operation=item.operation, dose=item.dose,
            groups='+'.join(item.groups), query_change_norm=float(write[-1].float().norm()),
            prefix_change_norm=float(write.float().norm()), heads=list(heads),
            receiver_positions=queries, source_positions=sources))
        return changed

    def query_positions(self, item):
        if item.queries:
            return list(item.queries)
        return [self.query] if item.scope == 'query' else list(range(self.prefix_length))

    @staticmethod
    def intervention_change(item, write):
        if item.operation == 'replace':
            write = write - item.replacement.to(write)
        if item.operation == 'random':
            return equal_norm_change(item.dose * write, item.seed, item.reference_norm)
        return item.dose * write

    def restore_heads(self, module, current, output):
        """After an upstream cut, restore only named downstream heads at the decision query."""
        head_count = self.model.config.num_attention_heads
        present = current[0, self.query].view(head_count, -1)
        baseline = self.restore['states'][0, self.query].to(present).view(head_count, -1)
        delta = torch.zeros_like(present)
        selected = self.restore['heads'] or tuple(range(head_count))
        delta[list(selected)] = baseline[list(selected)] - present[list(selected)]
        result = output.clone()
        result[0, self.query] += F.linear(delta.flatten(), module.o_proj.weight)
        return result

    def save_routes(self, index, module, attention, values, attention_output):
        queries = list(range(max(0, self.query - self.trace_window), self.query + 1))
        for layer, head in self.trace_heads:
            if layer == index:
                key = f'L{layer}H{head}'
                self.routes[key + '_attention'] = attention[0, head, queries, :self.prefix_length].float().cpu().numpy()
                self.routes[key + '_value_norm'] = values[0, head, :self.prefix_length].float().norm(dim=-1).cpu().numpy()
                self.routes[key + '_queries'] = torch.tensor(queries).numpy()
                width = values.shape[-1]
                output_weight = module.o_proj.weight[:, head * width:(head + 1) * width]
                messages = F.linear(values[0, head, :self.prefix_length], output_weight)
                messages *= attention[0, head, self.query, :self.prefix_length, None]
                residual = self.residuals[index] + attention_output[0, self.query]
                base = lens_margin(self.model, residual, self.correct, self.wrong)
                removed = lens_margin(self.model, residual[None] - messages, self.correct, self.wrong)
                self.routes[key + '_source_write_norm'] = messages.float().norm(dim=-1).cpu().numpy()
                self.routes[key + '_source_lens_support'] = (base - removed).cpu().numpy()

    def observe(self, index, module, attention, values, heads, output):
        count = self.model.config.num_attention_heads
        head_width = heads.shape[-1] // count
        residual = self.residuals[index] + output[0, self.query]
        base_margin = lens_margin(self.model, residual, self.correct, self.wrong)
        direction = local_readout_direction(self.model, residual, self.correct, self.wrong)
        all_sources = list(range(self.prefix_length))
        _, total, _ = source_write(attention, values, module.o_proj.weight, [self.query], all_sources, range(count))
        expected = output[0, self.query].float()
        actual = total[0].float()
        if module.o_proj.bias is not None:
            actual += module.o_proj.bias.float()
        error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.reconstruction.append(float(error))
        for name, sources in self.groups.items():
            components, group_write, mass = source_write(
                attention, values, module.o_proj.weight, [self.query], list(sources), range(count)
            )
            group_margin = lens_margin(
                self.model, residual - group_write[0], self.correct, self.wrong
            )
            self.writes.append(dict(
                layer=index, head=-1, source_group=name,
                attention_mass=float(mass[:, 0].sum()),
                value_norm=float(components[:, 0].float().norm()),
                write_norm=float(group_write[0].float().norm()),
                local_lens_support=float(base_margin - group_margin),
                local_linear_support=float(group_write[0].float() @ direction),
            ))

            blocks = module.o_proj.weight.view(-1, count, head_width).permute(1, 0, 2)
            messages = torch.einsum('hod,hd->ho', blocks, components[:, 0])
            margins = lens_margin(self.model, residual[None] - messages, self.correct, self.wrong)
            for head in range(count):
                self.writes.append(dict(layer=index, head=head, source_group=name,
                    attention_mass=float(mass[head, 0]), value_norm=float(components[head, 0].float().norm()),
                    write_norm=float(messages[head].float().norm()),
                    local_lens_support=float(base_margin - margins[head]),
                    local_linear_support=float(messages[head].float() @ direction)))


def forward(model, ids, probe, interventions=(), restore=None, branch='prefix'):
    # A full-candidate forward can have different bf16 rounding from a short
    # prefix. Restore the baseline from this exact candidate branch and shape.
    interventions = tuple(
        replace(item, replacement=item.replacement[branch])
        if isinstance(item.replacement, dict) else item
        for item in interventions
    )
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    first_ids = [candidate[0] for candidate in probe['candidates']]
    with torch.inference_mode(), NativeRun(model, len(probe['prefix_ids']), probe['groups'],
                                          first_ids, interventions, restore,
                                          probe.get('trace_heads', ()), probe.get('trace_window', 10),
                                          probe.get('record_writes', True), probe.get('capture_units', ())) as run:
        output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                       use_cache=False, output_attentions=False, return_dict=True)
        logits = output.logits[0].float()
    return logits, run


def evaluate(model, probe, interventions=(), restore=None):
    """One canonical prefix readout; see scoring.py for all candidate measures."""
    from .scoring import evaluate_candidates
    record, run = evaluate_candidates(model, probe, interventions, restore)
    return record, run.trajectory, run.writes, run.changes
