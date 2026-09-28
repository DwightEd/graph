"""Reuse native A/V/W_O observation; reroute source mass without deleting text."""
import numpy as np
import torch
from torch.nn import functional as F

from experiments.path_conflict.native import NativeRun
from experiments.path_conflict.operators import grouped_values, vocabulary_logits


def reroute_write(attention, values, weight, query, head, source, target, dose):
    """Move a fraction of one head's source mass to a specified source subset."""
    source_weights = attention[0, head, query, source].float()
    target_weights = attention[0, head, query, target].float()
    source_mass = source_weights.sum()
    target_distribution = target_weights / target_weights.sum().clamp_min(1e-30)
    if float(target_weights.sum()) == 0:
        raise ValueError('No observable attention on target; rerouting is undefined')
    current = source_weights @ values[0, head, source].float()
    requested = source_mass * (target_distribution @ values[0, head, target].float())
    width = values.shape[-1]
    block = weight[:, head * width:(head + 1) * width].float()
    delta = F.linear(dose * (requested - current), block)
    return delta, float(source_mass), float(dose * (source_mass - target_weights.sum()))


class FlowCapture(NativeRun):
    def __init__(self, model, prefix_length, groups, candidate_ids, patch=None, collect=True):
        super().__init__(model, prefix_length, groups, candidate_ids, record_writes=collect)
        self.patch = patch
        self.collect = collect
        self.states = {}
        self.patch_record = {}

    def save_residual(self, index):
        original = super().save_residual(index)
        def hook(module, inputs):
            original(module, inputs)
            if self.collect:
                self.states[f'{index}_pre'] = self.residuals[index].float().cpu().numpy()
        return hook

    def attention_hook(self, index):
        original = super().attention_hook(index)
        def hook(module, inputs, output):
            changed, attention = output[:2]
            if self.patch is not None and self.patch['layer'] == index:
                values = grouped_values(self.values[index], self.model.config.num_attention_heads,
                                        self.model.config.num_key_value_heads)
                delta, mass, moved = reroute_write(attention, values, module.o_proj.weight,
                    self.patch.get('query', self.query), self.patch['head'], self.groups['source'],
                    self.groups[self.patch['target']], self.patch['dose'])
                if self.patch['kind'] == 'random_direction':
                    generator = torch.Generator(device=delta.device).manual_seed(42)
                    noise = torch.randn(delta.shape, generator=generator, device=delta.device)
                    delta = noise * delta.norm() / noise.norm()
                changed = changed.clone()
                changed[0, self.patch.get('query', self.query)] += delta.to(changed)
                self.patch_record = dict(source_mass=mass, moved_mass=moved,
                                         write_delta_norm=float(delta.to(changed).float().norm()),
                                         random_direction=self.patch['kind'] == 'random_direction')
            if self.collect:
                state = self.residuals[index] + changed[0, self.query]
                self.states[f'{index}_attn'] = state.float().cpu().numpy()
            return original(module, inputs, (changed, *output[1:]))
        return hook

    def layer_hook(self, index):
        original = super().layer_hook(index)
        def hook(module, inputs, output):
            result = original(module, inputs, output)
            state = result[0] if isinstance(result, tuple) else result
            if self.collect:
                self.states[f'{index}_mlp'] = state[0, self.query].float().cpu().numpy()
            return result
        return hook


def capture(model, ids, groups, candidate_ids, patch=None, collect=True):
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode(), FlowCapture(model, len(ids), groups, candidate_ids,
                                            patch, collect) as run:
        output = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                             use_cache=False, return_dict=True)
        logits = vocabulary_logits(model, output.last_hidden_state[0, -1])
        logp = logits.double().log_softmax(-1)
    first, second = candidate_ids
    result = dict(correct_first_logp=float(logp[first]), wrong_first_logp=float(logp[second]),
                  margin=float(logp[first] - logp[second]), top1=int(logits.argmax()),
                  reconstruction_max=max(run.reconstruction) if run.reconstruction else None, **run.patch_record)
    return result, run


def state_comparison(left, right):
    rows = []
    for key in left:
        layer, site = key.split('_')
        first, second = left[key], right[key]
        cosine = float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))
        rows.append(dict(layer=int(layer), site=site, cosine=cosine,
                         difference_norm=float(np.linalg.norm(first - second)),
                         correct_norm=float(np.linalg.norm(first)),
                         wrong_norm=float(np.linalg.norm(second))))
    return rows


def sequence_logp(model, probe, side, patch=None):
    """Score the complete fixed candidate; patch only the original branch query."""
    prefix_length = len(probe['prefix_ids'])
    candidate = probe['variants'][side][prefix_length:]
    ids = probe['variants'][side][:-1]
    if patch is not None:
        patch = dict(patch, query=prefix_length-1)
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.inference_mode(), FlowCapture(model, len(ids), probe['groups'],
                                            probe['candidate_ids'], patch, collect=False):
        output = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                             use_cache=False, return_dict=True)
        states = output.last_hidden_state[0, prefix_length-1:]
        logp = vocabulary_logits(model, states).double().log_softmax(-1)
        targets = torch.tensor(candidate, device=logp.device)
        selected = logp[torch.arange(len(candidate), device=logp.device), targets]
    return float(selected.sum())
