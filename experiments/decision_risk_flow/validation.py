"""Native finite-gate witness; automatic selection, no hallucination annotations."""
import torch

from .data import inputs, write_json
from .kernel import contract_edges, fisher_seeds
from .native import prefill, replay


def validate_gate(model, record, output):
    prompt, response = inputs(record)
    tokens = prompt + response['answer_ids'][:-1]
    target = min(10, len(response['answer_ids']) - 1)
    position = torch.tensor([len(prompt) - 1 + target], device=model.device)
    cache, _, _ = prefill(model, prompt, response['answer_ids'])
    final, _, capture = replay(model, cache, tokens, position)
    logits = model.lm_head(final)
    actual = response['answer_ids'][target]
    ranking = logits[0].topk(2).indices.tolist()
    alternative = ranking[1] if ranking[0] == actual else ranking[0]
    margin = logits[0, actual] - logits[0, alternative]
    layer = len(capture.writes) // 2
    gradient = torch.autograd.grad(margin, capture.writes[layer], retain_graph=True)[0]
    edges = contract_edges(gradient, capture.messages[layer], model.model.layers[layer].self_attn.o_proj.weight)[0]
    masked = edges[:, :len(prompt)].abs().clone()
    special = torch.isin(torch.tensor(prompt, device=model.device),
                         torch.tensor(response['special_ids'], device=model.device))
    masked[:, special] = -1
    head, key = divmod(int(masked.argmax()), len(prompt))
    expected_margin = float(edges[head, key])
    seeds = fisher_seeds(logits.detach().softmax(-1), 16)
    projected = []
    for index, seed in enumerate(seeds):
        gradient = torch.autograd.grad((logits * seed).sum(), capture.writes[layer],
                                      retain_graph=index + 1 < len(seeds))[0]
        edge = contract_edges(gradient, capture.messages[layer], model.model.layers[layer].self_attn.o_proj.weight)
        projected.append(edge[0, head, key].detach())
    projected = torch.stack(projected)
    effects = []
    for epsilon in (.02, .05):
        changed = []
        with torch.no_grad():
            for scale in (1 - epsilon, 1 + epsilon):
                result, _, _ = replay(model, cache, tokens, position,
                    gate=dict(layer=layer, head=head, key=key, scale=scale))
                changed.append(model.lm_head(result))
        finite = (changed[1] - changed[0]) / (2 * epsilon)
        finite_margin = float(finite[0, actual] - finite[0, alternative])
        projected_finite = (seeds * finite).sum((1, 2))
        relative = float((projected_finite - projected).norm() / projected.norm().clamp_min(1e-10))
        probability = logits.detach().softmax(-1)
        centered = finite - (finite * probability).sum(-1, keepdim=True)
        exact_energy = float((probability * centered.square()).sum())
        effects.append(dict(epsilon=epsilon, finite_margin=finite_margin,
            expected_margin=expected_margin, projected_relative_error=relative,
            full_fisher_finite_energy=exact_energy,
            sketch8_energy=float(projected[:8].square().sum() * 2),
            sketch16_energy=float(projected.square().sum())))
    result = dict(id=record['id'], token=target, layer=layer, head=head, key=key,
        selection='largest absolute native margin response among nonspecial prompt edges at middle layer',
        effects=effects, labels_read=False)
    write_json(output / 'native_gate_validation.json', result)
    if min(item['projected_relative_error'] for item in effects) > .05:
        raise ValueError('Native finite-gate response does not match the predicted derivative')
    return result
