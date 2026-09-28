"""Native window-gate responses; no labels or manually selected source regions."""
import math

import torch

WINDOW = 16
GROUPS = ('prompt', 'recent', 'remote', 'special', 'mlp')


def fisher_seeds(probability, rank, seed=42):
    """Rows of R C, with C.T C = diag(p)-p p.T and E[R.T R]=I."""
    generator = torch.Generator(device=probability.device).manual_seed(seed)
    signs = torch.randint(0, 2, (rank, probability.shape[-1]),
                          generator=generator, device=probability.device).float()
    signs = (2 * signs - 1) / math.sqrt(rank)
    weighted = probability.sqrt()[None] * signs[:, None]
    return weighted - probability[None] * weighted.sum(-1, keepdim=True)


def node_addresses(tokens, positions, prompt_length, special_ids):
    """Future cached keys have zero attention; the appended key is query self."""
    keys = torch.arange(len(tokens), device=positions.device)
    keys = keys.expand(len(positions), -1)
    keys = torch.cat((keys, positions[:, None]), -1)
    ids = torch.tensor(tokens, device=positions.device)[keys]
    groups = torch.where(keys < prompt_length, 0, 2)
    recent = (keys >= prompt_length) & (keys >= positions[:, None] - WINDOW + 1)
    groups = torch.where(recent, 1, groups)
    special = torch.isin(ids, torch.tensor(special_ids, device=positions.device))
    groups = torch.where(special, 3, groups)
    windows = math.ceil(len(tokens) / WINDOW)
    nodes = groups * windows + keys // WINDOW
    return nodes, windows


def contract_edges(gradient, message, output_weight):
    """Return [batch,head,key] gate derivatives, including recomputed self V."""
    attention, past_value, self_value = message
    heads, _, head_dim = past_value.shape
    weights = output_weight.reshape(-1, heads, head_dim).permute(1, 0, 2)
    direction = torch.einsum('bd,hdk->hbk', gradient[0].float(), weights.float())
    past = torch.einsum('hbd,hkd->hbk', direction, past_value.float())
    own = (direction * self_value.float()).sum(-1, keepdim=True)
    return (torch.cat((past, own), -1) * attention.float()).permute(1, 0, 2)


def collect_response(model, capture, gradients, nodes, windows):
    """Sum heads/layers within predefined window gates, retaining two depth bands."""
    batch = nodes.shape[0]
    count = 4 * windows + 1
    result = torch.zeros(batch, 2, count, device=nodes.device)
    layers = len(capture.writes)
    for index, layer in enumerate(model.model.layers):
        band = index // (layers // 2)
        edges = contract_edges(gradients[index], capture.messages[index],
                               layer.self_attn.o_proj.weight)
        result[:, band].scatter_add_(1, nodes, edges.sum(1))
        mlp = (gradients[layers + index].float() * capture.mlp_writes[index].detach().float()).sum(-1)[0]
        result[:, band, -1] += mlp
    return result


def collect_attention(capture, nodes, windows):
    result = torch.zeros(nodes.shape[0], 2, 4 * windows + 1, device=nodes.device)
    layers = len(capture.writes)
    for index, (attention, _, _) in enumerate(capture.messages):
        result[:, index // (layers // 2)].scatter_add_(1, nodes,
            attention.float().mean(0) / (layers // 2))
    return result


def batch_responses(model, capture, logits, actual, alternative, nodes, windows, rank):
    margin = logits.gather(1, actual[:, None])[:, 0] - logits.gather(1, alternative[:, None])[:, 0]
    writes = capture.writes + capture.mlp_writes
    gradients = torch.autograd.grad(margin.sum(), writes, retain_graph=True)
    choice = collect_response(model, capture, gradients, nodes, windows)
    seeds = fisher_seeds(logits.detach().softmax(-1), rank)
    projected = []
    for index, seed in enumerate(seeds):
        gradients = torch.autograd.grad((logits * seed).sum(), writes,
                                        retain_graph=index + 1 < rank)
        projected.append(collect_response(model, capture, gradients, nodes, windows))
    sketch = torch.stack(projected, -1)
    return choice.detach(), sketch.detach(), collect_attention(capture, nodes, windows)
