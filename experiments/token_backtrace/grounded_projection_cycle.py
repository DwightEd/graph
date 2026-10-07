"""Query-conditioned source/history cycles, rather than transported node scores."""
import torch
from torch.nn.functional import normalize


def rewire_indices(target, count, device, generator):
    mapping = torch.arange(count, device=device)
    distance = target - torch.arange(target, device=device)
    for lower, upper in ((2, 4), (4, 8), (8, 16), (16, 32), (32, count + 1)):
        indices = torch.where((distance >= lower) & (distance < upper))[0]
        mapping[indices] = indices[torch.randperm(len(indices), generator=generator, device=device)]
    return mapping


@torch.no_grad()
def cycle_transport(attention, values, source, kernel, prompt, batch=4):
    """C_i(q) uses current-query AND past-carrier source paths, with full V.

    A normalized cycle A(q,s)A(i,s)K(token_i,s) defines source V for that
    particular historical edge. Edges with i>=t have exactly zero read weight.
    """
    count = kernel.shape[0]
    carrier = attention[1:, :, source].permute(1, 0, 2)
    carrier_kernel = carrier * kernel[None]
    source_values = values[source]
    original = values[prompt:].transpose(0, 1)
    norm = original.norm(dim=-1, keepdim=True)
    confidence = carrier.sum(-1).clamp(0, 1)[..., None]
    outputs = {name: [] for name in ('graph', 'rewired')}
    generator = torch.Generator(device=attention.device).manual_seed(73)
    for begin in range(0, count, batch):
        query = attention[begin:begin + batch, :, source]
        weights = query[:, :, None] * carrier_kernel[None]
        weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
        projected = torch.einsum('bhis,shd->bhid', weights, source_values)
        delta = confidence[None] * (normalize(projected, dim=-1) * norm[None] - original[None])
        history = attention[begin:begin + batch, :, prompt:]
        outputs['graph'].append(torch.einsum('bhi,bhid->bhd', history, delta).cpu())
        for offset in range(len(query)):
            mapping = rewire_indices(begin + offset, count, attention.device, generator)
            outputs['rewired'].append(torch.einsum('hi,hid->hd', history[offset], delta[offset, :, mapping]).cpu()[None])
    return {name: torch.cat(rows) for name, rows in outputs.items()}
