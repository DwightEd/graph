"""Source-conditioned reuse edges, with native layer/head identities intact.

The kernel is an explicit similarity assumption, not a causal attribution or a
truth probability. Each head compares fixed native coordinates before its own
attention weights the pair. Candidate ranks and missing-window flags never
enter these comparisons.
"""

import numpy as np
import torch

from .routes import EPS


def head_signatures(arrays, source_count, layers, heads, device):
    """Return [head,token,feature] with three equally weighted distance terms."""
    source = torch.as_tensor(
        arrays["group_effect"][:, layers, heads, :source_count],
        dtype=torch.float32, device=device,
    )
    source_size = source.norm(dim=-1).sum(dim=-1, keepdim=True)
    source = source / source_size.clamp_min(EPS).unsqueeze(-1)

    attention = torch.as_tensor(
        arrays["group_attention"][:, layers, heads],
        dtype=torch.float32, device=device,
    ).sqrt()
    ffn = torch.as_tensor(
        arrays["ffn_state"][:, layers], dtype=torch.float32, device=device,
    )
    ffn = ffn / ffn.norm(dim=-1, keepdim=True).clamp_min(EPS)

    # Concatenation makes squared distance the unweighted sum of the three
    # terms. FFN direction changes affect affinity, never a signed risk term.
    signature = torch.cat((source.flatten(start_dim=-2), attention, ffn), dim=-1)
    return signature.permute(1, 0, 2).contiguous()


def state_kernel(signatures):
    """Use each head's offline median nonzero adjacent squared distance."""
    adjacent = (signatures[:, 1:] - signatures[:, :-1]).square().sum(dim=-1)
    positive = adjacent.masked_fill(adjacent == 0, float("nan"))
    if adjacent.shape[1]:
        bandwidth = torch.nanquantile(positive, .5, dim=-1)
        bandwidth = torch.nan_to_num(bandwidth, nan=1.)
    else:
        bandwidth = torch.ones(len(signatures), device=signatures.device)

    # Direct cdist avoids cancellation that could turn equal states into small
    # positive distances. It materializes [head,T,T], not [head,T,T,feature].
    distance = torch.cdist(
        signatures, signatures, compute_mode="donot_use_mm_for_euclid_dist",
    ).square()
    return torch.exp(-distance / bandwidth[:, None, None]), bandwidth


def query_reuse(arrays, layers, heads, device):
    """Map answer key P+j to its query-state row j+1; remove query self."""
    attention = torch.as_tensor(
        arrays["history_attention"][:, layers, heads],
        dtype=torch.float32, device=device,
    ).permute(1, 0, 2)
    reuse = torch.zeros_like(attention)
    reuse[:, :, 1:] = attention[:, :, :-1]
    return reuse.tril(diagonal=-1)


@torch.no_grad()
def build_graph(arrays, source_count, device="cpu", head_batch=16):
    """Build one answer's graph; only a batch of head kernels is resident."""
    count, layer_count, head_count = arrays["group_attention"].shape[:3]
    total_heads = layer_count * head_count
    edges = torch.zeros((count, count), device=device, dtype=torch.float32)
    actual_reuse = torch.zeros_like(edges)
    bandwidths = np.empty(total_heads, dtype=np.float32)

    for start in range(0, total_heads, head_batch):
        index = np.arange(start, min(start + head_batch, total_heads))
        layers, heads = np.divmod(index, head_count)
        signatures = head_signatures(arrays, source_count, layers, heads, device)
        kernel, bandwidth = state_kernel(signatures)
        reuse = query_reuse(arrays, layers, heads, device)
        edges += (kernel * reuse).sum(dim=0)
        actual_reuse += reuse.sum(dim=0)
        bandwidths[index] = bandwidth.cpu().numpy()

    edges /= total_heads
    actual_reuse /= total_heads
    return {
        "edge_weight": edges.cpu().numpy(),
        "reuse_weight": actual_reuse.cpu().numpy(),
        "degree": (edges.sum(dim=0) + edges.sum(dim=1)).cpu().numpy(),
        "head_bandwidth": bandwidths.reshape(layer_count, head_count),
    }
