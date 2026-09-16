"""Checkpoint-compatible CHARM algebra, without a new detector or PyG dependency.

Reference: train_charm_grid.py at 13e907693aa954bf070e809d8afecdf26b3b88d8.
MLP names/shapes, residual, ReLU and prediction dropout match its active code.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graph import degree


class CharmMP(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, residual=True):
        super().__init__()
        self.residual = residual
        self.msg_mlp = nn.Sequential(nn.Linear(node_dim + edge_dim + 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.up_mlp = nn.Sequential(nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, node_dim))

    def forward(self, x, graph, divisor, edge_chunk):
        device = x.device
        aggregated = torch.zeros_like(x)
        source, target = graph["edge_index"]
        for start in range(0, len(source), edge_chunk):
            stop = start + edge_chunk
            dst = torch.as_tensor(target[start:stop], dtype=torch.long, device=device)
            def compute(values, lo=start, hi=stop):
                src = torch.as_tensor(source[lo:hi], dtype=torch.long, device=device)
                attr = torch.as_tensor(graph["edge_attr"][lo:hi], dtype=values.dtype, device=device)
                mark = torch.as_tensor(graph["edge_mark"][lo:hi], dtype=values.dtype, device=device)
                return self.msg_mlp(torch.cat((values[src], attr, mark), dim=-1))
            # Recreate edge inputs during backward rather than retaining E*C on GPU.
            message = checkpoint(compute, x, use_reentrant=False) if self.training else compute(x)
            aggregated.index_add_(0, dst, message)
        update = self.up_mlp(torch.cat((x, aggregated / divisor[:, None]), dim=-1))
        return x + update if self.residual else update


class CHARM(nn.Module):
    def __init__(self, base_dim, edge_dim, hp, normalization="in", edge_chunk=4096):
        super().__init__()
        hidden = int(hp["hidden_dim"])
        self.normalization, self.edge_chunk = normalization, edge_chunk
        self.in_proj = nn.Linear(base_dim, hidden)
        self.mp_layers = nn.ModuleList([
            CharmMP(hidden, edge_dim, hidden, hp.get("residual_mp", True))
            for _ in range(int(hp["gnn_layers"]))
        ])
        self.pred = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, graph, return_hidden=False, divisor=None):
        device = self.in_proj.weight.device
        h = self.in_proj(torch.as_tensor(graph["x"], dtype=torch.float32, device=device)).relu()
        d = degree(graph, self.normalization) if divisor is None else divisor
        d = torch.as_tensor(d, dtype=torch.float32, device=device)
        for layer in self.mp_layers:
            h = layer(h, graph, d, self.edge_chunk).relu()
        logits = self.pred(h).view(-1)
        return (logits, h) if return_hidden else logits


def load_checkpoint(path, device="cpu", normalization="out", edge_chunk=4096):
    """Read original grid_best_model.pt or original/train.py checkpoint.pt.

    weights_only=True: do not execute pickled repository classes.
    A strict state_dict load fails for GNN_train.py / hypergraph / other models.
    """
    # Original checkpoints may include sklearn's NumPy scalar test metrics.
    # Allow only numeric scalar reconstruction, not arbitrary checkpoint classes.
    from numpy.core.multiarray import scalar
    allowed = [np.dtype, (scalar, "numpy.core.multiarray.scalar"),
               (scalar, "numpy._core.multiarray.scalar")]
    allowed += [type(np.dtype(t)) for t in ("float16", "float32", "float64", "int32", "int64", "bool")]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("model_state", saved.get("state_dict", saved))
    hp = dict(saved.get("best_hp", saved.get("hyperparameters", saved.get("hp", {}))))
    hp.setdefault("hidden_dim", state["in_proj.weight"].shape[0])
    hp.setdefault("gnn_layers", len({k.split('.')[1] for k in state if k.startswith('mp_layers.')}))
    hp.setdefault("residual_mp", True)
    base_dim = state["in_proj.weight"].shape[1]
    edge_dim = state["mp_layers.0.msg_mlp.0.weight"].shape[1] - hp["hidden_dim"] - 2
    model = CHARM(base_dim, edge_dim, hp, normalization, edge_chunk)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, hp


def smooth_scores(scores, beta=.5):
    """Fixed past-only EWMA; no gold spans or test-fit parameters."""
    result = np.asarray(scores).copy()
    for t in range(1, len(result)):
        result[t] = (1 - beta) * result[t] + beta * result[t - 1]
    return result
