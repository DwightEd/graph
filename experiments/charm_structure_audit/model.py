"""Original CHARM: project nodes, send messages, update nodes, predict.

Parameter names/shapes match the saved parent checkpoint. Only explicitly
selected ablations change computation. Graph inputs contain no labels.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def degree(graph, normalization='in'):
    endpoint = 0 if normalization == 'out' else 1
    counts = np.bincount(graph['edge_index'][endpoint], minlength=len(graph['x']))
    return np.maximum(counts, 1).astype(np.float32)


class CharmMP(nn.Module):
    def __init__(self, hidden, edge_dim, residual=True):
        super().__init__()
        self.residual = residual
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden + edge_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.up_mlp = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def messages(self, sender, graph, start, stop, ablation):
        source = torch.as_tensor(graph['edge_index'][0, start:stop], device=sender.device)
        neighbor = sender[source]
        edge = torch.as_tensor(graph['edge_attr'][start:stop], device=sender.device, dtype=sender.dtype)
        mark = torch.as_tensor(graph['edge_mark'][start:stop], device=sender.device, dtype=sender.dtype)
        if ablation == 'no_source':
            neighbor = torch.zeros_like(neighbor)
        if ablation == 'no_edge':
            edge = torch.zeros_like(edge)
        if ablation == 'no_mark':
            mark = torch.zeros_like(mark)
        return self.msg_mlp(torch.cat((neighbor, edge, mark), dim=-1))

    def aggregate(self, sender, graph, chunk, ablation):
        aggregated = torch.zeros_like(sender)
        if ablation == 'no_graph':
            return aggregated
        for start in range(0, graph['edge_index'].shape[1], chunk):
            stop = start + chunk
            target = torch.as_tensor(graph['edge_index'][1, start:stop], device=sender.device)
            def compute(values, left=start, right=stop):
                return self.messages(values, graph, left, right, ablation)
            message = checkpoint(compute, sender, use_reentrant=False) if self.training else compute(sender)
            aggregated.index_add_(0, target, message)
        return aggregated

    def update(self, state, message, residual=True):
        update = self.up_mlp(torch.cat((state, message), dim=-1))
        return (state + update if self.residual and residual else update).relu()


class CHARM(nn.Module):
    def __init__(self, base_dim, edge_dim, hp, normalization='in', edge_chunk=4096):
        super().__init__()
        hidden = int(hp['hidden_dim'])
        self.normalization = normalization
        self.edge_chunk = edge_chunk
        self.in_proj = nn.Linear(base_dim, hidden)
        self.mp_layers = nn.ModuleList([
            CharmMP(hidden, edge_dim, hp['residual_mp']) for _ in range(hp['gnn_layers'])])
        self.pred = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(hidden // 2, 1))

    def forward(self, graph, ablation='full', divisor=None):
        device = self.in_proj.weight.device
        attributes = torch.as_tensor(graph['x'], dtype=torch.float32, device=device)
        if ablation == 'no_node':
            attributes = torch.zeros_like(attributes)
        state = self.in_proj(attributes).relu()
        own_state = state
        counts = degree(graph, self.normalization) if divisor is None else divisor
        counts = torch.as_tensor(counts, dtype=state.dtype, device=device)[:, None]
        layers = self.mp_layers[:1] if ablation == 'one_layer' else self.mp_layers
        for layer in layers:
            # no_relay still permits each source's own deep MLP transformations.
            sender = own_state if ablation == 'no_relay' else state
            message = layer.aggregate(sender, graph, self.edge_chunk, ablation) / counts
            state = layer.update(state, message, residual=ablation != 'no_residual')
            if ablation == 'no_relay':
                own_state = layer.update(own_state, torch.zeros_like(own_state))
        return self.pred(state).view(-1)


def load_checkpoint(path, device='cpu', normalization='in', edge_chunk=4096):
    """Load the project's saved model_state/hp or upstream model_state/best_hp."""
    from numpy.core.multiarray import scalar
    allowed = [np.dtype, (scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar')]
    allowed += [type(np.dtype(t)) for t in ('float16', 'float32', 'float64', 'int32', 'int64', 'bool')]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['model_state']
    hp = saved['hp'] if 'hp' in saved else saved['best_hp']
    hp = dict(hp, residual_mp=hp.get('residual_mp', True))
    node_dim = state['in_proj.weight'].shape[1]
    edge_dim = state['mp_layers.0.msg_mlp.0.weight'].shape[1] - hp['hidden_dim'] - 2
    model = CHARM(node_dim, edge_dim, hp, normalization, edge_chunk)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), hp
