"""Label-free graph autoencoder."""

import torch
from torch import nn


class GraphAutoencoder:
    """Encode local topology and reconstruct node/edge observables."""

    def __init__(self, input_dim, hidden_dim=64, message_steps=2, epochs=20, learning_rate=1e-3, seed=20260914):
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.epochs = epochs
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.message.parameters()) + list(self.decoder.parameters()) + [self.edge_scale]

    def _encode(self, graph):
        x = torch.as_tensor(graph.x, dtype=torch.float32)
        source, target = torch.as_tensor(graph.edge_index, dtype=torch.long)
        weight = torch.as_tensor(graph.edge_weight, dtype=torch.float32)
        z = self.encoder(x).relu()
        for _ in range(self.message_steps):
            messages = z[source] * weight[:, None]
            aggregated = torch.zeros_like(z).index_add_(0, target, messages)
            z = (z + self.message(aggregated).relu()).relu()
        return z, source, target, weight

    def fit(self, graphs):
        history = []
        for _ in range(self.epochs):
            total = 0.0
            for graph in graphs:
                self.optimizer.zero_grad()
                z, source, target, weight = self._encode(graph)
                node_loss = (self.decoder(z) - torch.as_tensor(graph.x, dtype=torch.float32)).square().mean()
                edge_prediction = self.edge_scale * (z[source] * z[target]).sum(-1).sigmoid()
                edge_loss = (edge_prediction - weight).square().mean() if len(weight) else node_loss * 0
                loss = node_loss + edge_loss
                loss.backward()
                self.optimizer.step()
                total += float(loss.detach())
            history.append(total / len(graphs))
        return history

    def reconstruct(self, graph):
        with torch.no_grad():
            z, source, target, weight = self._encode(graph)
            node_error = (self.decoder(z) - torch.as_tensor(graph.x, dtype=torch.float32)).square().mean(-1)
            edge_prediction = self.edge_scale * (z[source] * z[target]).sum(-1).sigmoid()
            edge_error = (edge_prediction - weight).square()
        return {"node_error": node_error.numpy(), "edge_prediction": edge_prediction.numpy(),
                "edge_error": edge_error.numpy()}
