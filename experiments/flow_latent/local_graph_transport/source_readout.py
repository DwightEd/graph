"""Source-program compatibility, with no natural hallucination training labels.

The feature encoder is a statistical full-coordinate reader. Its source-program
targets determine a compatibility direction; source access alone does not prove
truth. Supplying an observed natural answer token is offline candidate scoring,
even when its internal node was captured before that token was input.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .readout import LocalTransportReader


class SourceCompatibilityReader(LocalTransportReader):
    """Return high risk for a candidate incompatible with the source program."""

    def __init__(self, model_dim=4096, heads=32, head_dim=128, scalar_dim=8,
                 dropout=.1, temperature=16., use_bias=True, use_gate=True):
        super().__init__(model_dim, heads, head_dim, scalar_dim, dropout, use_gate)
        del self.classifier
        endpoint_dim = 128 + scalar_dim
        self.semantic_projection = nn.Linear(endpoint_dim, model_dim)
        self.node_semantic_weights = nn.Parameter(torch.zeros(6, model_dim))
        self.compatibility_bias = nn.Linear(endpoint_dim, 1, bias=False) if use_bias else None
        self.register_buffer('temperature', torch.tensor(float(temperature)))

    def semantic_vector(self, embedding, node_fields):
        """Keep a direct full-D residual-basis path alongside learned graph modes.

        Boundary heads are pre-W_O coordinates, so they receive no residual-
        basis shortcut. The six native/delta node sites retain coordinate IDs.
        """
        direct = torch.einsum('mfd,fd->md', node_fields, self.node_semantic_weights)
        return self.semantic_projection(embedding) + direct

    def compatibility(self, embedding, candidate_vectors, node_fields):
        """Full-D cosine with fixed native unembedding candidate rows, not token IDs."""
        semantic = F.normalize(self.semantic_vector(embedding, node_fields), dim=-1)
        candidates = F.normalize(candidate_vectors, dim=-1)
        result = self.temperature * (semantic * candidates).sum(-1)
        if self.compatibility_bias is not None:
            result = result + self.compatibility_bias(embedding).squeeze(-1)
        return result

    def forward(self, node_fields, boundary_fields, value_fields, local_attention,
                indices, valid, scalars, candidate_vectors, variant='real',
                rows=None, return_embedding=False):
        """Use BCE targets 1 minus source-program compatibility labels.

        Candidate vectors [M,D] align to selected rows [M], or to every graph
        row when rows is absent. Node/edge computation still uses the full graph.
        Program and natural-transfer scalar placeholders are zero, so unseen
        scalar columns cannot affect transfer. No natural label enters here.
        """
        nodes = self.node_embedding(node_fields, boundary_fields)
        graph = self.transport_embedding(nodes, value_fields, local_attention,
                                          indices, valid, variant)
        embedding = torch.cat([nodes, graph, scalars], dim=-1)
        if rows is not None:
            embedding = embedding[rows]
            node_fields = node_fields[rows]
        risk_logits = -self.compatibility(embedding, candidate_vectors, node_fields)
        if return_embedding:
            return risk_logits, embedding
        return risk_logits
