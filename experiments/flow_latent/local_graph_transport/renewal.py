"""Entropy barriers that weaken the existing token continuity penalty.

A high-entropy decision is an uncertainty cue, not a hallucination label. The
barrier preserves each token's own unary and also cuts edges that skip over it.
"""
import numpy as np
import torch

from .smooth import normalize_edge_weights, smooth_field
from .unlabeled import local_graph_edges


GATE_SCALE = 4.


def entropy_events(entropy_rank):
    """Map the upper half of a frozen fit CDF to uncertainty strength [0,1]."""
    return np.maximum(0., 2. * np.asarray(entropy_rank, dtype=np.float64) - 1.)


def crossing_entropy_gate(events, edge_sources, edge_targets, scale=GATE_SCALE):
    """Return exp(-scale*max(E[j+1:t+1])) for every strict-past edge j->t.

    Only events available by the receiving token enter a gate. Excluding the
    sender's event lets a new segment retain internal continuity after its
    entry, while long edges crossing that entry remain attenuated.
    """
    values = np.asarray(events, dtype=np.float64)
    source = np.asarray(edge_sources)
    target = np.asarray(edge_targets)
    maximum = np.zeros(len(source), dtype=np.float64)
    for lag in range(1, int((target - source).max(initial=0)) + 1):
        eligible = target - source >= lag
        maximum[eligible] = np.maximum(maximum[eligible], values[target[eligible] - lag + 1])
    return np.exp(-scale * maximum)


def gated_graph(weights, events):
    """Budget native weights first; gates never amplify or renormalize them."""
    sources, targets, raw = local_graph_edges(weights)
    budgeted = normalize_edge_weights(sources, targets, raw, len(weights))
    gates = torch.from_numpy(crossing_entropy_gate(events, sources.numpy(), targets.numpy()))
    return sources, targets, budgeted * gates


def renewal_field(unary, weights, events):
    """Use the unchanged Huber solver on the entropy-attenuated native edges."""
    sources, targets, gated = gated_graph(weights, events)
    result = smooth_field(torch.from_numpy(np.asarray(unary, dtype=np.float64)),
                          sources, targets, gated, penalty=.5, huber_delta=1.)
    return result['solution'].numpy(), result
