"""Checks that affect the meaning of Fisher sketches and functional graphs."""
import numpy as np
import torch

from experiments.decision_risk_flow.geometry import describe
from experiments.decision_risk_flow.kernel import fisher_seeds, node_addresses


def test_fisher_seeds_annihilate_constant_logit_shifts_and_match_metric():
    probability = torch.tensor([[.1, .2, .3, .4]])
    seeds = fisher_seeds(probability, 8192)[:, 0]
    torch.testing.assert_close(seeds.sum(-1), torch.zeros(8192), atol=1e-8, rtol=0)
    metric = torch.diag(probability[0]) - probability.T @ probability
    torch.testing.assert_close(seeds.T @ seeds, metric, atol=.006, rtol=.04)


def test_window_addresses_preserve_self_identity_and_special_priority():
    nodes, windows = node_addresses([1, 2, 3, 4, 5], torch.tensor([2, 4]), 3, [1])
    assert windows == 1
    assert nodes[0, 0].item() == 3  # Special takes priority over prompt.
    assert nodes[0, -1].item() == 0  # First query self is still a prompt key.
    assert nodes[1, -1].item() == 1  # Later query self is prior answer content.


def test_response_geometry_is_invariant_to_output_coordinate_rotation():
    rng = np.random.default_rng(10)
    sketch = rng.normal(size=(2, 9, 4))
    choice = rng.normal(size=(2, 9))
    attention = rng.uniform(size=(2, 9))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    original, _ = describe(choice, sketch, attention, 2)
    rotated, _ = describe(choice, sketch @ rotation, attention, 2)
    for key in original:
        if not key.startswith('shuffled_'):
            np.testing.assert_allclose(original[key], rotated[key], atol=1e-10)
