import torch

from experiments.attention_phenomenology.geometry import (
    hellinger_distance_matrix,
    zero_dimensional_persistence,
)


def test_h0_persistence_detects_two_head_coalitions():
    coherent = torch.tensor(
        [[[0.50, 0.50], [0.49, 0.51], [0.51, 0.49], [0.50, 0.50]]]
    )
    fractured = torch.tensor(
        [[[0.99, 0.01], [0.98, 0.02], [0.01, 0.99], [0.02, 0.98]]]
    )
    coherent_deaths = zero_dimensional_persistence(
        hellinger_distance_matrix(coherent)
    )
    fractured_deaths = zero_dimensional_persistence(
        hellinger_distance_matrix(fractured)
    )
    assert fractured_deaths.max() > coherent_deaths.max()
