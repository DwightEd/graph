import torch

from experiments.attention_phenomenology.majorization_nulls import (
    shuffle_prompt_source_identity,
    shuffle_prompt_time,
    uniform_prompt_excess,
)
from experiments.attention_phenomenology.routing import RoutingEdges


def edges_fixture():
    return RoutingEdges(
        num_layers=1,
        num_heads=1,
        num_response_tokens=3,
        num_tokens=6,
        response_idx=3,
        attention_floor=0.1,
        layer=torch.zeros(6, dtype=torch.long),
        head=torch.zeros(6, dtype=torch.long),
        query=torch.tensor([0, 0, 1, 1, 2, 2]),
        source=torch.tensor([0, 1, 0, 2, 1, 2]),
        weight=torch.tensor([0.7, 0.3, 0.6, 0.4, 0.8, 0.1]),
        diagonal=torch.zeros((3, 1, 1)),
    )


def test_uniform_null_preserves_support_but_removes_weight_concentration():
    original = edges_fixture()

    uniform = uniform_prompt_excess(original)

    torch.testing.assert_close(uniform.source, original.source)
    torch.testing.assert_close(
        uniform.weight,
        torch.tensor([1.1, 1.1, 1.1, 1.1, 1.1, 0.1]),
    )


def test_identity_and_time_nulls_preserve_their_registered_marginals():
    original = edges_fixture()

    identity = shuffle_prompt_source_identity(original, seed=7)
    time = shuffle_prompt_time(original, seed=8)

    torch.testing.assert_close(identity.query, original.query)
    torch.testing.assert_close(identity.weight, original.weight)
    assert not torch.equal(identity.source, original.source)
    torch.testing.assert_close(time.source, original.source)
    torch.testing.assert_close(time.weight, original.weight)
    assert sorted(time.query.tolist()) == sorted(original.query.tolist())
