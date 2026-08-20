from types import SimpleNamespace

import torch

from experiments.attention_phenomenology.compositions import (
    composition_views,
    provenance_composition,
)


def analysis_fixture():
    routing = SimpleNamespace(
        role_probability=torch.tensor([[[[0.2, 0.5, 0.1, 0.2]]]]),
        prompt_mass=torch.tensor([[[0.2]]]),
        response_mass=torch.tensor([[[0.5]]]),
        self_mass=torch.tensor([[[0.1]]]),
        unresolved_mass=torch.tensor([[[0.2]]]),
    )
    provenance = SimpleNamespace(
        head_lower=torch.tensor([[[0.45]]]),
        unsupported_response_lower=torch.tensor([[[0.1]]]),
        aggregate_lower=torch.tensor([[0.0, 0.45]]),
    )
    return SimpleNamespace(routing=routing, provenance=provenance)


def test_provenance_composition_splits_response_mass():
    view = provenance_composition(analysis_fixture())

    expected = torch.tensor([0.2, 0.25, 0.1, 0.15, 0.1, 0.2])
    torch.testing.assert_close(view.values[0, 0, 0], expected)
    torch.testing.assert_close(view.values.sum(dim=-1), torch.ones((1, 1, 1)))


def test_composition_views_include_role_and_graph_derived_provenance():
    views = composition_views(analysis_fixture())

    assert set(views) == {"role", "provenance"}
    assert views["role"].values.shape[-1] == 4
    assert views["provenance"].values.shape[-1] == 6
