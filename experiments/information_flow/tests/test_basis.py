import pytest
import torch

from experiments.information_flow.basis import source_basis


@pytest.mark.parametrize("dimensions", range(1, 8))
def test_source_basis_honors_every_requested_dimension(dimensions):
    assert source_basis(9, 5, dimensions).shape == (9, dimensions)


def test_source_basis_is_shared_and_normalized():
    first = source_basis(9, 5, 12)
    second = source_basis(9, 5, 12)
    longer = source_basis(19, 5, 12)

    assert first.shape == (9, 12)
    assert torch.equal(first, second)
    assert torch.equal(first, longer[:9])
    assert torch.allclose(first.norm(dim=-1), torch.ones(9), atol=1e-6)
    assert bool((first[:5, 0] > 0).all())
    assert torch.equal(first[:5, 1], torch.zeros(5))
    assert torch.equal(first[5:, 0], torch.zeros(4))
    assert bool((first[5:, 1] > 0).all())
