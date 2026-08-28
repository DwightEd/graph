import torch

from ..math_utils import effective_number


def test_zero_mass_has_zero_effective_sources_not_one_fake_source():
    mass = torch.zeros(3, 5)
    assert torch.equal(effective_number(mass), torch.zeros(3))
