import torch

from experiments.attention_phenomenology.majorization import (
    hill_diversity_spectrum,
    majorization_evidence,
)


def test_hill_spectrum_has_expected_uniform_and_point_mass_limits():
    distributions = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    spectrum = hill_diversity_spectrum(distributions)

    torch.testing.assert_close(spectrum[0], torch.full((5,), 4.0))
    torch.testing.assert_close(spectrum[1], torch.ones(5))


def test_majorization_evidence_distinguishes_concentration_from_redistribution():
    concentrated = torch.tensor([[0.8, 0.2, 0.0]])
    distributed = torch.tensor([[0.5, 0.3, 0.2]])

    toward_concentration = majorization_evidence(concentrated, distributed)
    toward_dispersion = majorization_evidence(distributed, concentrated)

    assert toward_concentration.evidence.item() > 0
    assert toward_concentration.violation.item() == 0
    assert toward_dispersion.evidence.item() < 0
    torch.testing.assert_close(
        toward_dispersion.violation,
        torch.tensor([0.3]),
    )
