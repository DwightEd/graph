import numpy as np

from experiments.holoroute.config import DetectionConfig
from experiments.holoroute.detection import SubspaceReference


def test_subspace_reference_round_trip():
    random = np.random.default_rng(7)
    feature = random.normal(size=(80, 24)).astype(np.float32)
    condition = random.normal(size=(80, 6)).astype(np.float32)
    task = np.repeat("QA", 80)
    config = DetectionConfig(pca_components=6)

    reference = SubspaceReference.fit(
        feature[:50],
        condition[:50],
        task[:50],
        config,
    ).calibrate(
        feature[50:65],
        condition[50:65],
        task[50:65],
    )
    score, energy = reference.transform(
        feature[65:],
        condition[65:],
        task[65:],
    )
    restored = SubspaceReference.from_arrays(reference.arrays())
    restored_score, restored_energy = restored.transform(
        feature[65:],
        condition[65:],
        task[65:],
    )

    assert score.shape == (15,)
    assert energy.shape == (15, 1)
    assert np.allclose(score, restored_score)
    assert np.allclose(energy, restored_energy)
