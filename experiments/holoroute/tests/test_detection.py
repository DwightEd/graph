import numpy as np

from experiments.holoroute.config import DetectionConfig
from experiments.holoroute.detection import ConditionalReference


def test_conditional_reference_round_trip():
    random = np.random.default_rng(7)
    residual = random.normal(size=(40, 1)).astype(np.float32)
    condition = random.normal(size=(40, 7)).astype(np.float32)
    task = np.repeat("QA", 40)

    reference = ConditionalReference.fit(residual, condition, task, DetectionConfig())
    score, standardized = reference.transform(residual, condition, task)
    restored = ConditionalReference.from_arrays(reference.arrays())
    restored_score, restored_standardized = restored.transform(residual, condition, task)

    assert score.shape == (40,)
    assert standardized.shape == residual.shape
    assert np.allclose(score, restored_score)
    assert np.allclose(standardized, restored_standardized)
