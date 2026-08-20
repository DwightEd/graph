import numpy as np

from experiments.attention_phenomenology.hypotheses import (
    FAMILY_NAMES,
    FEATURE_NAMES,
)
from experiments.attention_phenomenology.reference import (
    PhenomenologyReference,
    family_atypicality,
    family_layer_atypicality,
    robust_center_scale,
    standardize_features,
)


def test_reference_scale_has_an_explicit_numerical_floor():
    values = np.asarray([[0.0], [0.0], [1e-7]], dtype=np.float32)

    _, scale = robust_center_scale(values, minimum_scale=1e-3)

    np.testing.assert_array_equal(scale, np.asarray([1e-3], dtype=np.float32))


def test_reference_keeps_layer_resolved_family_scores():
    layers = 2
    features = len(FEATURE_NAMES)
    reference = PhenomenologyReference(
        task=np.asarray(["__all__"]),
        bucket=np.asarray([0], dtype=np.int16),
        center=np.zeros((1, layers, features), dtype=np.float32),
        scale=np.ones((1, layers, features), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        family_names=np.asarray(FAMILY_NAMES),
        config_json="{}",
    )
    values = np.ones((3, layers, features), dtype=np.float32)
    standardized = standardize_features(
        values,
        task="QA",
        buckets=np.asarray([0, 1, 2], dtype=np.int16),
        reference=reference,
    )
    layer_scores = family_layer_atypicality(standardized)
    scores = family_atypicality(layer_scores)
    assert layer_scores.shape == (3, layers, len(FAMILY_NAMES))
    assert scores.shape == (3, len(FAMILY_NAMES))


def test_standardized_features_are_bounded_for_stable_family_scores():
    layers = 1
    features = len(FEATURE_NAMES)
    reference = PhenomenologyReference(
        task=np.asarray(["__all__"]),
        bucket=np.asarray([0], dtype=np.int16),
        center=np.zeros((1, layers, features), dtype=np.float32),
        scale=np.ones((1, layers, features), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        family_names=np.asarray(FAMILY_NAMES),
        config_json="{}",
    )

    standardized = standardize_features(
        np.full((1, layers, features), 100.0, dtype=np.float32),
        task="QA",
        buckets=np.asarray([0], dtype=np.int16),
        reference=reference,
        maximum_value=10.0,
    )

    assert standardized.max() == 10.0
