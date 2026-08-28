import pytest

from ..binding import validate_exact_attention_against_cache
from .helpers import FakeAttentionCache, synthetic_bundle


def test_exact_dense_attention_is_strictly_bound_to_sparse_cache():
    bundle = synthetic_bundle(seed=23)
    cache = FakeAttentionCache(bundle.capture)
    result = validate_exact_attention_against_cache(
        cache,
        [layer.attention for layer in bundle.capture.layers],
        absolute_tolerance=1e-6,
    )
    assert result.verified
    assert result.retained_max_abs_error == 0.0
    assert result.diagonal_max_abs_error == 0.0
    assert result.censored_max_probability <= cache.attention_floor + 1e-6


def test_cache_binding_rejects_a_different_attention_tensor():
    bundle = synthetic_bundle(seed=29)
    cache = FakeAttentionCache(bundle.capture)
    changed = [layer.attention.clone() for layer in bundle.capture.layers]
    changed[0][0, 0, 0] += 0.2
    with pytest.raises(ValueError):
        validate_exact_attention_against_cache(
            cache,
            changed,
            absolute_tolerance=1e-6,
        )
