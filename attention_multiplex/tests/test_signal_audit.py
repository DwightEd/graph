import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "attention_multiplex"
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attention_multiplex.signal_audit import (  # noqa: E402
    apply_position_reference,
    extract_sample_features,
    fit_position_reference,
)


class SignalAuditTests(unittest.TestCase):
    def _arrays(self, seed=3):
        rng = np.random.default_rng(seed)
        layers, heads, prompt, response, rank = 4, 3, 2, 4, 3
        tokens = prompt + response
        return {
            "response_idx": np.asarray(prompt),
            "token_ids": np.arange(tokens),
            "mass_query_by_layer": rng.normal(size=(layers, response, rank)),
            "mass_source_by_head": rng.normal(size=(heads, tokens, rank)),
            "shape_query_by_layer": rng.normal(size=(layers, response, rank)),
            "shape_source_by_head": rng.normal(size=(heads, tokens, rank)),
            "self_attention": rng.uniform(size=(layers, heads, response)),
            "unresolved_row_mass": rng.uniform(size=(layers, heads, response)),
        }

    def test_features_are_invariant_to_joint_spectral_rotation(self):
        arrays = self._arrays()
        baseline = extract_sample_features(arrays)
        rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))
        rotated = dict(arrays)
        for prefix in ("mass", "shape"):
            rotated[f"{prefix}_query_by_layer"] = (
                arrays[f"{prefix}_query_by_layer"] @ rotation
            )
            rotated[f"{prefix}_source_by_head"] = (
                arrays[f"{prefix}_source_by_head"] @ rotation
            )
        transformed = extract_sample_features(rotated)
        self.assertEqual(set(baseline), set(transformed))
        for name in baseline:
            self.assertTrue(
                np.allclose(
                    baseline[name], transformed[name], atol=1e-5, equal_nan=True
                ),
                name,
            )

    def test_feature_set_avoids_retained_unresolved_duplicate(self):
        features = extract_sample_features(self._arrays())
        self.assertTrue(any(name.startswith("unresolved_mass") for name in features))
        self.assertFalse(any(name.startswith("retained_row_mass") for name in features))
        self.assertTrue(
            np.isnan(features["mass_history_route_per_source_mean"][0])
        )

    def test_position_reference_uses_train_bins(self):
        position = np.tile(np.linspace(0.0, 1.0, 100), 4)
        matrix = np.column_stack((10.0 * position, np.sin(position)))
        reference = fit_position_reference(matrix, position, bins=10, minimum=10)
        adjusted = apply_position_reference(matrix, position, reference)
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertLess(abs(float(np.median(adjusted[:, 0]))), 0.1)


if __name__ == "__main__":
    unittest.main()
