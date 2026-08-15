import unittest

import numpy as np
import torch

from attention_graph.causal_topology import TopologyEncoding
from attention_graph.one_class import OneClassConfig
from attention_graph.topology_one_class import TopologyOneClassModel


def _encoding(*, seed: int, provenance_width: int = 4, no_rr: bool = False) -> TopologyEncoding:
    generator = torch.Generator().manual_seed(seed)

    def values(width: int) -> torch.Tensor:
        return torch.randn(24, 1, 2, width, generator=generator)

    rr_one_hop = torch.zeros(24, 1, 2, 6) if no_rr else values(6)
    rr_two_hop = torch.zeros(24, 1, 2, 2) if no_rr else values(2)
    return TopologyEncoding(
        balance_log_scale=values(2),
        attention_marginals=values(3),
        retained_support=values(2),
        prompt_provenance=values(provenance_width),
        rr_one_hop=rr_one_hop,
        rr_two_hop=rr_two_hop,
        rewired_rr_one_hop=rr_one_hop.clone() if no_rr else values(6),
        rewired_rr_two_hop=rr_two_hop.clone() if no_rr else values(2),
    )


class TopologyOneClassModelTests(unittest.TestCase):
    def setUp(self):
        self.config = OneClassConfig(position_bins=1, subspace_components=2, seed=7)
        self.positions = np.linspace(0.0, 1.0, 24, dtype=np.float32)

    def _fit(self, fit: TopologyEncoding, calibration: TopologyEncoding) -> TopologyOneClassModel:
        return TopologyOneClassModel(self.config).fit(
            fit, self.positions, calibration, self.positions
        )

    def test_provenance_noise_cannot_change_retained_marginal_score(self):
        fit = _encoding(seed=1)
        calibration = _encoding(seed=2)
        target = _encoding(seed=3)
        noisy_fit = TopologyEncoding(
            fit.balance_log_scale, fit.attention_marginals, fit.retained_support,
            torch.randn(24, 1, 2, 16), fit.rr_one_hop, fit.rr_two_hop,
            fit.rewired_rr_one_hop, fit.rewired_rr_two_hop,
        )
        noisy_calibration = TopologyEncoding(
            calibration.balance_log_scale, calibration.attention_marginals,
            calibration.retained_support, torch.randn(24, 1, 2, 16),
            calibration.rr_one_hop, calibration.rr_two_hop,
            calibration.rewired_rr_one_hop, calibration.rewired_rr_two_hop,
        )
        noisy_target = TopologyEncoding(
            target.balance_log_scale, target.attention_marginals, target.retained_support,
            torch.randn(24, 1, 2, 16), target.rr_one_hop, target.rr_two_hop,
            target.rewired_rr_one_hop, target.rewired_rr_two_hop,
        )
        plain = self._fit(fit, calibration).transform(target, self.positions)
        noisy = self._fit(noisy_fit, noisy_calibration).transform(noisy_target, self.positions)

        np.testing.assert_allclose(
            plain.scores["attention_marginals"], noisy.scores["attention_marginals"]
        )

    def test_exact_and_lag_rewired_scores_match_without_rr_edges(self):
        fit = _encoding(seed=1, no_rr=True)
        calibration = _encoding(seed=2, no_rr=True)
        target = _encoding(seed=3, no_rr=True)
        scores = self._fit(fit, calibration).transform(target, self.positions).scores

        self.assertEqual(scores["rr_multihop_exact"].tolist(), scores["rr_multihop_lag_rewired"].tolist())
        self.assertEqual(scores["causal_topology_exact"].tolist(), scores["causal_topology_lag_rewired"].tolist())

    def test_score_fields_are_fixed_hierarchical_fusions(self):
        model = self._fit(_encoding(seed=1), _encoding(seed=2))
        result = model.transform(_encoding(seed=3), self.positions)

        self.assertEqual(
            tuple(result.scores),
            (
                "attention_marginals", "retained_support", "balance_scale",
                "prompt_topology", "rr_one_hop_exact", "rr_two_hop_exact",
                "rr_multihop_exact", "rr_multihop_lag_rewired",
                "causal_topology_exact", "causal_topology_lag_rewired", "full_signal",
            ),
        )

    def test_rewired_control_cannot_raise_the_deployment_score(self):
        model = self._fit(_encoding(seed=1), _encoding(seed=2))
        target = _encoding(seed=3)
        altered = TopologyEncoding(
            target.balance_log_scale, target.attention_marginals,
            target.retained_support, target.prompt_provenance,
            target.rr_one_hop, target.rr_two_hop,
            target.rewired_rr_one_hop + 10_000,
            target.rewired_rr_two_hop - 10_000,
        )

        original = model.transform(target, self.positions).scores
        changed = model.transform(altered, self.positions).scores

        np.testing.assert_array_equal(original["full_signal"], changed["full_signal"])
        self.assertFalse(np.array_equal(
            original["causal_topology_lag_rewired"],
            changed["causal_topology_lag_rewired"],
        ))

    def test_loader_fit_requests_one_atomic_block_at_a_time(self):
        fit = atomic = __import__(
            "attention_graph.topology_one_class", fromlist=["atomic_blocks"]
        ).atomic_blocks(_encoding(seed=1))
        calibration = __import__(
            "attention_graph.topology_one_class", fromlist=["atomic_blocks"]
        ).atomic_blocks(_encoding(seed=2))
        calls = {"fit": [], "cal": []}

        def loader(group, values):
            def load(name):
                calls[group].append(name)
                return values[name]
            return load

        bins = np.zeros(24, dtype=np.int16)
        model = TopologyOneClassModel(self.config).fit_loaders(
            tuple(atomic), bins, bins,
            loader("fit", fit), loader("cal", calibration),
        )

        self.assertEqual(calls["fit"], list(atomic))
        self.assertEqual(calls["cal"], list(atomic))
        self.assertIn("full_signal", model.calibration_scores())


if __name__ == "__main__":
    unittest.main()
