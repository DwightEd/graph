import unittest

import torch

from experiments.causal_setflow.config import CorruptionConfig, SourceSetConfig
from experiments.causal_setflow.corruptions import apply_corruption, sample_corruption_plan
from experiments.causal_setflow.data import CausalSourceSetGraph, SparseRRLayer


def _graph():
    layer = SparseRRLayer(
        head=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        query=torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
        source=torch.tensor([0, 0, 0, 1, 1, 0, 2, 1, 3, 2, 4, 3]),
        weight=torch.tensor([0.2, 0.1, 0.3, 0.25, 0.4, 0.2, 0.22, 0.18, 0.35, 0.27, 0.31, 0.29]),
    ).validate(num_heads=2, response_count=7)
    return CausalSourceSetGraph(
        layers=(layer, layer), num_heads=2, response_count=7, attention_floor=0.01
    ).validate()


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.graph = _graph()
        self.source_config = SourceSetConfig(
            max_route_sources=4,
            max_memory_sources=4,
            route_mass_coverage=1.0,
            materialize_query_chunk_size=2,
        )
        self.corruption_config = CorruptionConfig(
            token_span_min=3,
            token_span_max=5,
            layer_span_min=1,
            layer_span_max=2,
            selected_head_fraction=1.0,
        )

    def _apply(self, type_index):
        generator = torch.Generator().manual_seed(100 + type_index)
        plan = sample_corruption_plan(
            7, 2, 2, self.corruption_config,
            device="cpu", generator=generator, forced_type=type_index,
        )
        layer_index = int(torch.nonzero(plan.layer_mask)[0].item())
        source_sets = self.graph.materialize_layer(
            layer_index, self.source_config, device="cpu"
        )
        corrupted, changed = apply_corruption(
            source_sets, plan, layer_index=layer_index, config=self.corruption_config
        )
        return source_sets, corrupted, changed

    def test_every_corruption_remains_causal_and_finite(self):
        for type_index in range(5):
            with self.subTest(type_index=type_index):
                _, corrupted, changed = self._apply(type_index)
                self.assertTrue(bool(changed.any()))
                token = torch.arange(7)[:, None, None]
                self.assertTrue(bool((corrupted.route_source.long()[corrupted.route_mask] < token.expand_as(corrupted.route_source)[corrupted.route_mask]).all()))
                self.assertTrue(bool((corrupted.memory_source.long()[corrupted.memory_mask] < token.expand_as(corrupted.memory_source)[corrupted.memory_mask]).all()))
                for field in (
                    corrupted.route_weight,
                    corrupted.route_received,
                    corrupted.memory_received,
                    corrupted.memory_current_weight,
                ):
                    self.assertTrue(torch.isfinite(field).all())

    def test_collapse_and_self_reinforce_preserve_route_mass(self):
        for type_index in (0, 4):
            original, corrupted, changed = self._apply(type_index)
            before = original.route_weight.sum(dim=-1)
            after = corrupted.route_weight.sum(dim=-1)
            torch.testing.assert_close(after[changed], before[changed], rtol=1e-5, atol=1e-6)

    def test_localization_preserves_weight_multiset(self):
        original, corrupted, changed = self._apply(1)
        for token, head in torch.nonzero(changed, as_tuple=False).tolist():
            torch.testing.assert_close(
                torch.sort(original.route_weight[token, head]).values,
                torch.sort(corrupted.route_weight[token, head]).values,
            )

    def test_freeze_zeroes_selected_received_velocity(self):
        _, corrupted, changed = self._apply(2)
        selected = changed.unsqueeze(-1).expand_as(corrupted.route_received_delta)
        self.assertTrue(torch.allclose(
            corrupted.route_received_delta[selected],
            torch.zeros_like(corrupted.route_received_delta[selected]),
        ))


if __name__ == "__main__":
    unittest.main()