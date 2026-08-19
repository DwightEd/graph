import unittest

import torch

from experiments.causal_setflow.config import (
    CorruptionConfig,
    SetFlowModelConfig,
    SourceSetConfig,
)
from experiments.causal_setflow.corruptions import sample_corruption_plan
from experiments.causal_setflow.data import CausalSourceSetGraph, SparseRRLayer
from experiments.causal_setflow.losses import (
    corrupted_energy_loss,
    pairwise_ranking_loss,
    robust_clean_energy_loss,
    variance_covariance_loss,
)
from experiments.causal_setflow.model import CausalSetFlowModel


def _model_and_graph():
    layer = SparseRRLayer(
        head=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        query=torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        source=torch.tensor([0, 0, 0, 1, 1, 0, 2, 1, 3, 2]),
        weight=torch.tensor([0.2, 0.1, 0.3, 0.25, 0.4, 0.2, 0.22, 0.18, 0.35, 0.27]),
    ).validate(num_heads=2, response_count=6)
    graph = CausalSourceSetGraph(
        layers=(layer, layer), num_heads=2, response_count=6, attention_floor=0.01
    ).validate()
    source = SourceSetConfig(
        max_route_sources=4,
        max_memory_sources=4,
        route_mass_coverage=1.0,
        materialize_query_chunk_size=2,
    )
    config = SetFlowModelConfig(
        hidden_dim=8,
        scalar_fourier_dim=2,
        set_heads=2,
        induced_points=2,
        set_blocks=1,
        head_mixer_layers=1,
        depth_mixer_layers=1,
        set_row_chunk_size=4,
        mixer_token_chunk_size=3,
        activation_checkpointing=False,
        dropout=0.0,
    )
    return CausalSetFlowModel(2, 2, source, config), graph


class MethodTests(unittest.TestCase):
    def test_teacher_has_no_gradient_and_ema_updates(self):
        model, _ = _model_and_graph()
        self.assertTrue(all(not p.requires_grad for p in model.teacher_encoder.parameters()))
        teacher_before = [p.clone() for p in model.teacher_encoder.parameters()]
        with torch.no_grad():
            next(model.online_encoder.parameters()).add_(1.0)
        model.update_teacher(0.5)
        teacher_after = list(model.teacher_encoder.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(teacher_before, teacher_after)))

    def test_energy_and_type_shapes(self):
        model, graph = _model_and_graph()
        model.eval()
        with torch.inference_mode():
            encoded = model.encode_online(graph, device="cpu")
            energy = model.energy(encoded)
        self.assertEqual(energy.general.shape, (6,))
        self.assertEqual(energy.channel_general.shape, (6, 2, 2))
        self.assertEqual(energy.type_energy.shape, (6, 5))
        self.assertTrue(torch.isfinite(energy.general).all())

    def test_frozen_scores_are_finite_for_token_without_rr_sources(self):
        model, graph = _model_and_graph()
        scores = model.score_graph(graph, device="cpu")
        self.assertTrue(
            all(torch.isfinite(values).all() for values in scores.values())
        )

    def test_paired_energy_objective_is_finite_and_backpropagates(self):
        model, graph = _model_and_graph()
        model.train()
        corruption = CorruptionConfig(
            token_span_min=2,
            token_span_max=3,
            layer_span_min=1,
            layer_span_max=2,
            selected_head_fraction=1.0,
        )
        plan = sample_corruption_plan(
            6, 2, 2, corruption,
            device="cpu", generator=torch.Generator().manual_seed(13), forced_type=4,
        )
        clean = model.encode_online(graph, device="cpu")
        damaged = model.encode_online(
            graph, corruption_plan=plan, corruption_config=corruption, device="cpu"
        )
        clean_energy = model.energy(clean)
        damaged_energy = model.energy(damaged)
        channel_mask = damaged.channel_corruption_mask
        token_mask = channel_mask.any(dim=(1, 2))
        loss = (
            robust_clean_energy_loss(clean_energy.general, 0.9)
            + corrupted_energy_loss(damaged_energy.general, token_mask)
            + pairwise_ranking_loss(
                clean_energy.general, damaged_energy.general, token_mask, margin=1.0
            )
        )
        variance, covariance = variance_covariance_loss(clean.token_embedding)
        loss = loss + variance + 0.04 * covariance
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(any(bool((g.abs() > 0).any()) for g in gradients))


if __name__ == "__main__":
    unittest.main()
