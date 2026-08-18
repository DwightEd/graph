import unittest

import torch

from experiments.causal_setflow.config import (
    CorruptionConfig,
    SetFlowModelConfig,
    SourceSetConfig,
)
from experiments.causal_setflow.corruptions import sample_corruption_plan
from experiments.causal_setflow.data import CausalSourceSetGraph, SparseRRLayer
from experiments.causal_setflow.model import CausalSetFlowModel


def _layer(scale=1.0):
    return SparseRRLayer(
        head=torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]),
        query=torch.tensor([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5]),
        source=torch.tensor([0, 0, 0, 1, 1, 2, 0, 3, 1, 4, 2]),
        weight=torch.tensor(
            [0.2, 0.1, 0.3, 0.15, 0.25, 0.4, 0.2, 0.22, 0.18, 0.35, 0.27]
        ) * float(scale),
    ).validate(num_heads=2, response_count=6)


def _graph():
    return CausalSourceSetGraph(
        layers=(_layer(1.0), _layer(0.8)),
        num_heads=2,
        response_count=6,
        attention_floor=0.01,
    ).validate()


def _source(chunk):
    return SourceSetConfig(
        max_route_sources=4,
        max_memory_sources=4,
        route_mass_coverage=1.0,
        materialize_query_chunk_size=chunk,
    )


def _model_config(set_chunk, mixer_chunk, checkpointing):
    return SetFlowModelConfig(
        hidden_dim=8,
        scalar_fourier_dim=2,
        set_heads=2,
        induced_points=2,
        set_blocks=1,
        head_mixer_layers=1,
        depth_mixer_layers=1,
        set_row_chunk_size=set_chunk,
        mixer_token_chunk_size=mixer_chunk,
        activation_checkpointing=checkpointing,
        dropout=0.0,
    )


class CausalSetFlowMemoryTests(unittest.TestCase):
    def test_query_chunking_matches_dense_received_support(self):
        graph = _graph()
        config = _source(2)
        actual = graph.materialize_layer(0, config, device="cpu")
        layer = graph.layers[0]
        current = torch.zeros((2, 6, 6))
        current.index_put_((layer.head, layer.query, layer.source), layer.weight, accumulate=True)
        cumulative = current.cumsum(dim=1)
        target = torch.arange(6)[:, None]
        source = torch.arange(6)[None, :]
        age = (target - source + 1).clamp_min(1).float()
        received = torch.where(
            (source < target)[None], cumulative / age[None], torch.zeros_like(cumulative)
        )
        route_weight, route_source = torch.topk(current, 4, dim=-1)
        memory_received, memory_source = torch.topk(received, 4, dim=-1)
        torch.testing.assert_close(actual.route_weight, route_weight.permute(1, 0, 2))
        torch.testing.assert_close(actual.route_received, torch.gather(received, 2, route_source).permute(1, 0, 2))
        torch.testing.assert_close(actual.memory_received, memory_received.permute(1, 0, 2))
        self.assertTrue(torch.equal(actual.route_source.long(), route_source.permute(1, 0, 2)))
        self.assertTrue(torch.equal(actual.memory_source.long(), memory_source.permute(1, 0, 2)))

    def test_execution_chunks_do_not_change_eval_energy(self):
        graph = _graph()
        small = CausalSetFlowModel(
            2, 2, _source(1), _model_config(2, 2, False)
        )
        large = CausalSetFlowModel(
            2, 2, _source(6), _model_config(64, 64, False)
        )
        large.load_state_dict(small.state_dict())
        small.eval()
        large.eval()
        with torch.inference_mode():
            left = small.score_graph(graph, device="cpu")
            right = large.score_graph(graph, device="cpu")
        for name in ("embedding", "general_energy", "token_energy", "channel_energy"):
            torch.testing.assert_close(left[name], right[name], rtol=2e-5, atol=2e-6)

    def test_checkpointed_online_encoder_backpropagates(self):
        graph = _graph()
        model = CausalSetFlowModel(
            2, 2, _source(2), _model_config(3, 3, True)
        )
        model.train()
        generator = torch.Generator().manual_seed(7)
        plan = sample_corruption_plan(
            6, 2, 2, CorruptionConfig(token_span_min=2, token_span_max=3, layer_span_min=1, layer_span_max=2),
            device="cpu", generator=generator, forced_type=0,
        )
        teacher = model.encode_teacher(graph, device="cpu")
        corrupted = model.encode_online(
            graph, corruption_plan=plan, corruption_config=CorruptionConfig(token_span_min=2, token_span_max=3, layer_span_min=1, layer_span_max=2), device="cpu"
        )
        energy = model.energy(corrupted)
        projected = model.project(corrupted)
        loss = energy.general.mean() + (projected.token - teacher.token_embedding).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()