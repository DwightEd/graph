import unittest

import torch

from experiments.causal_setflow.config import (
    SetFlowModelConfig,
    SourceSetConfig,
)
from experiments.causal_setflow.data import (
    CausalSourceSetGraph,
    SparseRRLayer,
)
from experiments.causal_setflow.model import CausalSetFlowModel


def _sparse_layer(scale: float = 1.0) -> SparseRRLayer:
    # Sorted by (query, head, source), matching the production data contract.
    return SparseRRLayer(
        head=torch.tensor(
            [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            dtype=torch.long,
        ),
        query=torch.tensor(
            [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
            dtype=torch.long,
        ),
        source=torch.tensor(
            [0, 0, 0, 1, 1, 0, 2, 0, 2, 1, 3, 0, 3, 4, 2],
            dtype=torch.long,
        ),
        weight=(
            torch.tensor(
                [
                    0.20,
                    0.10,
                    0.30,
                    0.15,
                    0.25,
                    0.05,
                    0.40,
                    0.20,
                    0.10,
                    0.18,
                    0.22,
                    0.08,
                    0.30,
                    0.35,
                    0.27,
                ],
                dtype=torch.float32,
            )
            * float(scale)
        ),
    ).validate(num_heads=2, response_count=6)


def _graph() -> CausalSourceSetGraph:
    return CausalSourceSetGraph(
        layers=(_sparse_layer(1.0), _sparse_layer(0.8)),
        num_heads=2,
        response_count=6,
        attention_floor=0.01,
    ).validate()


def _dense_reference(
    graph: CausalSourceSetGraph,
    layer_index: int,
    config: SourceSetConfig,
):
    layer = graph.layers[layer_index]
    heads, tokens = graph.num_heads, graph.response_count
    current = torch.zeros((heads, tokens, tokens), dtype=torch.float32)
    current.index_put_(
        (layer.head, layer.query, layer.source),
        layer.weight,
        accumulate=True,
    )
    cumulative = current.cumsum(dim=1)
    target = torch.arange(tokens)[:, None]
    source = torch.arange(tokens)[None, :]
    age = (target - source + 1).clamp_min(1).float()
    causal = source < target
    received = torch.where(
        causal[None, :, :],
        cumulative / age[None, :, :],
        torch.zeros_like(cumulative),
    )
    previous = torch.zeros_like(received)
    if tokens > 1:
        previous[:, 1:] = received[:, :-1]
    received_delta = received - previous

    route_keep_count = min(config.max_route_sources, tokens)
    route_weight, route_source = torch.topk(
        current,
        k=route_keep_count,
        dim=-1,
        largest=True,
        sorted=True,
    )
    if route_keep_count < config.max_route_sources:
        pad = config.max_route_sources - route_keep_count
        route_weight = torch.cat(
            (route_weight, torch.zeros((*route_weight.shape[:-1], pad))),
            dim=-1,
        )
        route_source = torch.cat(
            (
                route_source,
                torch.zeros(
                    (*route_source.shape[:-1], pad), dtype=torch.long
                ),
            ),
            dim=-1,
        )
    route_received = torch.gather(received, 2, route_source)
    route_delta = torch.gather(received_delta, 2, route_source)
    total_mass = current.sum(dim=2)
    edge_count = (current > 0).sum(dim=2).float()
    before = route_weight.cumsum(dim=2) - route_weight
    route_mask = (route_weight > 0) & (
        before
        < config.route_mass_coverage
        * total_mass[:, :, None].clamp_min(config.epsilon)
    )
    tail_mass = (
        total_mass - (route_weight * route_mask).sum(dim=2)
    ).clamp_min(0.0)

    memory_keep_count = min(config.max_memory_sources, tokens)
    memory_received, memory_source = torch.topk(
        received,
        k=memory_keep_count,
        dim=-1,
        largest=True,
        sorted=True,
    )
    if memory_keep_count < config.max_memory_sources:
        pad = config.max_memory_sources - memory_keep_count
        memory_received = torch.cat(
            (
                memory_received,
                torch.zeros((*memory_received.shape[:-1], pad)),
            ),
            dim=-1,
        )
        memory_source = torch.cat(
            (
                memory_source,
                torch.zeros(
                    (*memory_source.shape[:-1], pad), dtype=torch.long
                ),
            ),
            dim=-1,
        )
    memory_delta = torch.gather(received_delta, 2, memory_source)
    memory_current = torch.gather(current, 2, memory_source)
    memory_mask = memory_received > 0

    transpose = lambda value: value.permute(1, 0, 2).contiguous()
    return {
        "route_source": transpose(route_source),
        "route_weight": transpose(route_weight),
        "route_received": transpose(route_received),
        "route_received_delta": transpose(route_delta),
        "route_mask": transpose(route_mask),
        "memory_source": transpose(memory_source),
        "memory_received": transpose(memory_received),
        "memory_received_delta": transpose(memory_delta),
        "memory_current_weight": transpose(memory_current),
        "memory_mask": transpose(memory_mask),
        "total_mass": total_mass.transpose(0, 1).contiguous(),
        "tail_mass": tail_mass.transpose(0, 1).contiguous(),
        "edge_count": edge_count.transpose(0, 1).contiguous(),
    }


def _model_config(
    *,
    set_row_chunk_size: int,
    mixer_token_chunk_size: int,
    activation_checkpointing: bool,
) -> SetFlowModelConfig:
    return SetFlowModelConfig(
        hidden_dim=8,
        scalar_fourier_dim=2,
        set_heads=2,
        induced_points=2,
        set_blocks=1,
        head_mixer_layers=1,
        depth_mixer_layers=1,
        set_row_chunk_size=set_row_chunk_size,
        mixer_token_chunk_size=mixer_token_chunk_size,
        activation_checkpointing=activation_checkpointing,
        dropout=0.0,
        element_mask_probability=0.25,
        head_mask_probability=0.25,
        layer_mask_probability=0.25,
    )


class CausalSetFlowMemoryTests(unittest.TestCase):
    def test_query_chunking_matches_dense_received_support_definition(self):
        graph = _graph()
        config = SourceSetConfig(
            max_route_sources=4,
            max_memory_sources=4,
            route_mass_coverage=1.0,
            materialize_query_chunk_size=2,
        )
        actual = graph.materialize_layer(0, config, device="cpu")
        expected = _dense_reference(graph, 0, config)

        torch.testing.assert_close(actual.route_mask, expected["route_mask"])
        torch.testing.assert_close(actual.memory_mask, expected["memory_mask"])
        for name in (
            "route_weight",
            "route_received",
            "route_received_delta",
            "memory_received",
            "memory_received_delta",
            "memory_current_weight",
            "total_mass",
            "tail_mass",
            "edge_count",
        ):
            torch.testing.assert_close(
                getattr(actual, name), expected[name], rtol=1e-6, atol=1e-7
            )
        self.assertTrue(
            torch.equal(
                actual.route_source[actual.route_mask].long(),
                expected["route_source"][expected["route_mask"]].long(),
            )
        )
        self.assertTrue(
            torch.equal(
                actual.memory_source[actual.memory_mask].long(),
                expected["memory_source"][expected["memory_mask"]].long(),
            )
        )

    def test_execution_chunk_sizes_do_not_change_eval_representation(self):
        graph = _graph()
        source_small = SourceSetConfig(
            max_route_sources=4,
            max_memory_sources=4,
            route_mass_coverage=1.0,
            materialize_query_chunk_size=1,
        )
        source_large = SourceSetConfig(
            max_route_sources=4,
            max_memory_sources=4,
            route_mass_coverage=1.0,
            materialize_query_chunk_size=6,
        )
        small = CausalSetFlowModel(
            2,
            2,
            source_config=source_small,
            model_config=_model_config(
                set_row_chunk_size=2,
                mixer_token_chunk_size=2,
                activation_checkpointing=False,
            ),
        )
        large = CausalSetFlowModel(
            2,
            2,
            source_config=source_large,
            model_config=_model_config(
                set_row_chunk_size=64,
                mixer_token_chunk_size=64,
                activation_checkpointing=False,
            ),
        )
        large.load_state_dict(small.state_dict())
        small.eval()
        large.eval()
        with torch.inference_mode():
            output_small = small(graph, apply_masks=False, device="cpu")
            output_large = large(graph, apply_masks=False, device="cpu")
        torch.testing.assert_close(
            output_small.token_embedding,
            output_large.token_embedding,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            output_small.depth_state,
            output_large.depth_state,
            rtol=2e-5,
            atol=2e-6,
        )

    def test_internal_layer_checkpointing_preserves_finite_backward(self):
        graph = _graph()
        source = SourceSetConfig(
            max_route_sources=4,
            max_memory_sources=4,
            route_mass_coverage=1.0,
            materialize_query_chunk_size=2,
        )
        model = CausalSetFlowModel(
            2,
            2,
            source_config=source,
            model_config=_model_config(
                set_row_chunk_size=3,
                mixer_token_chunk_size=3,
                activation_checkpointing=True,
            ),
        )
        model.train()
        output = model(
            graph,
            mask_seed=1234,
            apply_masks=True,
            device="cpu",
        )
        self.assertTrue(torch.isfinite(output.loss.total))
        output.loss.total.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(any(bool((value.abs() > 0).any()) for value in gradients))


if __name__ == "__main__":
    unittest.main()
