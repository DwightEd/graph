from types import SimpleNamespace

import pytest
import torch

from experiments.constraint_routing_rhythm.routes import FunctionalRouteAccumulator


def model_with_layers(count: int):
    return SimpleNamespace(config=SimpleNamespace(num_hidden_layers=count))


def test_every_layer_enters_the_correct_band_map():
    probability = torch.zeros(1, 1, 2, 2)
    value = torch.ones(1, 1, 2, 1)
    weight = torch.ones(1, 1)
    routes = FunctionalRouteAccumulator(
        model_with_layers(4), response_start=2, split_layer=1
    )
    for layer in range(4):
        source = int(layer >= 1)
        probability[0, 0, 1, source] = 1
        routes.observe(layer, probability, value, weight)
        probability.zero_()

    result = routes.finish()
    assert result.split_layer == 1
    torch.testing.assert_close(result.all_map, torch.tensor([[0.25, 0.75]]))
    torch.testing.assert_close(result.absolute_map, torch.tensor([[0.25, 0.75]]))
    torch.testing.assert_close(result.early_map, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(result.early_absolute_map, result.early_map)
    torch.testing.assert_close(result.late_map, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(result.late_absolute_map, result.late_map)


def test_output_projection_changes_functional_routes_not_raw_attention():
    probability = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    repeated_value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    raw_attention = probability[0, 0, 1].clone()

    base = FunctionalRouteAccumulator(model_with_layers(1), response_start=2)
    base.observe(0, probability, repeated_value, torch.eye(2))
    scaled = FunctionalRouteAccumulator(model_with_layers(1), response_start=2)
    scaled.observe(0, probability, repeated_value, torch.diag(torch.tensor([4.0, 1.0])))

    torch.testing.assert_close(probability[0, 0, 1], raw_attention)
    torch.testing.assert_close(base.finish().all_map, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(scaled.finish().all_map, torch.tensor([[0.8, 0.2]]))
    torch.testing.assert_close(scaled.finish().absolute_map, torch.tensor([[2.0, 0.5]]))


def test_each_layer_separates_local_and_global_heads_by_functional_lookback():
    heads = queries = sources = 4
    probability = torch.zeros(1, heads, queries, sources)
    probability[:, :, 0, 0] = 1
    chosen_source = (
        (1, 2, 3),
        (0, 1, 2),
        (0, 0, 1),
        (0, 0, 0),
    )
    for head, sources_by_query in enumerate(chosen_source):
        for query, source in enumerate(sources_by_query, start=1):
            probability[0, head, query, source] = 1

    routes = FunctionalRouteAccumulator(
        model_with_layers(1), response_start=2, head_quantile=0.25
    )
    routes.observe(
        0,
        probability,
        torch.ones(1, heads, sources, 1),
        torch.eye(heads),
    )
    result = routes.finish()

    expected_local = torch.zeros(3, 4)
    expected_local[torch.arange(3), torch.arange(1, 4)] = 1
    expected_global = torch.zeros(3, 4)
    expected_global[:, 0] = 1
    torch.testing.assert_close(result.local_map, expected_local)
    torch.testing.assert_close(result.global_map, expected_global)


def test_maps_are_row_normalized_and_gqa_values_are_already_repeated():
    generator = torch.Generator().manual_seed(8)
    heads, kv_heads, tokens, head_dim = 4, 2, 7, 2
    probability = torch.rand(1, heads, tokens, tokens, generator=generator)
    probability = probability.tril()
    probability /= probability.sum(-1, keepdim=True)
    kv_value = torch.randn(1, kv_heads, tokens, head_dim, generator=generator)
    repeated_value = kv_value.repeat_interleave(heads // kv_heads, dim=1)
    output_weight = torch.randn(heads * head_dim, heads * head_dim, generator=generator)

    routes = FunctionalRouteAccumulator(model_with_layers(2), response_start=3)
    with pytest.raises(ValueError, match="one row per query head"):
        routes.observe(0, probability, kv_value, output_weight)
    routes.observe(0, probability, repeated_value, output_weight)
    routes.observe(1, probability, repeated_value, output_weight)
    result = routes.finish()

    assert result.all_map.shape == (tokens - 2, tokens)
    for route_map in (result.all_map, result.local_map, result.global_map):
        torch.testing.assert_close(route_map.sum(-1), torch.ones(tokens - 2))
    torch.testing.assert_close(result.early_map.sum(-1), torch.ones(tokens - 2))
    torch.testing.assert_close(result.late_map.sum(-1), torch.ones(tokens - 2))


def test_query_chunking_is_numerically_invariant():
    generator = torch.Generator().manual_seed(12)
    heads, tokens, head_dim = 4, 9, 3
    probability = torch.rand(1, heads, tokens, tokens, generator=generator).tril()
    probability /= probability.sum(-1, keepdim=True)
    value = torch.randn(1, heads, tokens, head_dim, generator=generator)
    weight = torch.randn(heads * head_dim, heads * head_dim, generator=generator)

    outputs = []
    for chunk in (1, 4, 128):
        routes = FunctionalRouteAccumulator(
            model_with_layers(4), response_start=4, query_chunk=chunk
        )
        for layer in range(4):
            routes.observe(layer, probability, value, weight)
        outputs.append(routes.finish())

    first = outputs[0]
    for other in outputs[1:]:
        torch.testing.assert_close(first.all_map, other.all_map)
        torch.testing.assert_close(first.local_map, other.local_map)
        torch.testing.assert_close(first.global_map, other.global_map)
        torch.testing.assert_close(first.early_absolute_map, other.early_absolute_map)
        torch.testing.assert_close(first.early_map, other.early_map)
        torch.testing.assert_close(first.late_absolute_map, other.late_absolute_map)
        torch.testing.assert_close(first.late_map, other.late_map)


@pytest.mark.parametrize("query_chunk", [0, -1])
def test_query_chunk_must_be_positive(query_chunk: int) -> None:
    with pytest.raises(ValueError, match="query_chunk must be positive"):
        FunctionalRouteAccumulator(
            model_with_layers(2), response_start=2, query_chunk=query_chunk
        )


def test_callback_coverage_rejects_missing_repeated_or_shape_changed_layers():
    probability = torch.eye(3)[None, None].expand(1, 2, -1, -1).clone()
    value = torch.ones(1, 2, 3, 1)
    weight = torch.eye(2)

    missing = FunctionalRouteAccumulator(model_with_layers(2), response_start=2)
    missing.observe(0, probability, value, weight)
    with pytest.raises(RuntimeError, match="missed layers"):
        missing.finish()
    with pytest.raises(ValueError, match="repeated attention layer"):
        missing.observe(0, probability, value, weight)

    changed = FunctionalRouteAccumulator(model_with_layers(2), response_start=2)
    changed.observe(0, probability, value, weight)
    smaller = probability[:, :1]
    with pytest.raises(ValueError, match="count changed"):
        changed.observe(1, smaller, value[:, :1], torch.ones(1, 1))
