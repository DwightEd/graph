import torch

from experiments.source_reuse_contrast.baselines import received_support_topk
from experiments.source_reuse_contrast.data import (
    collect_source_reuse_graph,
    select_sample_ids,
)

from .helpers import SyntheticDataset, SyntheticSample, sequence_sample


def test_select_sample_ids_filters_task_before_applying_limit():
    dataset = SyntheticDataset(
        [
            sequence_sample(sample_id="summary-0", task_type="Summary"),
            sequence_sample(sample_id="qa-0", task_type="QA"),
            sequence_sample(sample_id="summary-1", task_type="Summary"),
            sequence_sample(sample_id="qa-1", task_type="QA"),
        ]
    )

    assert select_sample_ids(dataset, task_type="QA", limit=2) == ["qa-0", "qa-1"]


def test_collect_graph_keeps_exact_endpoints_and_query_order():
    graph = collect_source_reuse_graph(sequence_sample())
    assert graph.num_response_tokens == 5
    assert graph.diagonal.shape == (5, 2, 2)
    assert graph.query.tolist() == sorted(graph.query.tolist())
    for token in range(graph.num_response_tokens):
        current = graph.token_slice(token)
        assert torch.all(graph.source[current] < graph.response_idx + token)


def test_received_support_is_cumulative_source_reuse_not_current_entropy():
    sample = SyntheticSample(
        num_layers=1,
        num_heads=1,
        response_idx=1,
        num_response_tokens=3,
        edges=(
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 2, 2],
            [0, 1, 1, 2],
            [0.5, 0.4, 0.2, 0.6],
        ),
    )
    graph = collect_source_reuse_graph(sample)
    value = received_support_topk(graph, topk=2)[:, 0, 0]
    torch.testing.assert_close(value[1], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(value[2], torch.tensor([0.3, 0.2]))
