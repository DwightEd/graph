import torch

from experiments.grounded_anchor_flow.pipeline import analyze_graph


def test_pipeline_keeps_full_functional_maps_and_compact_capacity_controls():
    token_flow = torch.zeros(3, 4, 4)
    token_flow[0, 0, 0] = 1.0
    token_flow[1, 1, 0] = 1.0
    token_flow[2, 2, 0] = 1.0
    token_flow[..., 2] = token_flow[..., 0]
    token_flow[..., 3] = token_flow[..., 0]
    graph = {
        "schema": "functional-message-graph-v2",
        "sample_id": "sample",
        "source_id": "source",
        "task_type": "QA",
        "generator_model": "generator",
        "response_start": 1,
        "evidence_mask": torch.tensor([True]),
        "target_logprob": torch.tensor([-1.0, -2.0, -3.0]),
        "token_flow": token_flow,
    }

    result = analyze_graph(graph)

    assert result["labels_used"] is False
    assert result["schema"] == "grounded-anchor-flow-v1"
    assert result["functional_response_seeded_path_share"].shape == (3,)
    assert result["functional_anchor_occupancy"].shape == (3, 3)
    assert result["functional_source_path_posterior"].shape == (3, 3)
    assert result["attention_response_seeded_anchor_flow"].shape == (3,)
    assert "attention_anchor_occupancy" not in result
    assert "message_source_path_posterior" not in result
    assert result["functional_anchor_valid"].tolist() == [False, True, True]
