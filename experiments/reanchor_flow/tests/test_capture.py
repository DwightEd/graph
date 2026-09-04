import torch

from experiments.reanchor_flow.capture import evidence_gate, evidence_source_mask


def test_evidence_mask_stays_in_teacher_forcing_source_coordinates():
    evidence = evidence_source_mask([False, True, True], 6, response_start=3)
    torch.testing.assert_close(
        evidence,
        torch.tensor([False, True, True, False, False, False]),
    )


def test_direct_cut_starts_at_query_before_first_response_token():
    evidence = evidence_source_mask([False, True, True], 6, response_start=3)
    gate = evidence_gate(
        evidence, response_start=3, layer_count=4, direct_response_only=True
    )
    assert gate.source_targets.tolist() == [False, False, True, True, True, True]
    assert gate.split_layer == 4


def test_global_cut_keeps_mlp_parametric_path_available():
    evidence = evidence_source_mask([False, True, True], 6, response_start=3)
    gate = evidence_gate(
        evidence, response_start=3, layer_count=4, direct_response_only=False
    )
    assert gate.source_targets is None
    assert gate.early_edges is None
    assert gate.late_edges is None
