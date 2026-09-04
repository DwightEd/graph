from __future__ import annotations

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.constraint_routing_rhythm.artifacts import save_result
from experiments.constraint_routing_rhythm.capture import (
    capture_sample,
    matched_non_evidence_mask,
)
from experiments.constraint_routing_rhythm.routes import FunctionalRoutes


def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(17)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=43,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
    ).eval()


def test_capture_runs_one_primary_cut_and_keeps_route_maps_ephemeral(tmp_path) -> None:
    capture = capture_sample(
        tiny_model(),
        [1, 2, 3, 4, 5, 6, 7],
        response_start=4,
        prompt_evidence_mask=[False, True, True, False],
        sample_id="sample",
        source_id="source",
        task_type="QA",
        model_id="tiny",
        horizon_low=1,
        horizon_high=2,
    )

    arrays = capture.arrays
    torch.testing.assert_close(arrays["query_position"], torch.tensor([3, 4, 5]))
    torch.testing.assert_close(
        arrays["prediction_position"], arrays["query_position"] + 1
    )
    torch.testing.assert_close(
        arrays["constraint_deficit"],
        arrays["cut_margin"] - arrays["baseline_margin"],
    )
    assert arrays["baseline_target_logprob"].shape == arrays["baseline_margin"].shape
    assert arrays["baseline_entropy"].shape == arrays["baseline_margin"].shape
    assert torch.isfinite(arrays["baseline_target_logprob"]).all()
    assert torch.isfinite(arrays["baseline_entropy"]).all()
    assert capture.routes.all_map.shape == (3, 6)
    assert capture.routes.absolute_map.shape == (3, 6)
    assert capture.routes.split_layer == 1
    assert capture.routes.early_map.shape == capture.routes.late_map.shape == (3, 6)
    assert arrays["relay_capacity"].shape == arrays["constraint_deficit"].shape
    assert arrays["relay_mass"].shape == arrays["constraint_deficit"].shape
    assert "local_map" not in arrays
    assert "global_map" not in arrays
    assert int(arrays["evidence_tokens"]) == 2
    assert not bool(arrays["relay_audited"])
    assert not bool(arrays["control_audited"])
    assert capture.rhythm.upstream_edges.shape == (0, 0)
    assert capture.rhythm.downstream_edges.shape == (0, 0)
    assert np.isnan(np.asarray(arrays["direct_response_cut_delta"])).all()
    assert np.isnan(np.asarray(arrays["relay_interaction"])).all()

    save_result(tmp_path / "sample.npz", arrays)
    assert (tmp_path / "sample.npz").is_file()


def test_relay_audit_is_skipped_when_no_carrier_is_proposed() -> None:
    capture = capture_sample(
        tiny_model(),
        [1, 2, 3, 4, 5, 6],
        response_start=5,
        prompt_evidence_mask=[True, False, False, False, False],
        sample_id="short",
        source_id="source",
        task_type="Summary",
        model_id="tiny",
        audit_relay=True,
        horizon_low=10,
        horizon_high=20,
    )

    assert bool(capture.arrays["control_audited"])
    assert bool(capture.arrays["matched_control_available"])
    assert torch.isfinite(capture.arrays["direct_response_cut_delta"]).all()
    assert torch.isfinite(capture.arrays["matched_non_evidence_cut_delta"]).all()
    assert not bool(capture.arrays["relay_audited"])
    assert not capture.rhythm.carrier_mask.any()


def test_non_evidence_control_matches_count_without_reusing_sources() -> None:
    absolute = torch.zeros(2, 6)
    absolute[:, 0] = 1.0
    absolute[:, 2] = 3.0
    absolute[:, 1] = 0.9
    absolute[:, 3] = 2.9
    routes = FunctionalRoutes(
        row_start=3,
        split_layer=1,
        absolute_map=absolute,
        all_map=torch.zeros_like(absolute),
        early_absolute_map=absolute,
        early_map=torch.zeros_like(absolute),
        late_absolute_map=absolute,
        late_map=torch.zeros_like(absolute),
        local_map=torch.zeros_like(absolute),
        global_map=torch.zeros_like(absolute),
    )
    evidence = torch.tensor([True, False, True, False, False, False])

    matched = matched_non_evidence_mask(routes, evidence, response_start=4)

    assert matched is not None
    assert int(matched.sum()) == int(evidence.sum())
    assert not (matched & evidence).any()
    torch.testing.assert_close(matched[:4], torch.tensor([False, True, False, True]))
