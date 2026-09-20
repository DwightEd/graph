"""Independent graph path enumeration and real Llama intervention semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.reanchor_audit.graph import conditioned_flow
from experiments.reanchor_audit.selection import select_events, find_relay, make_plan
from experiments.reanchor_audit.report import report


def enumerate_paths(source, capacity, residual, target):
    layers, heads, length, slots = source.shape
    paths = []

    def walk(layer, position, weight, edges, root):
        if layer == layers:
            if position == target:
                paths.append((root, weight, edges))
            return
        walk(layer + 1, position, weight * residual[layer, position], edges, root)
        for head, receiver, slot in np.ndindex(heads, length, slots):
            if source[layer, head, receiver, slot] == position and capacity[layer, head, receiver, slot] > 0:
                edge = (layer, head, receiver, slot)
                walk(layer + 1, receiver, weight * capacity[edge], edges + [edge], root)

    for root in range(length):
        walk(0, root, 1., [], root)
    return paths


@pytest.mark.parametrize("seed", ["potential", "uniform"])
def test_flow_matches_enumerated_layer_ordered_paths(seed):
    source = np.tile(np.array([[0, 0], [0, 1], [1, 2]]), (2, 2, 1, 1))
    capacity = np.arange(1, 25, dtype=float).reshape(2, 2, 3, 2) / 100
    capacity[:, :, 0, 1] = 0.
    residual = np.full((2, 3), .3)
    paths = enumerate_paths(source, capacity, residual, 2)
    root_weights = np.bincount([root for root, _, _ in paths], weights=[weight for _, weight, _ in paths], minlength=3)
    expected = np.zeros_like(capacity)
    for root, weight, edges in paths:
        probability = weight / root_weights.sum() if seed == "potential" else weight / root_weights[root] / 3
        for edge in edges:
            expected[edge] += probability
    actual = conditioned_flow(source, capacity, residual, 2, seed)
    np.testing.assert_allclose(actual["potential"][0], root_weights)
    np.testing.assert_allclose(actual["edges"], expected)
    np.testing.assert_allclose(actual["nodes"].sum(axis=1), 1.)
    assert actual["nodes"][-1, 2] == pytest.approx(1.)
    assert not np.allclose(actual["edges"][:, 0], actual["edges"][:, 1])


def test_weak_roots_and_absent_paths_are_not_inflated_or_imputed():
    source = np.array([[[[0, 0], [0, 1]]]])
    capacity = np.array([[[[0., 0.], [1e-8, 1.]]]])
    residual = np.zeros((1, 2))
    primary = conditioned_flow(source, capacity, residual, 1)
    uniform = conditioned_flow(source, capacity, residual, 1, "uniform")
    assert primary["nodes"][0, 0] < 1e-7
    assert uniform["nodes"][0, 0] == .5
    missing = conditioned_flow(source, capacity * 0, residual, 1)
    assert not missing["reachable"] and not missing["nodes"].any()


def test_earlier_candidates_and_relays_do_not_use_same_layer_or_future_edges():
    rows = [dict(layer=layer, head=0, receiver=receiver, source=source,
        lookback_gain=gain, target_flow=flow, source_region="prompt")
        for layer, receiver, source, gain, flow in [(0, 3, 0, .2, .4), (0, 4, 0, .3, 9.),
            (0, 2, 0, .1, .02), (1, 4, 3, -.1, .3), (0, 4, 3, .1, 10.)]]
    edges = pd.DataFrame(rows)
    events = select_events(edges, 1, target=4)
    earlier = events[events.selection == "lookback_target_flow"]
    assert earlier.receiver.tolist() == [3]
    relay = find_relay(edges, earlier.iloc[0])
    assert relay["layer"] == 1 and relay["source"] == 3
    assert find_relay(edges, dict(layer=1, receiver=3)) is None
    assert events[events.selection == "decision_query_control"].receiver.tolist() == [4]


def test_inventory_only_report_does_not_report_missing_worlds_as_passed(tmp_path):
    pd.DataFrame([dict(case_id="claim", source_id="source", side="supported",
        readout_status="unreviewed", history_status="unreviewed")]).to_csv(tmp_path / "inventory.csv", index=False)
    summary = report(tmp_path)
    assert summary["native_graphs"] == 0 and summary["finite_effect_rows"] == 0
    assert summary["baseline_checks_missing"] == 1 and summary["baseline_checks_passed"] == 0
    assert summary["failed_control_rows"] == 0
    assert (tmp_path / "reanchor_review.tar.gz").is_file()


def test_common_key_lookback_gain_has_no_moving_boundary_artifact():
    import torch
    from experiments.reanchor_audit.capture import lookback_gain

    attention = torch.zeros(1, 6, 6)
    attention[0, 3:, 0] = .1
    attention[0, 3:, 2] = .4
    attention[0, 3:, 3] = .5
    gain = lookback_gain(attention, 2, 2, [True, False, False, False, False, False])
    # At r=5, key 2 enters the far set; both receivers already assigned it .4.
    assert gain[0, 5] == 0.
    attention[0, 5, 0] += .2  # A special-token increase must not be mistaken for evidence.
    changed = lookback_gain(attention, 2, 2, [True, False, False, False, False, False])
    assert changed[0, 5] == 0.


@pytest.fixture
def tiny_native():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(18)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=41, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, initializer_range=.1)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    probe = dict(prefix_ids=[1, 3, 5, 2, 6], candidates=[[7, 8], [9, 10, 11]],
        prompt_length=3, phase="onset", readout="sequence_margin", actual_token=7,
        record_writes=False, groups={name: np.array([index]) for index, name in enumerate(
            ["scope", "supported_value", "value_source"])})
    return model, probe


def test_native_capture_is_transparent_and_preserves_gqa_and_sparse_mass(tiny_native):
    from experiments.reanchor_audit.capture import capture_graph
    from experiments.reanchor_audit.graph import analyze_graph
    from experiments.reanchor_audit.run import route_checks
    from experiments.path_conflict.scoring import evaluate_candidates

    model, probe = tiny_native
    record, _ = evaluate_candidates(model, probe)
    graph = capture_graph(model, probe, top_k=5, window=2)
    expected = [record["correct_first_logp"], record["wrong_first_logp"]]
    np.testing.assert_allclose(graph["candidate_first_logp"], expected, atol=1e-7)
    np.testing.assert_allclose(graph["retained_mass"], 1., atol=1e-6)
    assert graph["source"].shape == (3, 4, 5, 5)
    assert route_checks(graph, analyze_graph(graph), 1e-5)["route_checks_ok"]
    sparse = capture_graph(model, probe, top_k=1, window=2)
    assert (sparse["retained_mass"][:, :, 1:] < 1.).all()
    assert route_checks(sparse, analyze_graph(sparse), 1e-5)["route_checks_ok"]


def test_native_state_identity_joint_shams_relay_restore_and_resume(tiny_native, tmp_path, monkeypatch):
    from experiments.reanchor_audit.trials import audit_trials

    model, probe = tiny_native
    events = pd.DataFrame([dict(layer=0, head=1, receiver=3, source=0, selection="lookback_target_flow")])
    edges = pd.DataFrame([dict(layer=1, head=2, receiver=4, source=3, target_flow=.1)])
    plan = make_plan(events, edges, probe["groups"])
    (tmp_path / "worlds").mkdir()
    full = audit_trials(model, probe, plan, tmp_path, [.25, 1.], [0], 1e-5)
    effects = pd.read_csv(tmp_path / "effects.csv")
    assert effects.numeric_ok.all()
    assert effects.support.abs().max() > 1e-5
    assert effects.restoration_identity_error.max() < 1e-7
    assert effects.blocked_identity_error.max() < 1e-7
    assert effects.joint_sham_error.max() < 1e-7
    np.testing.assert_allclose(effects.interaction, effects.gate_difference, atol=1e-12)
    np.testing.assert_allclose(effects.relay_restoration_gain, effects.restored_relay - effects.removed)
    assert effects.relay_restoration_gain.abs().max() > 1e-8
    def no_forward(*args, **kwargs):
        raise AssertionError("Cached resume should not run the model")
    monkeypatch.setattr(model, "forward", no_forward)
    resumed = audit_trials(model, probe, plan, tmp_path, [.25, 1.], [0], 1e-5)
    assert resumed == full


def test_capture_removes_hooks_when_model_fails(tiny_native, monkeypatch):
    from experiments.reanchor_audit.capture import capture_graph

    model, probe = tiny_native
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate")
    monkeypatch.setattr(model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        capture_graph(model, probe)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_complete_discovery_intervention_and_report_pipeline(tiny_native, tmp_path):
    from experiments.reanchor_audit.run import run_case

    class CharacterTokenizer:
        all_special_ids = []

        def decode(self, ids, **kwargs):
            return "".join(chr(64 + token) for token in ids)

        def __call__(self, text, **kwargs):
            return dict(input_ids=[ord(character) - 64 for character in text],
                        offset_mapping=[(index, index + 1) for index in range(len(text))])

    model, probe = tiny_native
    case = dict(case_id="tiny_software_check", source_id="synthetic", history_status={"supported": "test_only"},
                source_roles=dict(scope=["A"], supported_value=["C"], value_source=["E"]))
    item = dict(case=case, side="supported", prompt_length=3, context=probe)
    args = SimpleNamespace(output=tmp_path, stage="run", top_k=5, window=2, events=1,
        control_radius=10, doses=[1.], random_seeds=[0], sham_atol=1e-5, message_rtol=.02)
    pd.DataFrame([dict(case_id=case["case_id"], source_id="synthetic", side="supported",
        readout_status="software_check_only", history_status="test_only")]).to_csv(tmp_path / "inventory.csv", index=False)
    (tmp_path / "audit_config.json").write_text(json.dumps(dict(doses=[1.])))
    run_case(model, CharacterTokenizer(), item, args)
    summary = report(tmp_path)
    assert summary["native_graphs"] == 1 and summary["baseline_checks_passed"] == 1
    assert summary["finite_effect_rows"] > 0 and summary["failed_control_rows"] == 0
    assert (tmp_path / "cases" / case["case_id"] / "supported" / "route_timeline.png").is_file()
    roles = pd.read_csv(tmp_path / "role_interactions.csv")
    assert len(roles) > 0 and roles.numeric_ok.all()
    effects_path = tmp_path / "cases" / case["case_id"] / "supported" / "effects.csv"
    effects = pd.read_csv(effects_path)
    effects.loc[0, "numeric_ok"] = False
    effects.to_csv(effects_path, index=False)
    assert report(tmp_path, package=False)["failed_control_rows"] == 1
