"""Fixed-foil target isolation, exact-query causal cuts, shared edges, and cache controls."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from state_audit.intervention import intervene
from state_audit.model.replay import attention_backend
from state_audit.operations import Delete, Target
from state_audit.storage import read_arrays, read_json, write_json
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.message_carriers.run import main
from experiments.native_support.message_carriers.token_capture import capture_token
from experiments.native_support.message_carriers.token_native import observe_token, deleted_values, output_values
from experiments.native_support.message_carriers.token_selection import candidate_table, select_edges
from experiments.native_support.message_carriers.token_representation import aggregate_units, token_vector, schema
from test_evidence_contrast import tiny_model
from test_message_carriers import example_views, make_contrast_input


def dense_cut(model, prompt, answer, target, foil, edges):
    operations = [Delete(Target("attention", (layer,), heads=(head,),
        positions=(len(prompt) + query,), keys=(len(prompt) + key,)))
        for layer, head, query, key in edges]
    with torch.no_grad(), attention_backend(model, "eager"), intervene(model, operations):
        hidden = model.forward(prompt + answer[:target])
        return output_values(model.native.lm_head(hidden[-1]), answer[target], foil).numpy()


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_exact_receiver_gates_match_dense_and_earlier_last_layer_is_causally_silent(family):
    model, views = tiny_model(family), example_views()
    prompt, answer, target = views["prompt_with_source"], views["answer_ids"], 4
    observed = observe_token(model, prompt, answer, target)
    foil = observed["foil"]
    edges = [(0, 0, 1, 0), (1, 1, 3, 2)]
    np.testing.assert_allclose(deleted_values(model, prompt, answer, target, foil, edges),
                              dense_cut(model, prompt, answer, target, foil, edges), atol=1e-6)
    original = deleted_values(model, prompt, answer, target, foil)
    earlier_first_layer = [(0, 0, 1, 0)]
    propagated = deleted_values(model, prompt, answer, target, foil, earlier_first_layer)
    np.testing.assert_allclose(propagated, dense_cut(model, prompt, answer, target, foil, earlier_first_layer), atol=1e-6)
    assert abs(propagated - original).max() > 1e-7
    earlier_last_layer = [(1, 0, 1, 0)]
    np.testing.assert_array_equal(original, deleted_values(model, prompt, answer, target, foil, earlier_last_layer))
    np.testing.assert_array_equal(original, deleted_values(model, prompt, answer, target, foil, edges, strength=0))
    assert model.native.config._attn_implementation == "eager"
    assert all(parameter.grad is None for parameter in model.native.parameters())


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_conditional_margin_derivative_matches_two_world_finite_difference(family):
    model, views, target = tiny_model(family), example_views(), 4
    prompts = [views[f"prompt_{name}"] for name in ("with_source", "without_source")]
    first = observe_token(model, prompts[0], views["answer_ids"], target)
    second = observe_token(model, prompts[1], views["answer_ids"], target, first["foil"])
    table = candidate_table(model, [first, second], target, 2)
    selected = select_edges(table, 2, 37, "conditional")
    epsilon = .02
    for index, edge in enumerate(selected["edges"]):
        derivatives = []
        for prompt in prompts:
            plus = deleted_values(model, prompt, views["answer_ids"], target, first["foil"], [edge], -epsilon)
            minus = deleted_values(model, prompt, views["answer_ids"], target, first["foil"], [edge], epsilon)
            derivatives.append(float(plus[1] - minus[1]) / (2 * epsilon))
        np.testing.assert_allclose(derivatives, selected["approximation"][index], atol=2e-5, rtol=.03)
        assert derivatives[0] - derivatives[1] == pytest.approx(np.diff(selected["approximation"][index])[0] * -1, abs=3e-5)
    assert (selected["edges"][:, 3] <= selected["edges"][:, 2]).all()
    assert (selected["edges"][:, 2] < target).all()
    assert table["reconstruction_error"].max() < 1e-6


def test_candidate_screen_includes_earlier_receivers_and_energy_is_actual_WO_message():
    model, views, target = tiny_model(), example_views(), 4
    prompt = views["prompt_with_source"]
    observed = observe_token(model, prompt, views["answer_ids"], target)
    table = candidate_table(model, [observed, observed], target, 2)
    assert ((table["edges"][:, 0] == 0) & (table["edges"][:, 2] < target - 1)).any()
    current = candidate_table(model, [observed, observed], target, 1)
    assert (current["edges"][:, 2] == target - 1).all()
    for index in [0, len(table["edges"]) // 2]:
        layer, head, _, key = table["edges"][index]
        heads, kv_heads, width = model.head_layout(layer)
        value = observed["layers"][layer]["value"][0, len(prompt) + key].reshape(kv_heads, width)[head // (heads // kv_heads)].float()
        projection = model.layers[layer].self_attn.o_proj.weight[:, head * width:(head + 1) * width].float()
        message = table["route_mass"][index, 0] * (projection @ value)
        assert table["message_energy"][index, 0] == pytest.approx(float(message.square().sum().detach()), abs=1e-8, rel=1e-5)


def test_conditional_selection_rejects_large_common_effect_in_favor_of_source_difference():
    table = dict(edges=np.array([[0, 0, 2, 0], [0, 1, 2, 1], [0, 1, 2, 0]]),
        gradients=np.array([[20., 20.], [1., -1.], [.2, .2]]),
        route_mass=np.ones((3, 2)), message_energy=np.ones((3, 2)), reconstruction_error=np.zeros((2, 1)))
    conditional = select_edges(table, 1, 37, "conditional")
    magnitude = select_edges(table, 1, 37, "magnitude")
    np.testing.assert_array_equal(conditional["edges"], [[0, 1, 2, 1]])
    np.testing.assert_array_equal(magnitude["edges"], [[0, 0, 2, 0]])
    np.testing.assert_array_equal(conditional["edges"][:, :3], conditional["sham_edges"][:, :3])


def test_independent_target_ignores_unit_boundaries_and_future_suffix_and_keeps_zero_history_mask():
    model, views = tiny_model(), example_views()
    before = capture_token(model, views, 4, 2, 2, "conditional", 37)
    views["units"] = [dict(start=0, stop=7)]
    views["answer_ids"][5:] = [25, 27]
    after = capture_token(model, views, 4, 2, 2, "conditional", 37)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    vector, keys, receivers = token_vector(before)
    assert vector.shape == (schema(2, 2)["dimension"],)
    mask = vector[22:].reshape(2, 2, 7)[..., -1].astype(bool)
    np.testing.assert_array_equal(keys == -1, ~mask)
    assert (keys[mask] <= receivers[mask]).all()
    first = capture_token(model, views, 0, 2, 2, "conditional", 37)
    assert not first["reconstruction_measured"]
    assert first["edges"].shape == (0, 4)
    np.testing.assert_array_equal(first["keep"], first["joint"])


def test_position_weight_has_explicit_direction_and_beta_zero_recovers_uniform_mean():
    units = [dict(start=0, stop=3), dict(start=3, stop=4)]
    values = np.array([9., 0., 0., 7.])
    np.testing.assert_allclose(aggregate_units(values, units), [3, 3, 3, 7])
    assert aggregate_units(values, units, 1)[0] < 3
    assert aggregate_units(values, units, -1)[0] > 3
    assert aggregate_units(values, units, 1)[3] == 7


def test_token_pipeline_resumes_is_label_free_and_position_stage_is_cache_only(tmp_path):
    source, output = tmp_path / "input", tmp_path / "v2"
    pieces, annotations = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], f"token{ids[0]}")
    args = ["--mode", "token", "--input", str(source), "--output", str(output),
            "--device", "cpu", "--dtype", "float32", "--top-k", "2"]
    original_read = CaptureReader.json
    def observations_only(reader, name):
        assert name != "annotations.json"
        return original_read(reader, name)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch.object(CaptureReader, "json", observations_only):
        main([*args, "--stage", "capture"])
    assert not (output / "annotations.json").exists()
    main(["--output", str(output), "--stage", "score"])
    original = read_arrays(output / "responses/0000/node_features.npz")
    scores = read_arrays(output / "responses/0000/scores.npz")
    assert original["node_features"].shape == (5, 50)
    assert read_json(output / "coverage.json")["represented_tokens"] == 10
    assert (output / "selected_edges.csv").is_file()
    assert (output / "ranking_scope.json").is_file()
    for annotation in annotations.values():
        annotation["labels"] = [1 - value for value in annotation["labels"]]
    changed = tmp_path / "changed.json"
    write_json(changed, annotations)
    with patch("state_audit.model.load_model", side_effect=AssertionError("No model needed")):
        main([*args, "--resume", "--annotations", str(changed)])
        positions = tmp_path / "position"
        main(["--stage", "position", "--input", str(output.with_name("v2_review.zip")),
              "--output", str(positions), "--position-beta", "0"])
    for name, values in original.items():
        np.testing.assert_array_equal(values, read_arrays(output / "responses/0000/node_features.npz")[name])
    for name, values in scores.items():
        np.testing.assert_array_equal(values, read_arrays(output / "responses/0000/scores.npz")[name])
    weighted = read_arrays(positions / "responses/0000/scores.npz")
    np.testing.assert_allclose(weighted["choice_selected_unit_exp"], weighted["choice_selected_unit_mean"])
    with ZipFile(output.with_name("v2_review.zip")) as archive:
        assert "responses/0000/token_000004.npz" in archive.namelist()
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--selection", "magnitude"])


def test_interrupted_token_is_remeasured_and_completed_targets_remain_unchanged(tmp_path):
    source, output = tmp_path / "input", tmp_path / "v2"
    pieces, _ = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], str(ids[0]))
    args = ["--mode", "token", "--input", str(source), "--output", str(output),
            "--device", "cpu", "--dtype", "float32", "--top-k", "1", "--stage", "capture"]
    def interrupted(model, views, target, *rest):
        if target == 2:
            raise RuntimeError("interrupted")
        return capture_token(model, views, target, *rest)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch("experiments.native_support.message_carriers.token_capture.capture_token", side_effect=interrupted):
        with pytest.raises(RuntimeError, match="interrupted"):
            main(args)
    done = output / "responses/0000/token_000001.npz"
    original = done.read_bytes()
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())):
        main([*args, "--resume"])
    assert done.read_bytes() == original
