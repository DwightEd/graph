"""Counterfactual identities, native replay equivalence, and cohort isolation."""

import numpy as np
import pytest
import torch
from unittest.mock import patch
from transformers.cache_utils import DynamicCache

from state_audit.model.replay import attention_backend
from state_audit.storage import read_arrays, read_json, write_json
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.message_carriers.run import main as carriers_main
from experiments.native_support.source_mediation.run import main

from experiments.native_support.source_mediation.decomposition import factorial, gate_scores
from experiments.native_support.source_mediation.native import donor_world, measure, replay_world
from test_evidence_contrast import tiny_model
from test_message_carriers import make_contrast_input


def test_factorial_separates_additive_redundant_and_synergistic_games():
    baseline = np.array([2., 2., 2.])
    direct, history, interaction = np.array([3., 3., 3.]), np.array([4., 4., 4.]), np.array([0., -2., 5.])
    worlds = np.stack([np.stack([baseline, baseline+history]),
                       np.stack([baseline+direct, baseline+direct+history+interaction])])
    values = factorial(worlds)
    np.testing.assert_array_equal(values["direct"], direct)
    np.testing.assert_array_equal(values["history"], history)
    np.testing.assert_array_equal(values["interaction"], interaction)
    np.testing.assert_array_equal(values["context_shapley"] + values["history_shapley"], values["total"])


def test_joint_head_residual_is_not_sum_of_absolute_head_effects():
    keep = np.array([[-1., -1.], [-2., -2.]])
    effects = np.array([[[1., 2.], [-1., 3.]], [[0., 0.], [0., 0.]]])
    single = keep[:, None] - effects
    joint = keep - effects.sum(axis=1)
    scores, values = gate_scores(keep, joint, joint, single)
    np.testing.assert_array_equal(values["coalition"], 0)
    np.testing.assert_allclose(scores["head_cancellation"], [1., 0.])
    changed = joint.copy()
    changed[0] -= 2
    scores, values = gate_scores(keep, changed, joint, single)
    np.testing.assert_array_equal(scores["head_coalition_size"], 2)
    empty = single[:, :0]
    scores, _ = gate_scores(keep, keep, keep, empty)
    np.testing.assert_array_equal(scores["head_cancellation"], 0)


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_batched_hybrid_queries_equal_single_queries_and_native_diagonals(family):
    model = tiny_model(family)
    views = dict(prompt_with_source=[1, 3, 5, 7, 9, 11], answer_ids=[13, 15, 17, 19, 21],
                 removed_prompt_positions=[1, 2])
    batch, heads = measure(model, views, 3, 3, True)
    serial, _ = measure(model, views, 2, 1)
    assert batch["identity_max_error"].max() < 1e-6
    np.testing.assert_allclose(batch["worlds"], serial["worlds"], atol=1e-6)
    assert heads["heads_01"].shape == (5, 2, 2, 8)
    # No strictly earlier answer K/V exists for the first two target predictions.
    np.testing.assert_allclose(batch["worlds"][:, 0, :2], batch["worlds"][:, 1, :2], atol=1e-6)
    assert all(parameter.grad is None for parameter in model.native.parameters())
    assert model.native.config._attn_implementation == "eager"


def test_source_blocking_erases_content_access_without_changing_positions():
    model = tiny_model()
    prefix, answer = [1, 3, 5, 7, 9, 11], [13, 15, 17, 19, 21]
    source_mask = np.array([False, True, True, False, False, False])
    changed = [1, 25, 27, 7, 9, 11]
    _, original = donor_world(model, prefix, answer, source_mask, 0, 3)
    _, modified = donor_world(model, changed, answer, source_mask, 0, 2)
    np.testing.assert_allclose(original, modified, atol=1e-6)


def test_hybrid_replay_has_no_future_answer_access_and_patches_keys_as_well_as_values():
    model = tiny_model()
    views = dict(prompt_with_source=[1, 3, 5, 7], answer_ids=[9, 11, 13, 15, 17],
                 removed_prompt_positions=[1, 2])
    original, _ = measure(model, views, 3, 2)
    changed, _ = measure(model, {**views, "answer_ids": [9, 11, 13, 19, 21]}, 3, 2)
    np.testing.assert_allclose(original["worlds"][..., :3], changed["worlds"][..., :3], atol=1e-6)
    assert np.max(np.abs(original["worlds"][0, 0, 2:] - original["worlds"][0, 1, 2:])) > 1e-7


def test_native_rejects_source_predictor_and_truncation():
    model = tiny_model()
    with pytest.raises(ValueError, match="final prompt query"):
        measure(model, dict(prompt_with_source=[1, 3], answer_ids=[5], removed_prompt_positions=[1]), 2, 2)
    with pytest.raises(ValueError, match="no truncation"):
        measure(model, dict(prompt_with_source=[1, 3], answer_ids=[5]*130, removed_prompt_positions=[]), 2, 2)


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_hybrid_matches_independent_unmodified_native_attention_with_mixed_cache(family):
    model = tiny_model(family)
    prefix, answer = [1, 3, 5, 7], [9, 11, 13, 15, 17]
    source_mask = np.array([False, True, True, False])
    states = [donor_world(model, prefix, answer, source_mask, state, 3)[0] for state in (0, 1)]
    tokens = prefix + answer[:-1]
    for prompt_state, history_state in ((0, 1), (1, 0)):
        actual, _ = replay_world(model, states, prefix, answer, source_mask, (prompt_state, history_state), 3)
        expected = []
        for target, token in enumerate(answer):
            query = len(prefix)-1+target
            cache = DynamicCache()
            for layer in range(len(model.layers)):
                fields = []
                for field in (0, 1):
                    prompt_values = states[prompt_state][layer][field][:, :min(len(prefix), query)]
                    history_values = states[history_state][layer][field][:, len(prefix):query]
                    fields.append(torch.cat([prompt_values, history_values], dim=1)[None])
                cache.update(*fields, layer)
            allowed = torch.ones((1, query+1), dtype=torch.long)
            if not prompt_state:
                allowed[0, :len(prefix)] = torch.as_tensor(~source_mask)
            with torch.no_grad(), attention_backend(model, "eager"):
                hidden = model.native.model(input_ids=model.input_ids([tokens[query]]),
                    attention_mask=allowed, past_key_values=cache, use_cache=True).last_hidden_state[0, 0]
                expected.append(float(model.native.lm_head(hidden).float().log_softmax(-1)[token]))
        np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("mode", ["unit", "token"])
def test_cached_pipeline_uses_no_labels_until_evaluation_and_preserves_inputs(tmp_path, mode):
    source, capture, output = tmp_path / "input", tmp_path / "carrier", tmp_path / "output"
    pieces, annotations = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], str(ids[0]))
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())):
        carriers_main(["--mode", mode, "--input", str(source), "--output", str(capture),
                       "--device", "cpu", "--dtype", "float32", "--top-k", "2"])
    archive = capture.with_name("carrier_review.zip")
    original = archive.read_bytes()
    read_bytes = CaptureReader.bytes
    def guarded_read(reader, name):
        if name == "annotations.json":
            assert (output / "coverage.json").exists()
        return read_bytes(reader, name)
    args = ["--input", str(archive), "--output", str(output), "--mode", "cache"]
    with patch("state_audit.model.load_model", side_effect=AssertionError("Cache experiment loaded a model")), \
            patch.object(CaptureReader, "bytes", guarded_read):
        main([*args, "--stage", "score"])
        assert not (output / "annotations.json").exists()
        main(["--output", str(output), "--stage", "evaluate"])
    first = read_arrays(output / "responses/0000/scores.npz")
    carrier = read_arrays(capture / "responses/0000/scores.npz")
    selected = "source_selected" if mode == "unit" else "token_source_selected"
    np.testing.assert_allclose(first["gate_source_controlled"], carrier[selected], atol=1e-6)
    np.testing.assert_array_equal(first["raw_route"], carrier["raw_route"])
    edges = read_arrays(output / "responses/0000/head_effects.npz")
    if mode == "unit":
        assert (edges["receiver"] == -1).all()
    for annotation in annotations.values():
        annotation["labels"] = [1 - value for value in annotation["labels"]]
    flipped = tmp_path / "flipped.json"
    write_json(flipped, annotations)
    main([*args, "--resume", "--annotations", str(flipped)])
    for name, value in first.items():
        np.testing.assert_array_equal(value, read_arrays(output / "responses/0000/scores.npz")[name])
    assert archive.read_bytes() == original
    assert output.with_name("output_review.zip").is_file()


def test_native_pipeline_reuses_complete_measurements_and_freezes_protocol(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    pieces, _ = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], str(ids[0]))
    args = ["--input", str(source), "--output", str(output), "--mode", "native",
            "--device", "cpu", "--dtype", "float32", "--identity-tolerance", "0.00001", "--save-heads"]
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())):
        main([*args, "--stage", "capture"])
    assert not (output / "coverage.json").exists()
    assert not (output / "annotations.json").exists()
    with patch("state_audit.model.load_model", side_effect=AssertionError("Completed native worlds were repeated")):
        main([*args, "--resume", "--query-chunk-size", "2"])
    values = read_arrays(output / "responses/0000/worlds.npz")
    assert values["identity_max_error"].max() < 1e-6
    assert (output / "responses/0000/head_readouts.npz").is_file()
    assert read_json(output / "summary.json")["protocol"]["primary_candidate"] == "mediated_direct"
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--identity-tolerance", ".1"])


def test_failed_identity_measurement_is_saved_but_never_completed_or_scored(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    pieces, _ = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], str(ids[0]))
    broken = dict(worlds=np.zeros((2, 2, 5)), identity_max_error=np.ones(2),
                  token_id=np.arange(5), target=np.arange(5))
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch("experiments.native_support.source_mediation.native.measure", return_value=(broken, {})), \
            pytest.raises(ValueError, match="identity error"):
        main(["--input", str(source), "--output", str(output), "--mode", "native", "--device", "cpu"])
    assert (output / "responses/0000/worlds.npz").is_file()
    assert not (output / "responses/0000/capture_complete.json").exists()
    assert not (output / "coverage.json").exists()
