"""Native identities, complete original-token coverage, and one-command execution."""

import csv
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM

from state_audit.functional_capture import capture_observed, log_probability, rms_readout
from state_audit.functional_hooks import functional_hooks
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from experiments.native_support.functional_run import arguments, main
from experiments.native_support.functional_units import answer_units, capture_units


@pytest.fixture
def model():
    torch.manual_seed(37)
    torch.set_num_threads(1)
    return ModelAdapter(LlamaForCausalLM(LlamaConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)))


def test_rms_scaling_can_suppress_margin_without_direct_write(model):
    with torch.no_grad():
        model.native.lm_head.weight[:, -1] = 0
        before = torch.randn(2, 24)
        before[:, -1] = 0
        message = torch.zeros_like(before)
        message[:, -1] = 5
        logits = model.native.lm_head(model.native.model.norm(before + message))
        target = logits.argmax(-1)
        rows = rms_readout(model, before, message, target, logits)
    np.testing.assert_allclose(rows["rms_direct_margin"], 0, atol=1e-7)
    assert np.all(rows["rms_rescale_margin"] < 0)
    np.testing.assert_allclose(rows["rms_identity_error"], 0, atol=1e-7)


def test_capture_matches_native_sequence_and_root_directional_derivative(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13, 15]
    captured = capture_observed(model, prefix, phrase, [0, 0, 1, 1], 2)
    ids = prefix + phrase[:-1]
    with attention_backend(model, "sdpa"):
        embeddings = model.native.model.embed_tokens(model.input_ids(ids)).detach()
        def objective(values):
            hidden = model.native.model(inputs_embeds=values, use_cache=False).last_hidden_state
            logits = model.native.lm_head(hidden[0, len(prefix) - 1:])
            return log_probability(logits, torch.tensor(phrase)).sum()
        with torch.no_grad():
            expected = float(objective(embeddings))
            direction = torch.zeros_like(embeddings)
            direction[0, 1] = embeddings[0, 1]
            finite = float((objective(embeddings + .01 * direction) - objective(embeddings - .01 * direction)) / .02)
    assert captured["log_probability"].sum() == pytest.approx(expected, abs=2e-6)
    assert captured["prefix_root_sensitivity"][1] == pytest.approx(finite, abs=2e-4)
    np.testing.assert_allclose(captured["head_total"], captured["head_residual"] + captured["head_ffn_mediated"], atol=1e-7)
    assert captured["head_total"].shape == (3, 2, 4, 3)
    assert captured["prefix_root_sensitivity"].shape == (4,)
    assert captured["within_unit_root_sensitivity"].shape == (2,)
    np.testing.assert_allclose(captured["head_attention"].sum(-1), 1, atol=2e-7)


def test_ffn_channel_matches_native_local_jacobian(model):
    prefix, phrase = [1, 5, 7, 9], [11, 13]
    row = capture_observed(model, prefix, phrase, [0, 0, 1, 1], 2)
    with attention_backend(model, "sdpa"), functional_hooks(model) as records:
        output = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]), use_cache=False)
        logits = model.native.lm_head(output.last_hidden_state[0, 3:])
        objective = log_probability(logits, torch.tensor(phrase)).sum()
        mid, write = records[0]["mid"], records[0]["ffn"]
        upstream, downstream = torch.autograd.grad(objective, (mid, write), retain_graph=True)
        via_ffn, = torch.autograd.grad(write, mid, grad_outputs=downstream, retain_graph=True)
        torch.testing.assert_close(upstream, downstream + via_ffn)
        head = records[0]["head"][0, 3].view(4, 6)
        direction = (via_ffn[0, 3] @ model.layers[0].self_attn.o_proj.weight).view(4, 6)
        expected = (head * direction).sum(-1).detach().numpy()
    np.testing.assert_allclose(row["head_ffn_mediated"][0, 0].sum(-1), expected, atol=1e-6)


def test_native_hook_resources_restore_on_failure(model):
    original = model.layers[0].mlp.forward
    with pytest.raises(RuntimeError), functional_hooks(model):
        raise RuntimeError("stop")
    assert model.layers[0].mlp.forward == original
    assert not model.layers[0].mlp._forward_hooks
    capture_observed(model, [1, 5, 7], [9], [0, 0, 1], 2)
    assert all(parameter.requires_grad for parameter in model.native.parameters())
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_sliding_window_rows_match_native_attention():
    model = ModelAdapter(MistralForCausalLM(MistralConfig(vocab_size=47, hidden_size=24,
        intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, sliding_window=3)))
    prefix, phrase = [1, 5, 7, 9, 11], [13, 15]
    groups = list(range(len(prefix)))
    captured = capture_observed(model, prefix, phrase, groups, len(prefix))
    with torch.no_grad(), attention_backend(model, "eager"):
        native = model.native.model(input_ids=model.input_ids(prefix + phrase[:-1]),
                                    use_cache=False, output_attentions=True)
    for layer, attention in enumerate(native.attentions):
        np.testing.assert_allclose(captured["head_attention"][:, layer],
            attention[0, :, 4:6].transpose(0, 1).numpy(), atol=2e-7)


class TinyTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join({1: 'Q ', 5: 'Heading\n', 7: ' word', 9: '.'}[value] for value in ids)


def make_input(directory, answer=None):
    answer = [5, 9, 7, 7, 7, 7, 9] if answer is None else answer
    ids = [1, 1] + answer
    response = dict(id="a", source_id="s", prompt_length=2, token_ids=ids,
                    token_text=[TinyTokenizer().decode([token]) for token in ids])
    settings = dict(model="tiny", responses=[response])
    write_json(directory / "settings.json", settings)
    write_json(directory / "value_transport/capture_settings.json", settings)
    write_json(directory / "value_transport/capture/0000/sources.json",
               dict(blocks=[[0, 1]], group_ids=[0, 0] + [1] * len(answer)))
    # The audit copies these bytes only after capture. It must not parse labels.
    (directory / "annotations.json").write_text("NOT JSON")
    return response


def command(source, destination):
    return ["--input", str(source), "--output", str(destination), "--device", "cpu",
            "--dtype", "float32", "--max-unit-tokens", "2"]


def test_all_text_types_and_long_units_retain_every_original_token(tmp_path):
    response = make_input(tmp_path)
    chunks = list(capture_units(response, 0, None, 2))
    targets = [target for unit in chunks for target in range(unit["start"], unit["stop"])]
    assert targets == list(range(7))
    assert max(unit["stop"] - unit["start"] for unit in chunks) == 2
    assert chunks[-1]["punctuation_stop"] == 7
    assert len(chunks) == 5


def test_selection_is_exact_and_numbered_steps_stay_with_their_text():
    pieces = ['Prompt', '1', '.', ' Lay', ' it', ' flat.\n', '2', '.', ' Fold', ' it', '.']
    response = dict(prompt_length=1, token_text=pieces, token_ids=list(range(len(pieces))))
    assert list(answer_units(response)) == [(0, 5), (5, 10)]
    chunks = list(capture_units(response, 3, 8, 2))
    assert [(unit["start"], unit["stop"]) for unit in chunks] == [(3, 5), (5, 7), (7, 8)]


def test_run_resume_report_and_pack_never_generate_or_read_labels(tmp_path, model):
    source, destination = tmp_path / "input", tmp_path / "output"
    response = make_input(source)
    args = command(source, destination)
    with patch("state_audit.model.load_model", return_value=(model, TinyTokenizer())), \
         patch.object(model.native, "generate", side_effect=AssertionError("generated text")):
        main(args)
    summary = read_json(destination / "summary.json")
    assert summary["status"] == "captured"
    assert summary["observed_tokens"] == summary["expected_tokens"] == 7
    assert summary["completed_units"] == summary["planned_units"] == 5
    assert summary["semantic_filter"] is False and summary["new_auroc"] is None
    coverage, = read_json(destination / "coverage.json")
    assert coverage["missing_targets"] == [] and coverage["captured_tokens"] == 7
    with (destination / "observed_tokens.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["token_id"]) for row in rows] == response["token_ids"][2:]
    assert [int(row["query"]) for row in rows] == list(range(1, 8))
    assert (destination / "annotations.json").read_text() == "NOT JSON"
    assert (destination / "ffn_layers.csv").is_file()
    assert not list(destination.rglob("bank.json"))
    assert not list(destination.rglob("candidate_*.npz"))
    with patch("state_audit.model.load_model", side_effect=AssertionError("loaded model")):
        main(args + ["--resume"])
        main(["--stage", "report", "--output", str(destination)])
        main(["--stage", "pack", "--output", str(destination)])
    assert read_json(destination / "summary.json") == summary
    with ZipFile(tmp_path / "output_review.zip") as archive:
        assert "coverage.csv" in archive.namelist()
        assert "responses/0000/unit_000000/observed.npz" in archive.namelist()


def test_native_entropy_is_full_vocabulary_and_causal(model):
    prefix, targets = [1, 5, 7], [9, 11, 13]
    captured = capture_observed(model, prefix, targets, [0, 0, 1], 2)
    with torch.no_grad(), attention_backend(model, "sdpa"):
        hidden = model.forward(prefix + targets[:-1])[2:]
        logp, entropy = model.score(hidden, torch.tensor(targets))
    np.testing.assert_allclose(captured["entropy"], entropy.numpy(), atol=1e-7)
    np.testing.assert_allclose(captured["log_probability"], logp.numpy(), atol=1e-7)
    # Later target changes may affect the unit gradient, never earlier native probabilities.
    changed = capture_observed(model, prefix, [9, 15, 17], [0, 0, 1], 2)
    np.testing.assert_allclose(captured["entropy"][:2], changed["entropy"][:2], atol=1e-7)
    np.testing.assert_allclose(captured["log_probability"][:1], changed["log_probability"][:1], atol=1e-7)


def test_interruption_reports_missing_tokens_and_resume_fills_only_missing_units(tmp_path, model):
    from experiments.native_support.functional_run import capture_unit
    source, destination = tmp_path / "input", tmp_path / "output"
    make_input(source, answer=[9, 9])
    args = command(source, destination)
    def interrupted(*values):
        if values[-1]["start"] == 1:
            raise RuntimeError("interrupted capture")
        return capture_unit(*values)
    with patch("state_audit.model.load_model", return_value=(model, TinyTokenizer())), \
         patch("experiments.native_support.functional_run.capture_unit", side_effect=interrupted), \
         pytest.raises(RuntimeError, match="interrupted capture"):
        main(args)
    with patch("state_audit.model.load_model", side_effect=AssertionError("loaded model")):
        main(["--stage", "report", "--output", str(destination)])
    summary = read_json(destination / "summary.json")
    assert summary["status"] == "partial_capture" and summary["missing_tokens"] == 1
    assert read_json(destination / "coverage.json")[0]["missing_targets"] == [1]
    first = destination / "responses/0000/unit_000000/observed.npz"
    original = first.read_bytes()
    # A stale rejected bank is irrelevant to the native v3 path.
    write_json(destination / "responses/0000/unit_000001/bank.json",
               dict(valid=False, reason="duplicate_candidate_tokens"))
    with patch("state_audit.model.load_model", return_value=(model, TinyTokenizer())), \
         patch("experiments.native_support.functional_run.capture_unit", wraps=capture_unit) as capture:
        main(args + ["--resume"])
    assert capture.call_count == 1 and first.read_bytes() == original
    assert read_json(destination / "summary.json")["observed_tokens"] == 2


def test_explicit_budget_is_not_reported_as_full_answer_coverage(tmp_path, model):
    source, destination = tmp_path / "input", tmp_path / "output"
    make_input(source)
    with patch("state_audit.model.load_model", return_value=(model, TinyTokenizer())):
        main(command(source, destination) + ["--max-units", "1"])
    summary = read_json(destination / "summary.json")
    assert summary["observed_tokens"] == 1 and summary["unselected_tokens"] == 6
    assert summary["missing_tokens"] == 0


def test_empty_selection_is_packaged_but_does_not_claim_success(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    make_input(source)
    with patch("state_audit.model.load_model", side_effect=AssertionError("loaded model")), \
         pytest.raises(SystemExit, match="capture incomplete"):
        main(command(source, destination) + ["--start-target", "100"])
    assert read_json(destination / "summary.json")["status"] == "no_selected_tokens"
    assert (tmp_path / "output_review.zip").exists()


def test_completed_capture_token_corruption_is_not_silently_skipped(tmp_path, model):
    source, destination = tmp_path / "input", tmp_path / "output"
    make_input(source, answer=[9])
    with patch("state_audit.model.load_model", return_value=(model, TinyTokenizer())):
        main(command(source, destination))
    path = destination / "responses/0000/unit_000000/observed.npz"
    arrays = read_arrays(path)
    arrays["target_ids"][0] = 7
    write_arrays(path, **arrays)
    with patch("state_audit.model.load_model", side_effect=AssertionError("loaded model")), \
         pytest.raises(ValueError, match="target IDs differ"):
        main(command(source, destination) + ["--resume"])


def test_v2_output_is_preserved_and_cannot_resume_as_v3(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    make_input(source)
    write_json(destination / "protocol.json", dict(version=2))
    path = destination / "protocol.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="settings changed"):
        main(command(source, destination) + ["--resume"])
    assert path.read_bytes() == before


def test_launcher_forwards_arguments_and_failure_from_any_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "experiments/native_support/run_functions.sh"
    executable = tmp_path / "python fixture"
    executable.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                          'print(json.dumps(dict(cwd=os.getcwd(), args=sys.argv[1:])))\n'
                          'raise SystemExit(23)\n')
    executable.chmod(0o755)
    result = subprocess.run(["bash", str(launcher), "--device", "cpu"], cwd=tmp_path,
        env={**os.environ, "FUNCTION_AUDIT_PYTHON": str(executable)}, text=True, capture_output=True)
    assert result.returncode == 23, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation["cwd"] == str(root)
    assert invocation["args"][:3] == ["-u", "main.py", "transport-functions"]
    parsed = arguments(invocation["args"][3:])
    assert parsed.stage == "run" and parsed.resume is True
    assert parsed.output.name == "function_audit_v3"
    assert parsed.device == "cpu" and parsed.response_ids == ["12219"]
