"""Value-path integration and mathematical invariants, not natural AUROC evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.storage import read_arrays, read_json, write_json
from state_audit.value_path_capture import iter_value_paths
from state_audit.value_paths import value_path_hooks
from test_dynamics_pipeline import response, write_cohort
from transformers import MistralConfig, MistralForCausalLM

from experiments.native_support.transport import main, root_groups
from experiments.native_support.transport_readout import score_answer

pytest_plugins = ("test_dynamics_pipeline",)


def options(output):
    return ["--output", str(output), "--device", "cpu", "--dtype", "float32",
            "--choices", "3", "--block-tokens", "3", "--gradient-batch", "2", "--cpu-threads", "1"]


def collect(model, item, targets, batch=1, block_tokens=3):
    groups, blocks = root_groups(item, None, [0], block_tokens)
    return list(iter_value_paths(model, item["token_ids"], item["prompt_length"], targets,
        groups, len(blocks) + 3, choices=3, gradient_batch=batch))


def test_native_forward_is_preserved_and_root_contributions_close_margin(tiny_model):
    item = response("native")
    rows = collect(tiny_model, item, [0, 2, 5])
    with torch.no_grad(), attention_backend(tiny_model, "sdpa"):
        hidden = tiny_model.forward(item["token_ids"][:11])
        logits = tiny_model.native.lm_head(hidden).float().numpy()
    for row in rows:
        query = int(row["query"])
        np.testing.assert_allclose(row["candidate_logits"], logits[query, row["candidate_ids"]], atol=1e-7)
        np.testing.assert_allclose(row["root_token_choice"].sum(0), row["margin"], atol=2e-7)
        np.testing.assert_allclose((row["root_positive"] - row["root_negative"]).sum(0), row["margin"], atol=2e-7)
        assert row["head_positive"].shape[:2] == (3, 4)
        assert row["ffn_choice"].shape == (3, 2)


def test_sliding_window_diagnostics_match_native_attention():
    model = ModelAdapter(MistralForCausalLM(MistralConfig(
        vocab_size=41, hidden_size=24, intermediate_size=36, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, sliding_window=3,
    )))
    item = response("window")
    rows = collect(model, item, [0, 3])
    with torch.no_grad():
        native = model.native.model(input_ids=model.input_ids(item["token_ids"][:9]),
                                    use_cache=False, output_attentions=True)
    for row in rows:
        query = int(row["query"])
        for layer, attention in enumerate(native.attentions):
            np.testing.assert_allclose(row["attention"][layer],
                attention[0, :, query, :query + 1].numpy(), atol=2e-7)
        assert not row["attention"][..., :query - 2].any()
        np.testing.assert_allclose(row["root_token_choice"].sum(0), row["margin"], atol=2e-7)


def test_future_tokens_resume_and_vjp_batch_do_not_change_provenance(tiny_model):
    item = response("causal")
    together = collect(tiny_model, item, [0, 3, 8], batch=2)
    single = collect(tiny_model, item, [3])
    altered = {**item, "token_ids": item["token_ids"][:10] + [35] * 6}
    poisoned = collect(tiny_model, altered, [0, 3, 8])
    for field in ("margin", "root_token_choice", "root_positive", "head_positive", "ffn_choice", "attention"):
        np.testing.assert_allclose(together[1][field], single[0][field], atol=2e-7, rtol=2e-5)
        np.testing.assert_allclose(together[1][field], poisoned[1][field], atol=2e-7, rtol=2e-5)


def test_source_partition_does_not_erase_opposing_token_contributions(tiny_model):
    item = response("blocks")
    fine = collect(tiny_model, item, [0, 1], block_tokens=1)
    coarse = collect(tiny_model, item, [0, 1], block_tokens=6)
    # Rectification precedes grouping, making source totals partition-invariant.
    for left, right in zip(fine, coarse):
        for field in ("root_positive", "root_negative"):
            np.testing.assert_allclose(left[field][:6].sum(0), right[field][:1].sum(0), atol=2e-7)
    fine_score, _ = score_answer(item, fine, 6, None)
    coarse_score, _ = score_answer(item, coarse, 1, None)
    np.testing.assert_allclose(fine_score["root_route"], coarse_score["root_route"], atol=1e-6)


def test_hooks_and_parameter_flags_restore_when_iterator_closes(tiny_model):
    item = response("close")
    groups, blocks = root_groups(item, None, [0], 3)
    flags = [p.requires_grad for p in tiny_model.native.parameters()]
    original = tiny_model.layers[0].mlp.forward
    iterator = iter_value_paths(tiny_model, item["token_ids"], 6, [0, 1], groups, len(blocks) + 3)
    next(iterator)
    iterator.close()
    assert tiny_model.layers[0].mlp.forward == original
    assert [p.requires_grad for p in tiny_model.native.parameters()] == flags
    assert all(p.grad is None for p in tiny_model.native.parameters())
    assert not tiny_model.layers[0].self_attn.q_proj._forward_hooks
    with pytest.raises(RuntimeError), value_path_hooks(tiny_model, torch.tensor([5])):
        raise RuntimeError("interrupted")
    assert tiny_model.layers[0].mlp.forward == original


def test_capture_score_evaluate_and_labels_are_separate(tmp_path, tiny_model):
    write_cohort(tmp_path, ["a", "b"])
    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids: str(ids[0]))
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", return_value=(tiny_model, tokenizer)):
        main(["--stage", "run", *options(tmp_path)])
    directory = tmp_path / "value_transport"
    summary = read_json(directory / "summary.json")
    assert summary["scored_tokens"] == 20
    assert not summary["labels_used_for_scoring"] and not summary["parameter_fitting"]
    assert summary["evaluation"]["status"] == "evaluated"
    assert summary["max_abs_ledger_error"] < 1e-6
    scores = read_arrays(directory / "responses/0000/scores.npz")
    assert np.isfinite(scores["risk"]).all()
    assert np.all(np.abs(scores["risk"]) <= 1.000001)
    assert (directory / "source_choices.csv").is_file()
    assert not (tmp_path / "source_transport").exists()
    labels = read_json(tmp_path / "annotations.json")
    labels["a"]["labels"] = [1 - label for label in labels["a"]["labels"]]
    write_json(tmp_path / "annotations.json", labels)
    with patch("state_audit.model.load_model", side_effect=AssertionError("score loaded LLM")), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("score loaded tokenizer")):
        main(["--stage", "score", *options(tmp_path)])
    changed = read_arrays(directory / "responses/0000/scores.npz")
    for name in scores:
        np.testing.assert_array_equal(changed[name], scores[name])
    with patch("state_audit.model.load_model", side_effect=AssertionError("resume loaded LLM")), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        main(["--stage", "run", "--resume", *options(tmp_path)])
    with patch("experiments.native_support.transport.score_answer", side_effect=AssertionError("evaluate scored again")):
        main(["--stage", "evaluate", *options(tmp_path)])


def test_missing_annotations_keeps_scores_without_inventing_metrics(tmp_path, tiny_model):
    write_cohort(tmp_path, ["a"])
    (tmp_path / "annotations.json").unlink()
    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids: str(ids[0]))
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", return_value=(tiny_model, tokenizer)):
        main(["--stage", "run", *options(tmp_path)])
    summary = read_json(tmp_path / "value_transport/summary.json")
    assert summary["evaluation"]["status"] == "unavailable"
