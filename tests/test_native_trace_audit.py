"""Scientific input alignment, fixed-set reanchor and an actual tiny-model report."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from state_audit.model.adapter import ModelAdapter
from state_audit.storage import read_json, write_json
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from experiments.native_trace_audit.inputs import (
    compile_panels,
    inventory,
    partition_sources,
    query_plans,
)
from experiments.native_trace_audit.report import write_report
from experiments.native_trace_audit.run import EXAMPLES, capture_panel
from experiments.native_trace_audit.timeline import reanchor_rows


class PieceTokenizer:
    """Lossless text pieces solely for testing the experiment's prefix compiler."""

    def __init__(self):
        self.pieces = []

    def encode(self, text, add_special_tokens=False):
        result = []
        for piece in re.findall(r"\s*\S+", text):
            if piece not in self.pieces:
                self.pieces.append(piece)
            result.append(self.pieces.index(piece))
        return result

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[index] for index in ids)


@pytest.fixture
def byte_tokenizer():
    """Real byte decoding and whitespace cleanup, without downloading a model."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {piece: index for index, piece in enumerate(alphabet)}
    merges = [("Ġ", char) for char in ".abcfho"]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, clean_up_tokenization_spaces=True
    )


def archived_context(tokenizer, text, case_id="14315_headwear_scope"):
    ids = tokenizer.encode(text, add_special_tokens=False)
    candidate_texts = [" caps", " a headdress"]
    return {
        "case_id": case_id,
        "prefix_ids": ids,
        "token_text": [tokenizer.decode([token]) for token in ids],
        "candidates": [
            tokenizer.encode(text, add_special_tokens=False) for text in candidate_texts
        ],
        "reviewed_case": {"candidates": candidate_texts},
    }


@pytest.mark.parametrize("case_id", ["14315_headwear_scope", "14375_onion_stage"])
@pytest.mark.parametrize("prefix_text", ["Hello . They wore", "中文 They wore"])
def test_nonadditive_decoding_keeps_saved_ids(byte_tokenizer, case_id, prefix_text):
    context = archived_context(byte_tokenizer, prefix_text, case_id)
    prefix = context["prefix_ids"]
    assert byte_tokenizer.decode(prefix, clean_up_tokenization_spaces=False) != (
        "".join(context["token_text"])
    )
    with patch.object(byte_tokenizer, "encode", wraps=byte_tokenizer.encode) as encode:
        natural, control = compile_panels(context, byte_tokenizer)
    assert natural["token_ids"] is prefix
    assert natural["candidates"] is context["candidates"]
    assert control["token_ids"][: len(prefix)] == prefix
    # Only new diagnostic text is encoded; archived prefixes are never retokenized.
    assert [call.args[0] for call in encode.call_args_list] == [
        control["suffix"],
        *control["candidate_texts"],
    ]


def test_real_tokenizer_mismatch_reports_position_and_id(byte_tokenizer):
    context = archived_context(byte_tokenizer, "Hello . They wore")
    wrong_id = byte_tokenizer.encode("X", add_special_tokens=False)[0]
    context["prefix_ids"][0] = wrong_id
    with pytest.raises(ValueError, match=f"position 0, token ID {wrong_id}"):
        compile_panels(context, byte_tokenizer)


def test_saved_candidate_mismatch_is_rejected(byte_tokenizer):
    context = archived_context(byte_tokenizer, "Hello . They wore")
    context["candidates"][0] = byte_tokenizer.encode(" shoes", add_special_tokens=False)
    with pytest.raises(ValueError, match="saved candidate 0 token IDs"):
        compile_panels(context, byte_tokenizer)


@pytest.mark.parametrize("case_id", ["14315_headwear_scope", "14375_onion_stage"])
def test_binding_panels_preserve_native_prefix_and_do_not_invent_observed_token(
    case_id,
):
    tokenizer = PieceTokenizer()
    ids = tokenizer.encode("Prompt source constraint. They wore")
    candidates = [tokenizer.encode(text) for text in [" caps", " a headdress"]]
    context = {
        "case_id": case_id,
        "side": "supported",
        "prefix_ids": ids,
        "prompt_length": 3,
        "token_text": [tokenizer.decode([token]) for token in ids],
        "candidates": candidates,
        "roles": {"scope": [0], "supported_value": [1], "value_source": [2]},
        "reviewed_case": {"candidates": [" caps", " a headdress"]},
    }
    natural, control = compile_panels(context, tokenizer)
    assert natural["token_ids"] == ids
    assert control["token_ids"][: len(ids)] == ids
    assert not control["natural"] and control["suffix"]
    plans = query_plans(context, control, [], 8, 10, 2)
    assert len(plans) == 1 and plans[0]["observed_id"] is None
    assert plans[0]["candidate_ids"][0] != plans[0]["candidate_ids"][1]


def test_real_archived_prefixes_have_verified_disjoint_roles_and_same_prompt():
    manifest = read_json(EXAMPLES)
    assert len(inventory(manifest)) == 4
    for context in manifest["contexts"]:
        ids = context["prefix_ids"]
        groups = partition_sources(context, ids, [128000, 128006, 128007, 128009], 10)
        assert len(groups) == len(ids)
        assert set(groups) <= set(range(8))
        assert groups[0] == 7
        assert groups[-1] == 6
        panel = {"token_ids": ids, "candidates": context["candidates"], "natural": True}
        plans = query_plans(context, panel, [128000, 128006, 128007, 128009], 8, 10, 8)
        assert len(plans) == 9
        assert [p["dependencies"] for p in plans] == [False] * 8 + [True]
        assert plans[-2]["observed_id"] == ids[-1]
        assert plans[-1]["query"] == len(ids) - 1


def test_role_mismatch_is_not_silently_called_evidence():
    manifest = read_json(EXAMPLES)
    manifest["contexts"][0]["roles"]["scope"] = [1, 2]
    with pytest.raises(ValueError, match="reviewed quote"):
        inventory(manifest)


def trace(query, mass):
    return {
        "query": np.asarray(query),
        "attention": np.asarray(mass, dtype=float)[None, None],
    }


def test_reanchor_requires_bidirectional_change_and_preserves_unknown_prefix():
    traces = [
        trace(4, [0.1, 0.1, 0.1, 0.35, 0.35]),
        trace(5, [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]),
        trace(6, [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]),
        trace(7, [0.4, 0.3, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02]),
    ]
    rows = reanchor_rows(traces, 3, [], list(range(8)), recent=5)
    assert len(rows) == 1 and rows[0]["confirmed"]
    assert rows[0]["predicted_position"] == 8
    assert not rows[0]["prior_state_observed"]
    assert not reanchor_rows(traces[:3], 3, [], list(range(8)), recent=5)


def test_moving_local_boundary_alone_does_not_create_reanchor():
    traces = []
    for query in range(5, 10):
        mass = np.zeros(query + 1)
        mass[3] = 0.7
        mass[query] = 0.3
        traces.append(trace(query, mass))
    rows = reanchor_rows(traces, 2, [], list(range(10)), recent=5)
    assert rows and not any(row["confirmed"] for row in rows)


def test_tiny_capture_resume_and_offline_report(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = ModelAdapter(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=31,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
    )
    context = {
        "case_id": "software_fixture",
        "source_id": "fixture",
        "side": "supported",
        "prefix_ids": list(range(1, 14)),
        "prompt_length": 6,
        "roles": {"scope": [1], "supported_value": [2], "value_source": [3]},
        "token_text": [str(i) for i in range(1, 14)],
        "history_status": "synthetic_random_weight_model",
    }
    panel = {
        "name": "natural",
        "token_ids": context["prefix_ids"],
        "candidates": [[5], [7]],
        "candidate_texts": ["5", "7"],
        "natural": True,
        "suffix": "",
        "meaning": "Software fixture, not a natural claim",
    }
    args = SimpleNamespace(
        output=tmp_path,
        window=3,
        recent=3,
        receiver_budget=2,
        prefill_chunk_size=4,
        resume=True,
    )
    tokenizer = SimpleNamespace(all_special_ids=[0])
    settings = {
        "model": "random_weight_fixture",
        "purpose": "software_verification_only",
        "window": 3,
        "recent": 3,
        "inputs": {"contexts": [context]},
    }
    write_json(tmp_path / "settings.json", settings)
    capture_panel(model, tokenizer, context, panel, args)
    paths = list(tmp_path.glob("cases/*/*/*/query_*.npz"))
    times = {path: path.stat().st_mtime_ns for path in paths}
    with patch.object(
        model.native.model, "forward", side_effect=AssertionError("resume reran model")
    ):
        capture_panel(model, tokenizer, context, panel, args)
        summary = write_report(tmp_path)
    assert len(paths) == 4
    assert all(path.stat().st_mtime_ns == times[path] for path in paths)
    assert summary["completed_queries"] == 4
    assert summary["max_abs_ledger_error"] < 1e-6
    assert (tmp_path / "review.tar.gz").exists()
    assert "后续读取与 FFN" in (tmp_path / "report.html").read_text()
    assert "Top 32" in summary["dependency_csv"]
    assert not list(tmp_path.rglob("*.partial.*"))
