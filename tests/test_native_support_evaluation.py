"""Missing labels are explicit; official token labels are derived, never invented."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from state_audit.demo import build_demo
from state_audit.storage import read_json, write_arrays, write_json
from transformers import AutoTokenizer

from experiments.native_support.evaluate import evaluate, ranking
from experiments.native_support.ragtruth import (
    encode_response,
    prepare_official,
    select_responses,
)
from experiments.native_support.run import main


@pytest.fixture
def official_fixture(tmp_path):
    _, model = build_demo(tmp_path / "model_fixture", "llama")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = {"source_id": "s1", "task_type": "QA",
              "prompt": "question: What does Mira wear?\npassages:\nMira wears a blue coat.\noutput:",
              "source_info": {"question": "What does Mira wear?", "passages": "Mira wears a blue coat."}}
    rows = []
    for index in range(4):
        text = "Mira wears a red coat." if index % 2 else "Mira wears a blue coat."
        labels = []
        if index % 2:
            start = text.index("red")
            labels = [{"start": start, "end": start + 3, "text": "red"}]
        rows.append({"id": str(index), "source_id": "s1", "split": "test",
                     "model": "llama-2-7b-chat", "response": text, "labels": labels})
    (dataset / "source_info.jsonl").write_text(json.dumps(source) + "\n")
    (dataset / "response.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    args = SimpleNamespace(dataset=dataset, task="QA", split="test", generator="llama-2-7b-chat",
                           limit=4, balanced=True)
    return args, model, rows, {"s1": source}


def test_absent_default_annotations_are_unavailable_not_zero_labels(tmp_path):
    previous = {"previous_result": "preserve"}
    write_json(tmp_path / "evaluation.json", previous)
    result = evaluate(tmp_path)
    assert result["status"] == "unavailable"
    assert result["auroc"] is None and result["ap"] is None
    assert not (tmp_path / "annotations.json").exists()
    assert read_json(tmp_path / "evaluation.json") == previous


def test_original_placeholder_command_has_no_traceback_or_model_load(tmp_path, capsys):
    with patch("state_audit.model.load_model", side_effect=AssertionError("GPU load forbidden")):
        main(["--stage", "evaluate", "--output", str(tmp_path),
              "--annotations", str(tmp_path / "token_labels.json")])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unavailable"
    assert result["reason"] == "missing_token_annotations"


def test_official_answer_preparation_has_real_aligned_labels(official_fixture):
    args, model, _, _ = official_fixture
    manifest, annotations, cohort = prepare_official(args, str(model))
    assert len(manifest["responses"]) == 4
    assert cohort["labels_used_for_selection"] is True
    assert cohort["labels_used_for_scoring"] is False
    for response in manifest["responses"]:
        label = annotations[response["id"]]
        assert "labels" not in response
        assert response["token_ids"][response["prompt_length"]:] == label["token_ids"]
        assert sum(label["labels"]) == int(response["id"]) % 2
        assert label["annotation_origin"] == "RAGTruth/response.jsonl"


def test_labels_change_no_model_input_for_a_fixed_official_answer(official_fixture):
    _, model, rows, sources = official_fixture
    tokenizer = AutoTokenizer.from_pretrained(model)
    row = rows[1]
    response, annotation = encode_response(row, sources["s1"], tokenizer)
    changed = dict(row, labels=[])
    other_response, other_annotation = encode_response(changed, sources["s1"], tokenizer)
    assert response == other_response
    assert annotation["labels"] != other_annotation["labels"]


def test_adjacent_annotation_spans_keep_distinct_onsets(official_fixture):
    _, model, rows, sources = official_fixture
    row = dict(rows[0], labels=[{"start": 0, "end": 4, "text": "Mira"},
                                {"start": 5, "end": 10, "text": "wears"}])
    _, annotation = encode_response(row, sources["s1"], AutoTokenizer.from_pretrained(model))
    assert annotation["labels"][:2] == [1, 1]
    assert annotation["span_onsets"][:2] == [True, True]


def test_invalid_span_text_cannot_be_silently_relabelled(official_fixture):
    _, model, rows, sources = official_fixture
    row = dict(rows[0], labels=[{"start": 0, "end": 4, "text": "wrong"}])
    with pytest.raises(ValueError, match="label text disagrees"):
        encode_response(row, sources["s1"], AutoTokenizer.from_pretrained(model))


def test_balanced_selection_is_explicit_and_cannot_invent_missing_class(official_fixture):
    _, _, rows, sources = official_fixture
    with pytest.raises(ValueError, match="Not enough official"):
        select_responses(rows[:1], sources, task="QA", split="test",
                         generator="llama-2-7b-chat", limit=4, balanced=True)
    result = ranking(np.array([0, 0]), np.array([0.1, 0.4]))
    assert result["auroc"] is None and result["auroc_status"] == "requires_both_classes"


def test_prepare_does_not_overwrite_existing_different_cache(official_fixture, tmp_path):
    args, model, _, _ = official_fixture
    output = tmp_path / "output"
    write_json(output / "settings.json", {"original": "keep"})
    with pytest.raises(ValueError, match="settings changed"):
        main(["--stage", "prepare", "--dataset", str(args.dataset), "--model", str(model),
              "--balanced", "--output", str(output), "--resume"])
    assert read_json(output / "settings.json") == {"original": "keep"}
    assert not (output / "annotations.json").exists()


def test_real_cli_prepares_scores_and_evaluates_without_placeholder(official_fixture, tmp_path, capsys):
    import torch

    torch.set_num_threads(1)
    args, model, _, _ = official_fixture
    output = tmp_path / "pilot"
    command = ["--dataset", str(args.dataset), "--model", str(model), "--balanced",
               "--limit", "4", "--output", str(output), "--device", "cpu", "--dtype", "float32",
               "--prefill-chunk-size", "8", "--resume"]
    main(command)
    result = json.loads(capsys.readouterr().out)
    evaluated = result["evaluation"]
    assert result["evaluation_performed_by_this_stage"] is True
    assert evaluated["status"] == "evaluated"
    assert evaluated["methods"]["routing_imbalance"]["all_error"]["positives"] == 2
    assert evaluated["methods"]["routing_imbalance"]["all_error"]["auroc"] is not None
    assert set(evaluated["methods"]) == {"routing_imbalance", "attention_displacement", "entropy"}
    assert result["state_fitting"] is False
    assert (output / "annotations.json").is_file()
    with patch("state_audit.model.load_model", side_effect=AssertionError("GPU load forbidden")):
        main(command + ["--query-chunk-size", "3", "--compress-cache"])
        assert json.loads(capsys.readouterr().out) == result
        main(["--stage", "evaluate", "--output", str(output)])
    repeated = json.loads(capsys.readouterr().out)
    assert repeated == evaluated


def test_valid_token_mask_excludes_controls_and_keeps_annotation_onsets(tmp_path):
    response = {"id": "1", "source_id": "s1"}
    write_json(tmp_path / "settings.json", {"responses": [response]})
    write_arrays(tmp_path / "responses/0000/scores.npz", token_id=np.array([1, 2, 3, 4]),
                 risk=np.array([0.4, 0.5, -0.1, 100.0]), direct_risk=np.zeros(4))
    write_json(tmp_path / "annotations.json", {"1": {
        "token_ids": [1, 2, 3, 4], "labels": [1, 1, 0, 0],
        "span_onsets": [True, True, False, False], "valid_tokens": [True, True, True, False],
    }})
    result = evaluate(tmp_path)
    assert result["methods"]["support_graph"]["all_error"]["tokens"] == 3
    assert result["methods"]["support_graph"]["span_onset_vs_normal"]["positives"] == 2
    assert result["methods"]["support_graph"]["all_error"]["auroc"] == 1.0
