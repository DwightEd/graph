"""Review tests for SourceRel natural transfer boundaries."""

import argparse
import hashlib
import json

from next_iteration import source_relation_transfer as transfer
from next_iteration.surface_graph import surface_graph


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [ord(ch) % 251 + 1 for ch in text]
        return ([0] + ids) if add_special_tokens else ids


def _model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text("{}\n")
    (path / "tokenizer.json").write_text("{}\n")
    return path


def _row(rid, sid, task, source, response, official_split="train"):
    prompt = "Instruction. Source: " + source
    return {
        "id": str(rid),
        "source_id": str(sid),
        "task": task,
        "generator": "fixture",
        "official_split": official_split,
        "prompt": prompt,
        "source_span": [prompt.index(source), prompt.index(source) + len(source)],
        "response": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "prompt_length": 1,
        "token_ids": [1],
        "source_mask": [False],
        "offsets": [],
    }


def test_natural_view_masks_only_current_target_and_keeps_prior_context():
    text = "Earlier sentence has 4.0 stars. The business has 5.0 stars."
    graph = surface_graph(text, sample_id="1", side="response")
    slot = next(s for s in graph["slots"] if s["quote"] == "5.0")

    view = transfer.natural_view(graph, slot)

    assert "Earlier sentence has 4.0 stars." in view["text"]
    assert "The business has <VALUE> stars." in view["text"]
    assert "5.0" not in view["text"]
    assert view["target_surface_input"] is False
    assert view["target_span"] == slot["span"]


def test_prepare_preserves_denominator_and_has_no_response_supervision(monkeypatch, tmp_path):
    monkeypatch.setattr(transfer.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    source = repr({
        "name": "Cafe",
        "attributes": {"WiFi": "free", "Parking": "paid"},
        "business_stars": 4.0,
    })
    rows = [
        _row(101, 201, "Data2txt", source, "The business offers free WiFi and has 4.0 stars."),
        _row(102, 202, "Summary", "Plain source text", "A different 123 claim."),
        _row(103, 203, "Data2txt", source, "The business has 4.0 stars.", official_split="test"),
    ]
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "transfer_features"

    transfer.prepare(argparse.Namespace(inputs=inputs, model=_model_dir(tmp_path), output=output))

    bindings = json.loads((output / "bindings.json").read_text())
    data2txt = next(b for b in bindings if b["response_id"] == "101")
    outtask = next(b for b in bindings if b["response_id"] == "102")
    data2txt_test = next(b for b in bindings if b["response_id"] == "103")
    assert data2txt["source_record_root_id"] == data2txt["source_graph"]["literal_graph"]["root"]
    assert data2txt["source_hash_partition"] in {"source_train", "source_validation"}
    assert data2txt["source_reconstruction_training_eligible"] is True
    assert data2txt["source_reconstruction_training_split"] == data2txt["source_hash_partition"]
    assert data2txt_test["source_hash_partition"] in {"source_train", "source_validation"}
    assert data2txt_test["source_reconstruction_training_eligible"] is False
    assert data2txt_test["source_reconstruction_training_split"] == "not_in_source_reconstruction_training"
    assert all(q["positive_ids"] is None for q in data2txt["queries"])
    assert all(q["semantic_role_status"] == "unverified" for q in data2txt["queries"])
    assert all(q["target_surface_input"] is False for q in data2txt["queries"])
    assert len(data2txt["queries"]) + len(data2txt["unresolved_slots"]) == len(data2txt["response_graph"]["slots"])
    assert len(outtask["unresolved_slots"]) == len(outtask["response_graph"]["slots"])
    assert {slot["status"] for slot in outtask["unresolved_slots"]} == {"source_domain_unavailable"}
