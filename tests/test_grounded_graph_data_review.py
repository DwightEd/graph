import hashlib
import json

from next_iteration.constraint_inventory import compile_inventory
from next_iteration.grounded_graph_data import PROTOCOL, choose_sources, compile_templates, source_split, target_candidates
from next_iteration.grounded_graph_synthesize import decode_result


TASKS = tuple(PROTOCOL["tasks"])


def _row(*, row_id, source_id, task, source, official_split="train"):
    prompt = f"Task: {task}\nSOURCE:\n{source}\nEND"
    return {
        "id": str(row_id),
        "source_id": str(source_id),
        "task": task,
        "official_split": official_split,
        "prompt": prompt,
        "source_span": [prompt.index(source), prompt.index(source) + len(source)],
        "response": "unused natural response",
        "response_sha256": hashlib.sha256(b"unused natural response").hexdigest(),
        "prompt_length": 3,
        "token_ids": [1, 2, 3, 4],
    }


def _source_for_split(task, desired_split, index):
    probe = 0
    while True:
        source = repr({"task": task, "name": f"biz-{desired_split}-{index}-{probe}", "rating": float(index + 1)})
        sha = hashlib.sha256(source.encode()).hexdigest()
        if source_split(sha) == desired_split:
            return source, sha
        probe += 1


def test_choose_sources_uses_official_train_and_raw_source_sha_dedup(monkeypatch):
    monkeypatch.setitem(PROTOCOL, "train_sources_per_task", 1)
    monkeypatch.setitem(PROTOCOL, "validation_sources_per_task", 1)
    rows = []
    sid = 100
    for task in TASKS:
        for desired in ("train", "validation"):
            source, _ = _source_for_split(task, desired, sid)
            rows.append(_row(row_id=sid, source_id=sid, task=task, source=source))
            rows.append(_row(row_id=sid + 1000, source_id=sid + 1000, task=task, source=source))
            rows.append(_row(row_id=sid + 2000, source_id=sid + 2000, task=task, source=source, official_split="test"))
            sid += 1

    selected, census = choose_sources(rows)

    assert len(selected) == len(TASKS) * 2
    assert len({s["text_sha256"] for s in selected}) == len(selected)
    assert {s["split"] for s in selected} == {"train", "validation"}
    assert all(s["row"]["official_split"] == "train" for s in selected)
    assert all(census[f"{task}:train"]["selected"] == 1 for task in TASKS)
    assert all(census[f"{task}:validation"]["selected"] == 1 for task in TASKS)


def test_target_candidates_keep_unknowns_out_of_payload_targets_and_add_explicit_weekday_bundle():
    source = repr({
        "name": "Cafe",
        "attributes": {"WiFi": None},
        "hours": {
            "Monday": "8:0-16:0",
            "Tuesday": "8:0-16:0",
            "Wednesday": "8:0-16:0",
            "Thursday": "8:0-16:0",
            "Friday": "8:0-16:0",
            "Saturday": "8:0-16:0",
            "Sunday": "8:0-16:0",
        },
    })
    inventory = compile_inventory(source, source_id="42", task="Data2txt")

    targets, census = target_candidates(inventory)

    assert len(targets) <= PROTOCOL["targets_per_source"]
    assert census["unknown_fields_retained_in_graph_not_used_as_payload_targets"] >= 1
    assert all(t["value"] != "None" for t in targets)
    bundle = next(t for t in targets if t["kind"] == "identical_weekday_bundle")
    assert bundle["owner_id"] not in bundle["source_member_ids"]
    assert len(bundle["source_member_ids"]) == 7
    assert len(bundle["source_spans"]) == 7
    assert bundle["field_path"] == ["hours", "all_seven_identical_days"]


def test_compile_templates_duplicate_ids_do_not_erase_other_valid_templates():
    request = {
        "source_id": "s1",
        "split": "train",
        "task": "Data2txt",
        "targets": [
            {"id": "t0", "value": "4.0", "owner_id": "owner0", "source_member_ids": ["owner0"], "source_spans": [[10, 13]]},
            {"id": "t1", "value": "free", "owner_id": "owner1", "source_member_ids": ["owner1"], "source_spans": [[20, 24]]},
        ],
    }
    raw = json.dumps({"items": [
        {"id": "t0", "template": "The rating is <VALUE>."},
        {"id": "t0", "template": "Its score is <VALUE>."},
        {"id": "t1", "template": "WiFi is <VALUE>."},
    ]})

    compiled = compile_templates(request, raw, "complete")

    assert [example["target_id"] for example in compiled["examples"]] == ["t1"]
    assert compiled["semantic_ground_truth"] is False
    assert compiled["labels_read"] is False
    duplicate_failures = [f for f in compiled["failures"] if f["reason"] == "duplicate target ID"]
    assert len(duplicate_failures) == 2
    assert compiled["mechanically_usable"] == 1


def test_compile_templates_malformed_json_keeps_selected_target_denominator():
    request = {
        "source_id": "s1",
        "split": "validation",
        "task": "QA",
        "targets": [
            {"id": "t0", "value": "Paris", "owner_id": "owner0", "source_member_ids": ["owner0"], "source_spans": [[0, 5]]},
            {"id": "t1", "value": "France", "owner_id": "owner1", "source_member_ids": ["owner1"], "source_spans": [[10, 16]]},
        ],
    }

    compiled = compile_templates(request, "[1, 2, 3]", "complete")

    assert compiled["selected_target_count"] == 2
    assert compiled["mechanically_usable"] == 0
    assert any(f["kind"] == "parse" for f in compiled["failures"])
    assert {f["target_id"] for f in compiled["failures"] if f["kind"] == "missing_target"} == {"t0", "t1"}


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        kept = [i for i in ids if not (skip_special_tokens and i == 99)]
        return " ".join(str(i) for i in kept)


def test_decode_result_handles_batch_padding_eos_and_thinking_close():
    tokenizer = ToyTokenizer()

    decoded = decode_result(tokenizer, [7, 42, 10, 11, 99, 99], [99], 42)
    assert decoded["generation_status"] == "complete"
    assert decoded["generated_ids"] == [7, 42, 10, 11, 99]
    assert decoded["final_output"] == "10 11"

    missing_close = decode_result(tokenizer, [7, 10, 99, 99], [99], 42)
    assert missing_close["generation_status"] == "missing_thinking_close"
    assert missing_close["final_output"] == ""


def test_inventory_parent_rejects_stale_or_mismatched_inventory_settings(tmp_path):
    from next_iteration.constraint_inventory import PROTOCOL as INVENTORY_PROTOCOL
    from next_iteration.grounded_graph_data import inventory_parent
    from route_graph.audit_artifacts import file_sha256

    inventory = tmp_path / "inventory"
    inventory.mkdir()
    settings = {"input_sha256": "expected", "protocol": INVENTORY_PROTOCOL}
    (inventory / "settings.json").write_text(json.dumps(settings) + "\n")
    (inventory / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "settings_file_sha256": file_sha256(inventory / "settings.json"),
        "artifacts": {},
    }) + "\n")

    parent, loaded = inventory_parent(inventory, "expected")
    assert parent["status"] == "complete"
    assert loaded == settings

    try:
        inventory_parent(inventory, "other-input")
    except ValueError as error:
        assert "protocol/input-bound" in str(error)
    else:
        raise AssertionError("inventory input SHA mismatch should be rejected")

    stale = {**settings, "protocol": {**INVENTORY_PROTOCOL, "labels_read": True}}
    (inventory / "settings.json").write_text(json.dumps(stale) + "\n")
    (inventory / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "settings_file_sha256": file_sha256(inventory / "settings.json"),
        "artifacts": {},
    }) + "\n")
    try:
        inventory_parent(inventory, "expected")
    except ValueError as error:
        assert "protocol/input-bound" in str(error)
    else:
        raise AssertionError("inventory protocol labels_read mismatch should be rejected")
