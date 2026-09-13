"""Source-coordinate reconstruction without generated semantic pseudo-labels.

Copying provenance is observed. It is not natural-answer ownership truth, and
improvement on this objective alone does not establish hallucination detection.
"""

import argparse
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

from next_iteration.grounded_graph_data import (
    ROOT,
    choose_sources,
    inventory_parent,
    source_split,
)
from next_iteration.surface_graph import _check
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import rows
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "grounded-source-reconstruction@1", "seed": 20260913,
    "source_selection": "same frozen 240 official-train sources; raw source SHA split192/48",
    "response": "verbatim complete source text; original task prompt retained",
    "pointer_provenance": "raw source occurrence coordinate; not semantic owner truth",
    "pointer_cap_per_source": 64, "minimum_global_prefix_words": 8,
    "natural_context_prefix_words": 2, "natural_context_stride_words": 8,
    "labels_read": False, "generated_templates": False, "semantic_ground_truth": False,
    "model_forwards": 0, "source_copy_distribution_shift_unresolved": True}


def reconstruction(inventory):
    _check(inventory)
    text = inventory["text"]
    global_words = list(re.finditer(r"\w+(?:['’\-]\w+)*", text))
    cutoff = global_words[PROTOCOL["minimum_global_prefix_words"]].start() if len(global_words) > PROTOCOL["minimum_global_prefix_words"] else len(text)
    units = inventory["fields"] if inventory["task"] == "Data2txt" else inventory["contexts"]
    targets = []
    for unit in units:
        if unit.get("value_status") == "unknown":
            continue
        a, b = unit["raw_span"]
        words = list(re.finditer(r"\w+(?:['’\-]\w+)*", text[a:b]))
        if inventory["task"] == "Data2txt":
            words = words[:1]
        else:
            words = words[PROTOCOL["natural_context_prefix_words"]::PROTOCOL["natural_context_stride_words"]]
        for word in words:
            span = [a + word.start(), a + word.end()]
            if span[0] >= cutoff:
                targets.append({"target_span": span, "source_owner_id": unit["id"],
                    "source_span": span, "raw_value": text[slice(*span)],
                    "supervision": "observed_copy_coordinate; semantic_owner_unverified"})
    # Overlapping lexical context proposals cannot silently invent multiple
    # equivalent semantic owners. Exact duplicate target coordinates are omitted.
    counts = {}
    for target in targets:
        key = tuple(target["target_span"])
        counts[key] = counts.get(key, 0) + 1
    unique = [x for x in targets if counts[tuple(x["target_span"])] == 1]
    unique.sort(key=lambda x: digest([PROTOCOL["seed"], inventory["source_id"], x["source_owner_id"], x["target_span"]]))
    selected = sorted(unique[:PROTOCOL["pointer_cap_per_source"]], key=lambda x: x["target_span"])
    return {"text": text, "targets": selected, "eligible_pointers": len(unique),
        "ambiguous_coordinate_proposals": len(targets) - len(unique), "semantic_ground_truth": False}


def read(path):
    import json
    return json.loads(Path(path).read_text())


def verify(path):
    settings, manifest = read(path / "settings.json"), read(path / "manifest.json")
    if (settings["protocol"] != PROTOCOL or manifest["status"] != "complete"
            or file_sha256(path / "settings.json") != manifest["settings_sha256"]
            or file_sha256(settings["input_path"]) != settings["input_sha256"]):
        raise ValueError("source reconstruction input/settings/protocol changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(path / "executed_code" / name) != sha:
            raise ValueError("source reconstruction live/snapshot code changed")
    expected = {"examples.json", "summary.json"}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("source reconstruction artifact census differs")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("source reconstruction artifact changed")
    inventory = Path(settings["inventory_path"])
    parent, inv_settings = inventory_parent(inventory, settings["input_sha256"])
    if (file_sha256(inventory / "manifest.json") != settings["inventory_manifest_sha256"]
            or parent["settings_file_sha256"] != settings["inventory_settings_sha256"]
            or digest(inv_settings) != settings["inventory_settings_digest"]):
        raise ValueError("source reconstruction inventory settings changed")
    for name, sha in settings["source_artifacts"].items():
        if file_sha256(inventory / name) != sha:
            raise ValueError("source reconstruction inventory artifact changed")
    examples = read(path / "examples.json")
    expected_ids = {Path(name).stem for name in settings["source_artifacts"]}
    if len({e["source_id"] for e in examples}) != len(examples) or {e["source_id"] for e in examples} != expected_ids:
        raise ValueError("source reconstruction omits or duplicates selected source IDs")
    census = Counter(e["task"] + ":" + e["split"] for e in examples)
    if dict(census) != {k: v["selected"] for k, v in settings["selection_census"].items()}:
        raise ValueError("source reconstruction task/split census differs")
    summary = read(path / "summary.json")
    if (summary["sources"] != len(examples) or summary["census"] != settings["selection_census"]
            or summary["pointers"] != sum(len(e["targets"]) for e in examples)
            or summary["sources_without_pointer"] != sum(not e["targets"] for e in examples)):
        raise ValueError("source reconstruction summary differs from complete roster")
    for example in examples:
        wrapped = read(inventory / "sources" / f"{example['source_id']}.json")
        if wrapped["settings_object_digest"] != digest(inv_settings):
            raise ValueError("copied source wrapper changed")
        rebuilt = reconstruction(wrapped["data"])
        text_sha = hashlib.sha256(wrapped["data"]["text"].encode()).hexdigest()
        if (example["source_text_sha256"] != text_sha or example["split"] != source_split(text_sha)
                or example["task"] != wrapped["data"]["task"]):
            raise ValueError("source reconstruction actual text SHA split/task differs")
        if any(example[k] != v for k, v in rebuilt.items()):
            raise ValueError("reconstruction source coordinate receipt differs")
    return settings, examples


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh reconstruction output required")
    code = {}
    paths = list((ROOT / "route_graph").glob("*.py")) + [ROOT / "next_iteration" / n for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "grounded_graph_data.py", "grounded_graph_reconstruction.py")]
    for path in paths:
        name = str(path.relative_to(ROOT))
        code[name] = file_sha256(path)
        destination = args.output / "executed_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    input_sha = file_sha256(args.inputs)
    parent, inv_settings = inventory_parent(args.inventory, input_sha)
    selected, census = choose_sources(rows(args))
    settings = {"protocol": PROTOCOL, "input_path": str(args.inputs.resolve()), "input_sha256": input_sha,
        "inventory_path": str(args.inventory.resolve()), "inventory_manifest_sha256": file_sha256(args.inventory / "manifest.json"),
        "inventory_settings_sha256": parent["settings_file_sha256"], "inventory_settings_digest": digest(inv_settings),
        "code_sha256": code, "source_artifacts": {}, "selection_census": census, "labels_read": False}
    examples = []
    for source in selected:
        name = f"sources/{source['source_id']}.json"
        wrapped = read(args.inventory / name)
        sha = file_sha256(args.inventory / name)
        inventory = wrapped["data"]
        if (sha != parent["artifacts"][name] or wrapped["settings_object_digest"] != digest(inv_settings)
                or inventory["source_id"] != source["source_id"] or inventory["task"] != source["task"]
                or inventory["text"] != source["text"]):
            raise ValueError("source reconstruction task/text/coordinate binding differs")
        settings["source_artifacts"][name] = sha
        row = source["row"]
        prompt = {"source_id": source["source_id"], "task": source["task"], "split": source["split"],
            "original_row_id": str(row["id"]), "prompt": row["prompt"], "prompt_length": row["prompt_length"],
            "prompt_token_ids": row["token_ids"][:row["prompt_length"]], "source_span": row["source_span"]}
        examples.append({"id": source["source_id"], "source_id": source["source_id"], "task": source["task"],
            "split": source["split"], "source_text_sha256": hashlib.sha256(source["text"].encode()).hexdigest(),
            "prompt_record": prompt, **reconstruction(inventory)})
    write_json_once(args.output / "settings.json", settings)
    write_json_once(args.output / "examples.json", examples)
    summary = {"sources": len(examples), "census": census, "pointers": sum(len(e["targets"]) for e in examples),
        "sources_without_pointer": sum(not e["targets"] for e in examples), "labels_read": False, "model_forwards": 0,
        "semantic_ground_truth": False, "status": "complete"}
    write_json_once(args.output / "summary.json", summary)
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_sha256": file_sha256(args.output / "settings.json"),
        "artifacts": {n: file_sha256(args.output / n) for n in ("examples.json", "summary.json")}})
    verify(args.output)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
