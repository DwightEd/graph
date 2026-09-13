"""Prepare annotation-free raw inventories and literal field graphs on CPU."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from route_graph.frozen_reader import write_json_once
from route_graph.source_event_graph import (
    compile_literal_fields,
    raw_inventory,
    validate_pointer_graph,
)


def prepare(inputs, output):
    if output.exists():
        raise FileExistsError("use a fresh inventory output directory")
    raw = inputs.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    allowed = {
        "generator", "id", "official_split", "offsets", "prompt", "prompt_length",
        "response", "response_sha256", "source_id", "source_mask", "source_span",
        "task", "token_ids",
    }
    if any(set(row) != allowed for row in rows):
        raise ValueError("inventory inputs differ from the annotation-free roster schema")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate response IDs")
    sources = {}
    for row in rows:
        if not str(row["id"]).isdigit() or not str(row["source_id"]).isdigit():
            raise ValueError("IDs must be digits for output paths")
        if hashlib.sha256(row["response"].encode()).hexdigest() != row["response_sha256"]:
            raise ValueError("response hash mismatch")
        a, b = row["source_span"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(row["prompt"]):
            raise ValueError("invalid raw source coordinates")
        value = (row["task"], row["prompt"][a:b])
        sid = str(row["source_id"])
        if sid in sources and sources[sid] != value:
            raise ValueError("same source ID has different task/raw source")
        sources[sid] = value
    output.mkdir(parents=True)
    for row in rows:
        write_json_once(
            output / "response_inventory" / f"{row['id']}.json",
            raw_inventory(row["response"], side="response", sample_id=row["id"]),
        )
    literal_results = []
    for sid, (task, source) in sources.items():
        inventory = raw_inventory(source, side="source", sample_id=sid)
        write_json_once(output / "source_inventory" / f"{sid}.json", inventory)
        if task != "Data2txt":
            continue
        try:
            graph = compile_literal_fields(inventory)
            validate_pointer_graph(inventory, graph)
        except (ValueError, TypeError, SyntaxError) as error:
            graph = {"status": "literal_compile_failed", "reason": str(error),
                     "inventory_sha256": inventory["sha256"], "absence_inference_allowed": False}
        write_json_once(output / "literal_fields" / f"{sid}.json", graph)
        literal_results.append({"source_id": sid, "status": graph.get("schema", graph.get("status")),
                                "nodes": len(graph.get("nodes", []))})
    code = output / "executed_code"
    code.mkdir()
    for path in (Path(__file__), Path(__file__).parents[1] / "route_graph/source_event_graph.py",
                 Path(__file__).parents[1] / "route_graph/frozen_reader.py"):
        shutil.copy2(path, code / path.name)
    manifest = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.glob("*/*")) if p.is_file()
    }
    result = {
        "schema": "source-event-inventory-preparation@1", "status": "complete",
        "input_path": str(inputs.resolve()), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "responses": len(rows), "sources": len(sources), "literal_results": literal_results,
        "files": manifest, "labels_used": False, "model_calls": 0,
        "scope": "raw_inventory_and_literal_structure_only; no_event_alignment_or_detection",
    }
    write_json_once(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.inputs, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
