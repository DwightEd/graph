"""Freeze full-roster typed-hour predictions and exact native inputs on CPU."""

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from next_iteration.typed_hours import PROTOCOL, native_contrast, prepare_row
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import rows as validated_rows
from route_graph.frozen_reader import digest, write_json_once

GRAPH = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.output.exists():
        raise FileExistsError("typed preflight requires a fresh output directory")
    input_sha = file_sha256(args.inputs)
    roster = validated_rows(args)
    if not roster:
        raise ValueError("empty response roster")
    paths = sorted((GRAPH / "route_graph").glob("*.py")) + [GRAPH / "next_iteration" / n for n in
        ("__init__.py", "graph_boundaries.py", "surface_graph.py", "typed_hours.py", "typed_hours_prepare.py")]
    code = {str(p.relative_to(GRAPH)): file_sha256(p) for p in paths}
    tokenizer = AutoTokenizer.from_pretrained(args.observer_model, local_files_only=True)
    tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json",
        "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model"}
    tokenizer_files = {p.name: file_sha256(p) for p in args.observer_model.iterdir() if p.is_file() and p.name in tokenizer_names}
    settings = {"schema": "typed-hours-full-prepare@1", "protocol": PROTOCOL,
        "input_path": str(args.inputs.resolve()), "input_sha256": input_sha,
        "roster": [{k: r[k] for k in ("id", "source_id", "task", "generator", "official_split")} for r in roster],
        "code_sha256": code, "observer_model": str(args.observer_model.resolve()),
        "tokenizer_files": tokenizer_files, "native_forward_calls": 0, "reader_calls": 0, "labels_used": False,
        "prediction_scope": "all rows retained; typed Data2txt hours only; no semantic-provider fallback",
        "test_usage": "freeze predictions only; no test gold or accuracy used for design"}
    args.output.mkdir(parents=True)
    for name in code:
        target = args.output / "executed_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GRAPH / name, target)
    write_json_once(args.output / "settings.json", settings)
    settings_digest = digest(settings)
    total, by_split, artifacts, sources, eligible = Counter(), {}, {}, set(), []
    start = time.monotonic()
    for index, row in enumerate(roster):
        p = prepare_row(row)
        counts = Counter({"responses": 1, "words": len(re.findall(r"\S+", row["response"])), "response:" + p["status"]: 1})
        schedule_ref = None
        if "schedule" in p:
            schedule = p["schedule"]
            schedule_ref = "sources/" + schedule["sha256"] + ".json"
            if schedule_ref not in sources:
                write_json_once(args.output / schedule_ref, schedule)
                artifacts[schedule_ref] = file_sha256(args.output / schedule_ref)
                sources.add(schedule_ref)
        bridges = []
        for fact_index, fact in enumerate(p["facts"]):
            counts["bases"] += 1
            counts["fact:" + fact["status"]] += 1
            if fact["status"] == "typed_contrast_available":
                try:
                    bridge = native_contrast(row, p, fact_index, tokenizer)
                except (ValueError, TypeError) as error:
                    bridges.append({"fact_index": fact_index, "status": "native_alignment_unresolved", "reason": str(error)})
                    counts["native_alignment_unresolved"] += 1
                else:
                    name = f"contrasts/{row['id']}_{fact_index}.json"
                    write_json_once(args.output / name, bridge)
                    artifacts[name] = file_sha256(args.output / name)
                    record = {"response_id": row["id"], "source_id": row["source_id"], "fact_index": fact_index,
                        "official_split": row["official_split"], "path": name, "sha256": artifacts[name],
                        "bridge_sha256": bridge["sha256"], "source_value_keys": len(bridge["source_keys"])}
                    eligible.append(record)
                    bridges.append(record)
                    counts["native_inputs_compiled"] += 1
        data = {"response_id": row["id"], "source_id": row["source_id"], "task": row["task"],
            "official_split": row["official_split"], "generator": row["generator"],
            "row_sha256": digest(row), "prepared_sha256": p["sha256"], "status": p["status"],
            "reason": p.get("reason"), "source_schedule": schedule_ref, "facts": p["facts"],
            "native_inputs": bridges, "counts": dict(counts), "labels_used": False,
            "native_forward_calls": 0, "settings_object_digest": settings_digest}
        name = f"rows/{row['id']}.json"
        write_json_once(args.output / name, data)
        artifacts[name] = file_sha256(args.output / name)
        total.update(counts)
        by_split.setdefault(row["official_split"], Counter()).update(counts)
        if (index + 1) % 500 == 0 or index + 1 == len(roster):
            print(json.dumps({"done": index + 1, "total": len(roster), "native_inputs": len(eligible),
                "seconds": time.monotonic() - start}), flush=True)
    if file_sha256(args.inputs) != input_sha or any(file_sha256(GRAPH / n) != sha
            or file_sha256(args.output / "executed_code" / n) != sha for n, sha in code.items()):
        raise ValueError("frozen input or code changed during typed prepare")
    if any(file_sha256(args.observer_model / n) != sha for n, sha in tokenizer_files.items()):
        raise ValueError("observer tokenizer files changed during prepare")
    summary = {"status": "complete", "counts": dict(total), "by_split": {k: dict(v) for k, v in by_split.items()},
        "unique_source_schedules": len(sources), "native_inputs": eligible,
        "labels_read": False, "reader_calls": 0, "model_forwards": 0,
        "settings_object_digest": settings_digest, "seconds": time.monotonic() - start,
        "scope": "typed compile/predictions only, not accuracy or native mechanism results"}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"settings_file_sha256": file_sha256(args.output / "settings.json"),
        "artifacts": artifacts, "status": "complete", "labels_read": False, "model_forwards": 0})
    print(json.dumps({k: v for k, v in summary.items() if k != "native_inputs"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-model", type=Path, default=GRAPH.parent.parent / "models/Meta-Llama-3.1-8B-Instruct")
    prepare(parser.parse_args())
