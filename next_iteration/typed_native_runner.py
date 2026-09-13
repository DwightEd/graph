"""Frozen typed source-path measurements on a predetermined official-train roster."""

import argparse
import fcntl
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from next_iteration.surface_graph import _check
from next_iteration.typed_hours import native_contrast, prepare_row
from next_iteration.typed_native_measure import (
    PROTOCOL,
    match_controls,
    measure_path,
    root_group,
    search_queries_layers,
    source_control_census,
)
from route_graph.audit_alignment import align_row
from route_graph.audit_artifacts import donor_writer, file_sha256
from route_graph.audit_phase_native import event_contrast
from route_graph.audit_runner import lock_stream, model_manifest, rows
from route_graph.causal_groups import CausalOracle
from route_graph.frozen_reader import digest, write_json_once

GRAPH = Path(__file__).resolve().parents[1]


def parent_data(path):
    manifest = json.loads((path / "manifest.json").read_text())
    settings = json.loads((path / "settings.json").read_text())
    if manifest["status"] != "complete" or manifest["labels_read"] or settings["labels_used"]:
        raise ValueError("typed parent must be complete label-free predictions")
    if file_sha256(path / "settings.json") != manifest["settings_file_sha256"]:
        raise ValueError("typed parent settings changed")
    if file_sha256(path / "summary.json") != manifest["artifacts"]["summary.json"]:
        raise ValueError("typed parent summary changed")
    summary = json.loads((path / "summary.json").read_text())
    if summary["settings_object_digest"] != digest(settings) or summary["status"] != "complete":
        raise ValueError("typed parent summary is not bound to settings")
    if file_sha256(settings["input_path"]) != settings["input_sha256"]:
        raise ValueError("typed input changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(GRAPH / name) != sha or file_sha256(path / "executed_code" / name) != sha:
            raise ValueError("typed parent live/snapshot code changed")
    return manifest, settings, summary


def select_roster(summary, cap):
    if cap < 1:
        raise ValueError("positive source cap required")
    eligible = [r for r in summary["native_inputs"] if r["official_split"] == "train"]
    ordered = sorted(eligible, key=lambda r: (digest({"seed": 20260913, "bridge": r["bridge_sha256"]}), r["path"]))
    selected, sources = [], set()
    for row in ordered:
        if row["source_id"] not in sources and len(sources) < cap:
            selected.append(row)
            sources.add(row["source_id"])
    return selected, {"train_eligible_contrasts": len(eligible), "train_eligible_sources": len({r["source_id"] for r in eligible}),
        "selected": len(selected), "cap_sources": cap,
        "selection": "fixed seed bridge hash, first per source; no native effects, generation preference or gold used",
        "all_eligible_train": eligible, "unselected_remain_in_parent": True}


def settings(args):
    parent_manifest, parent_settings, summary = parent_data(args.preflight)
    selected, selection = select_roster(summary, args.max_sources)
    if not selected:
        raise ValueError("no typed train contrasts; no GPU load justified")
    by_id = {str(r["id"]): r for r in rows(argparse.Namespace(inputs=Path(parent_settings["input_path"])))}
    tokenizer = AutoTokenizer.from_pretrained(args.observer_model, local_files_only=True)
    if str(args.observer_model.resolve()) != parent_settings["observer_model"]:
        raise ValueError("native observer differs from typed token preparation")
    bridges, censuses = {}, {}
    for entry in selected:
        path = args.preflight / entry["path"]
        if file_sha256(path) != entry["sha256"] or parent_manifest["artifacts"].get(entry["path"]) != entry["sha256"]:
            raise ValueError("selected native bridge is not a frozen parent artifact")
        bridge = json.loads(path.read_text())
        _check(bridge)
        row = by_id[str(entry["response_id"])]
        prepared = prepare_row(row)
        recomputed = native_contrast(row, prepared, entry["fact_index"], tokenizer)
        # JSON round-trips token tuples as lists; the sealed canonical digest
        # compares every value while ignoring only that storage distinction.
        if digest(recomputed) != digest(bridge):
            raise ValueError("typed semantic proof/native bridge recomputation differs")
        contrast = event_contrast(bridge)
        per_day = bridge["source_keys_per_day"]
        if (len(per_day) != 7 or any(not keys for keys in per_day)
                or bridge["source_keys"] != sorted({k for keys in per_day for k in keys})
                or bridge["source_keys"] != recomputed["source_keys"]
                or any(not 0 <= k < contrast.prompt_length or not row["source_mask"][k] for k in bridge["source_keys"])):
            raise ValueError("native A differs from all seven original source endpoint keys")
        if max(len(contrast.prefix) + len(c) - 1 for c in contrast.continuations) > 4096:
            raise ValueError("typed native input exceeds frozen 4096 limit")
        bridges[entry["path"]] = bridge
        census = source_control_census(prepared, align_row(row, tokenizer), len(contrast.prefix))
        censuses[entry["path"]] = census
    names = sorted((GRAPH / "route_graph").glob("*.py")) + [GRAPH / "next_iteration" / n for n in (
        "__init__.py", "graph_boundaries.py", "surface_graph.py", "typed_hours.py", "typed_hours_prepare.py",
        "typed_native_measure.py", "typed_native_runner.py")]
    names += [GRAPH / "experiments" / n for n in ("interleave_soft_graph.py", "interleave_typed_native.py")]
    code = {str(p.relative_to(GRAPH)): file_sha256(p) for p in names}
    expected = {"protocol": PROTOCOL, "preflight": str(args.preflight), "parent_manifest_sha256": file_sha256(args.preflight / "manifest.json"),
        "parent_summary_sha256": file_sha256(args.preflight / "summary.json"), "input_path": parent_settings["input_path"],
        "input_sha256": parent_settings["input_sha256"], "selected": selected, "selection": selection,
        "frozen_control_censuses": censuses, "observer_model": str(args.observer_model.resolve()),
        "observer_files": model_manifest(args.observer_model), "code_sha256": code, "seed": 20260913,
        "torch": str(torch.__version__), "transformers": transformers.__version__, "dtype": "bfloat16", "attention": "eager",
        "labels_used": False, "reader_calls": 0, "original_generator_causality": False,
        "graph_features": "actual branch-B baseline decoder output per layer and original source-A/query token; stored raw bf16 bytes"}
    target = args.output / "settings.json"
    if target.exists():
        if json.loads(target.read_text()) != expected:
            raise ValueError("native frozen settings/code/model changed; use fresh revision")
        for name, sha in code.items():
            if file_sha256(args.output / "executed_code" / name) != sha:
                raise ValueError("native snapshot differs")
    else:
        for name, sha in code.items():
            destination = args.output / "executed_code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("partial code snapshot exists; preserve and use fresh output")
            shutil.copy2(GRAPH / name, destination)
            if file_sha256(destination) != sha:
                raise ValueError("code changed while freezing")
        write_json_once(target, expected)
    return expected, bridges


class GraphOracle(CausalOracle):
    """Capture high-dimensional nodes in the first already-counted baseline."""

    def __init__(self, model, contrast, keys, **kwargs):
        self.feature_keys = sorted(set(keys) | set(contrast.shared_queries))
        self.features = {}
        super().__init__(model, contrast, **kwargs)

    def _forward(self, ids, *args, **kwargs):
        handles = []
        if self.calls == 0:
            for index, layer in enumerate(self.model.model.layers):
                def capture(module, inputs, output, index=index):
                    hidden = output[0] if isinstance(output, tuple) else output
                    values = hidden[0, self.feature_keys].detach().contiguous().cpu()
                    self.features[index] = {"shape": list(values.shape), "dtype": str(values.dtype),
                        "bytes": values.view(torch.uint8).numpy().copy()}
                handles.append(layer.register_forward_hook(capture))
        try:
            return super()._forward(ids, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()


def progress(output, **record):
    data = {"pid": os.getpid(), "unix_time": time.time(), **record}
    path = output / "progress.partial.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(path, output / "progress.json")
    print(json.dumps(data), flush=True)


def run(args, frozen, bridges):
    if any((args.output / n).exists() for n in ("results", "nodes", "summary.json")):
        raise FileExistsError("native execution requires fresh result artifacts; no silent partial rerun")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive free GPU required; do not overlap population")
    progress(args.output, status="loading_model", total=len(frozen["selected"]))
    torch.manual_seed(frozen["seed"])
    model = AutoModelForCausalLM.from_pretrained(args.observer_model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="eager").eval()
    model.requires_grad_(False)
    total_calls, counts, artifacts, start = 0, Counter(), {}, time.monotonic()
    for index, entry in enumerate(frozen["selected"]):
        key = f"{entry['response_id']}_{entry['fact_index']}"
        bridge = bridges[entry["path"]]
        oracle = GraphOracle(model, event_contrast(bridge), bridge["source_keys"], budget=PROTOCOL["budget_actual_forwards"],
            artifact_writer=donor_writer(args.output / "donors" / key))
        initial_controls = match_controls(oracle, root_group(oracle, bridge["source_keys"]), frozen["frozen_control_censuses"][entry["path"]])
        fixed_ids = {kind: [c["id"] for c in initial_controls[kind]] for kind in ("position", "origin")}
        write_json_once(args.output / "controls_initial" / f"{key}.json", {"settings_object_digest": digest(frozen),
            "controls": initial_controls, "fixed_ids": fixed_ids, "actual_forward_calls_before_freeze": oracle.calls,
            "selection_before_search_or_intervention": True})
        search = search_queries_layers(oracle, bridge["source_keys"])
        if any(m["group"]["keys"] != bridge["source_keys"] for m in search["measured"] + search["pending"]):
            raise ValueError("native search changed the quantified all-day A key set")
        controls = match_controls(oracle, search["selected"]["group"], frozen["frozen_control_censuses"][entry["path"]], fixed_ids=fixed_ids)
        # Recheck the same pre-search IDs in the selected scope, never replace.
        write_json_once(args.output / "controls" / f"{key}.json", {"settings_object_digest": digest(frozen), "controls": controls,
            "search_sha256": digest(search), "selection_before_validation": True})
        path_result = measure_path(oracle, search, controls, bridge["slot_masks"])
        feature_path = args.output / "nodes" / f"{key}.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        with feature_path.open("xb") as stream:
            np.savez_compressed(stream, **{str(layer): info["bytes"] for layer, info in oracle.features.items()})
        features = {"file": str(feature_path.relative_to(args.output)), "sha256": file_sha256(feature_path),
            "token_keys": oracle.feature_keys, "layers": {str(layer): {k: v for k, v in info.items() if k != "bytes"} for layer, info in oracle.features.items()},
            "source_A_keys": bridge["source_keys"], "query_keys": list(oracle.contrast.shared_queries),
            "captured_forward": "branch B original baseline; one of two accounted baseline forwards",
            "topology": "token coordinates join typed source incidence to measured layer/query hyperedges; no per-edge causal score inferred from group effect"}
        result = {"settings_object_digest": digest(frozen), "bridge_sha256": bridge["sha256"], "entry": entry,
            "base": oracle.base, "search": search, "controls": controls, "path": path_result, "nodes": features,
            "records": oracle.records, "donor_artifacts": oracle.donor_artifacts, "actual_forward_calls": oracle.calls,
            "tokens_processed": oracle.tokens_processed, "labels_read": False, "semantic_provider": bridge["source_semantics"]}
        destination = args.output / "results" / f"{key}.json"
        write_json_once(destination, result)
        artifacts[str(destination.relative_to(args.output))] = file_sha256(destination)
        artifacts[features["file"]] = features["sha256"]
        controls_name = f"controls/{key}.json"
        artifacts[controls_name] = file_sha256(args.output / controls_name)
        controls_initial_name = f"controls_initial/{key}.json"
        artifacts[controls_initial_name] = file_sha256(args.output / controls_initial_name)
        for ref in oracle.donor_artifacts:
            for kind, sha_name in (("manifest", "manifest_sha256"), ("values", "npz_sha256")):
                artifacts[str(Path(ref[kind]).relative_to(args.output))] = ref[sha_name]
        total_calls += oracle.calls
        counts["contrasts"] += 1
        counts[path_result["status"]] += 1
        counts[path_result["full_continuation_preference"]] += 1
        counts["A_directed_position_raw"] += path_result["raw_position_delta"] <= -.5
        counts["A_directed_origin_raw"] += path_result["raw_origin_delta"] <= -.5
        progress(args.output, status="running", done=index + 1, total=len(frozen["selected"]), actual_forward_calls=total_calls,
            current=key, base_F=oracle.base_f, position_delta=path_result["raw_position_delta"], origin_delta=path_result["raw_origin_delta"])
        del oracle
    # Recheck frozen proof/model/code before declaring completion.
    settings(args)
    summary = {"status": "complete", "counts": dict(counts), "actual_forward_calls": total_calls,
        "seconds": time.monotonic() - start, "labels_read": False, "settings_object_digest": digest(frozen),
        "scope": PROTOCOL["claim_scope"], "selection": frozen["selection"], "routing": "unresolved", "aggregation": "unresolved"}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    for name, sha in artifacts.items():
        if file_sha256(args.output / name) != sha:
            raise ValueError("native artifact changed before completion")
    write_json_once(args.output / "manifest.json", {"settings_file_sha256": file_sha256(args.output / "settings.json"), "artifacts": artifacts,
        "status": "complete", "labels_read": False, "actual_forward_calls": total_calls})
    progress(args.output, status="complete", done=len(frozen["selected"]), actual_forward_calls=total_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-model", type=Path, default=GRAPH.parent.parent / "models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--stage", choices=["prepare", "all"], default="prepare")
    parser.add_argument("--inherited-lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.preflight, args.output = args.preflight.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with lock_stream(args) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen, bridges = settings(args)
        if args.stage == "all":
            run(args, frozen, bridges)
        else:
            print(json.dumps({"status": "prepared", "selected": len(frozen["selected"]), "settings_object_digest": digest(frozen)}), flush=True)


if __name__ == "__main__":
    main()
