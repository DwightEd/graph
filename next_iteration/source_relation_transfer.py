"""Parser-light natural candidate transfer; no natural-owner truth labels.

This explicitly exposes the structured-source-to-natural-query domain shift.
All original response slots stay in the denominator, including out-of-domain
tasks, untyped slots, empty source pools and unavailable encodings.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from next_iteration.source_relation_data import compile_source
from next_iteration.source_relation_features import (
    GRAPH,
    freeze_code,
    load_features,
    prepare_inventory,
    read_json,
    verify_code,
)
from next_iteration.source_relation_features import (
    PROTOCOL as FEATURE_PROTOCOL,
)
from next_iteration.source_relation_model import (
    SourceRelMini,
    owner_beam,
    source_role_mass,
)
from next_iteration.source_relation_train import TYPES
from next_iteration.surface_graph import surface_graph
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import rows
from route_graph.frozen_reader import digest, write_json_once

PROTOCOL = {"schema": "source-rel-mini-natural-transfer@1", "topk": 5,
    "domain": "Data2txt structured source scalar occurrences <=40words",
    "query": "all prior response context plus whole current parent with exact target span masked",
    "mask": "coordinate-only; prior same-string occurrences are not assumed equivalent",
    "surface_to_broad_type": {"time": "time_range", "time_range": "time_range", "date": "date",
        "number": "number", "number_range": "number", "duration": "number", "content_run": "text", "explicit_quote": "text"},
    "type_status": "lexical proposal; not verified semantic role",
    "result_scope": "owner candidate supply only; no SCNI/no natural-owner accuracy/no native certificate",
    "labels_read": False, "training_on_responses": False}


def natural_view(response, slot):
    parent = next(p for p in response["base_units"] if p["base_id"] == slot["base_id"])
    a, b = parent["span"]
    x, y = slot["span"]
    text = response["text"]
    if not a <= x < y <= b or text[x:y] != slot["quote"]:
        raise ValueError("natural slot is not an exact member of its original parent")
    masked_parent = text[a:x] + "<VALUE>" + text[y:b]
    return {"text": "Earlier response context:\n" + text[:a] + "\nCurrent response context:\n" + masked_parent
        + "\nFind the source occurrence for the withheld <VALUE>. Field binding:",
        "target_span": [x, y], "parent_span": [a, b], "target_surface_input": False,
        "mask_dependency_scope": "one exact proposal span; no unverified alias closure",
        "previous_context_span": [0, a], "context_is_original_generation_trace": False}


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh natural transfer inventory required")
    input_sha = file_sha256(args.inputs)
    roster = rows(argparse.Namespace(inputs=args.inputs))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    documents, bindings, counts, cache = {}, [], Counter(), {}

    def document(text):
        did = digest(text)
        if did not in documents:
            ids = tokenizer.encode(text, add_special_tokens=True)
            documents[did] = {"id": did, "text": text, "input_ids": ids, "token_count": len(ids),
                "status": "encodable" if 0 < len(ids) <= FEATURE_PROTOCOL["max_tokens"] else "unavailable_length"}
        return did

    for row in roster:
        response = surface_graph(row["response"], sample_id=row["id"], side="response")
        binding = {"source_id": str(row["source_id"]), "response_id": str(row["id"]),
            "task": row["task"], "official_split": row["official_split"], "generator": row["generator"],
            "response_sha256": row["response_sha256"], "response_graph": response,
            "queries": [], "candidates": [], "unresolved_slots": [], "result_scope": PROTOCOL["result_scope"]}
        counts["responses"] += 1
        counts["slots"] += len(response["slots"])
        source = None
        if row["task"] == "Data2txt":
            text = row["prompt"][slice(*row["source_span"])]
            key = (str(row["source_id"]), digest(text))
            if key not in cache:
                cache[key] = compile_source(text, str(row["source_id"]))
            source = cache[key]
        if source is None or source["status"] != "compiled":
            reason = "source_domain_unavailable" if source is None else "source_graph_unresolved"
            binding["unresolved_slots"] = [{"slot_id": s["slot_id"], "span": s["span"], "status": reason} for s in response["slots"]]
            counts[reason] += len(response["slots"])
            bindings.append(binding)
            continue
        binding["source_graph"] = source["source_graph"]
        binding["source_record_root_id"] = source["record_root_id"]
        binding["source_hash_partition"] = source["split"]
        binding["source_reconstruction_training_eligible"] = row["official_split"] == "train"
        binding["source_reconstruction_training_split"] = (source["split"] if row["official_split"] == "train"
            else "not_in_source_reconstruction_training")
        for candidate in source["candidates"]:
            if candidate["epistemic_status"] == "observed_literal" and candidate["value_type"] != "null":
                binding["candidates"].append({**candidate, "document_id": document(candidate["text"])})
        for slot in response["slots"]:
            broad = PROTOCOL["surface_to_broad_type"].get(slot["surface_type"])
            pool = [c["id"] for c in binding["candidates"] if c["value_type"] == broad]
            if not pool:
                binding["unresolved_slots"].append({"slot_id": slot["slot_id"], "span": slot["span"], "status": "no_broad_type_source_pool"})
                counts["no_broad_type_source_pool"] += 1
                continue
            view = natural_view(response, slot)
            binding["queries"].append({"id": slot["slot_id"], **view, "document_id": document(view["text"]),
                "value_type": broad, "candidate_ids": pool, "family": None,
                "semantic_role_status": "unverified", "status": "candidate_query", "positive_ids": None})
            counts["candidate_queries"] += 1
        bindings.append(binding)
    if input_sha != file_sha256(args.inputs):
        raise ValueError("natural roster changed during prepare")
    settings = prepare_inventory(args.output, args.model, sorted(documents.values(), key=lambda d: d["id"]), bindings,
        {"kind": "natural_transfer", "input_path": str(args.inputs.resolve()), "input_sha256": input_sha,
            "protocol": PROTOCOL, "counts": dict(counts), "supervision": "none"},
        extra_code=[Path(__file__).with_name(n) for n in ("source_relation_data.py", "source_relation_model.py",
            "source_relation_train.py", "source_relation_transfer.py")])
    print(json.dumps({"counts": dict(counts), "lengths": settings["length_preflight"]}), flush=True)


def load_head(path):
    manifest, settings = read_json(path / "manifest.json"), read_json(path / "settings.json")
    if (manifest["status"] != "complete" or file_sha256(path / "settings.json") != manifest["settings_file_sha256"]
            or settings["sanity_only"] or settings["phase"] != "train"):
        raise ValueError("requires complete full source-only training checkpoint")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(path / name) != sha:
            raise ValueError("training artifact changed")
    verify_code(path, settings)
    feature_parent = Path(settings["features_path"])
    if file_sha256(feature_parent / "manifest.json") != settings["features_manifest_sha256"]:
        raise ValueError("training feature parent changed")
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["settings_object_digest"] != digest(settings) or checkpoint["types"] != TYPES:
        raise ValueError("checkpoint/settings binding mismatch")
    model = SourceRelMini(input_dim=checkpoint["input_dim"], hidden_dim=checkpoint["hidden_dim"],
        num_types=len(TYPES), temperature=checkpoint["temperature"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), settings


def predict(args):
    if args.output.exists():
        raise FileExistsError("fresh natural candidate prediction output required")
    parent, docs, bindings, matrix = load_features(args.features)
    if parent["parent"]["kind"] != "natural_transfer" or parent["parent"]["protocol"] != PROTOCOL:
        raise ValueError("not this frozen natural transfer inventory")
    if file_sha256(parent["parent"]["input_path"]) != parent["parent"]["input_sha256"]:
        raise ValueError("original natural roster changed")
    model, train_settings = load_head(args.training)
    if read_json(Path(train_settings["features_path"]) / "settings.json")["model_files"] != parent["model_files"]:
        raise ValueError("source and natural vectors use different encoder weights/tokenizer")
    settings = {"protocol": PROTOCOL, "feature_manifest_sha256": file_sha256(args.features / "manifest.json"),
        "feature_path": str(args.features.resolve()), "training_path": str(args.training.resolve()),
        "training_manifest_sha256": file_sha256(args.training / "manifest.json"),
        "code_sha256": freeze_code(args.output, [GRAPH / name for name in train_settings["code_sha256"]]
            + [Path(__file__)])}
    write_json_once(args.output / "settings.json", settings)
    index = {d["id"]: i for i, d in enumerate(docs)}
    available = {d["id"] for d in docs if d["status"] == "encodable"}
    counts, artifacts = Counter(), {}
    with torch.inference_mode():
        for binding in bindings:
            results = list(binding["unresolved_slots"])
            candidates = {c["id"]: c for c in binding["candidates"]}
            for query in binding["queries"]:
                pool = [candidates[cid] for cid in query["candidate_ids"]]
                common = {"slot_id": query["id"], "span": query["target_span"], "candidate_count": len(pool),
                    "factual_status": "unverified", "native_certificate_count": 0, "source_reuse_allowed": True}
                if query["document_id"] not in available or any(c["document_id"] not in available for c in pool):
                    results.append({**common, "status": "whole_query_pool_unavailable_length"})
                    continue
                q = torch.from_numpy(np.array(matrix[[index[query["document_id"]]]], copy=True))
                c = torch.from_numpy(np.array(matrix[[index[p["document_id"]] for p in pool]], copy=True))
                learned = model(q, c, torch.tensor([TYPES.index(query["value_type"])], dtype=torch.long))
                frozen = F.normalize(q, dim=-1) @ F.normalize(c, dim=-1).T / model.temperature
                beams = {}
                for name, scores in (("source_rel_mini", learned), ("same_view_frozen_cosine", frozen)):
                    beam = owner_beam(scores, topk=PROTOCOL["topk"])[0]
                    families, mass = source_role_mass(scores, [p["family"] for p in pool])
                    beams[name] = {**beam, "candidate_ids": [pool[j]["id"] for j in beam["candidate_indices"]],
                        "field_paths": [pool[j]["field_path"] for j in beam["candidate_indices"]],
                        "family_mass": dict(zip(families, mass[0].tolist(), strict=True))}
                results.append({**common, "status": "candidate_supply_only", "rankings": beams,
                    "query_document_id": query["document_id"], "broad_type": query["value_type"]})
                counts["trained_frozen_top1_changed"] += beams["source_rel_mini"]["candidate_ids"][0] != beams["same_view_frozen_cosine"]["candidate_ids"][0]
            if {r["slot_id"] for r in results} != {s["slot_id"] for s in binding["response_graph"]["slots"]} or len(results) != len(binding["response_graph"]["slots"]):
                raise ValueError("natural slot denominator lost or duplicated")
            counts["responses"] += 1
            counts["slots"] += len(results)
            counts.update(r["status"] for r in results)
            name = f"responses/{binding['response_id']}.json"
            write_json_once(args.output / name, {"response_id": binding["response_id"], "source_id": binding["source_id"],
                "task": binding["task"], "official_split": binding["official_split"], "results": results,
                "actual_training_source_membership": train_settings["internal_split"].get(binding["source_id"], "not_in_source_reconstruction_training"),
                "settings_object_digest": digest(settings), "query_supervision": None})
            artifacts[name] = file_sha256(args.output / name)
    if file_sha256(args.features / "manifest.json") != settings["feature_manifest_sha256"] or file_sha256(args.training / "manifest.json") != settings["training_manifest_sha256"]:
        raise ValueError("upstream manifest changed during transfer")
    verify_code(args.output, settings)
    write_json_once(args.output / "summary.json", {"status": "complete", "counts": dict(counts),
        "labels_read": False, "natural_owner_ground_truth": None, "complete_detector": False,
        "scope": PROTOCOL["result_scope"], "native_forwards": 0})
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_file_sha256": file_sha256(args.output / "settings.json"), "artifacts": artifacts})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "predict"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.inputs is None or args.model is None:
            parser.error("prepare requires --inputs and --model")
        prepare(args)
    else:
        if args.features is None or args.training is None:
            parser.error("predict requires --features and --training")
        predict(args)
