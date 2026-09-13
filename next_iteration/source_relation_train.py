"""Train/evaluate source-field pointer reconstruction; not hallucination labels."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import functional as F

from next_iteration.source_relation_features import (
    GRAPH,
    freeze_code,
    load_features,
    verify_code,
    verify_prepared,
)
from next_iteration.source_relation_model import SourceRelMini, owner_beam, owner_nce
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once

TYPES = ["boolean", "number", "null", "time_range", "date", "text", "unknown"]
PROTOCOL = {"schema": "source-rel-mini-train@1", "seed": 20260913, "hidden_dim": 128,
    "temperature": .07, "lr": 1e-3, "weight_decay": 1e-4, "batch_sources": 16,
    "checkpoint_selection": "minimum source_validation query-mean ownerNCE",
    "task": "source field pointer reconstruction only", "labels_read": False,
    "frozen_cosine": "identical vectors/pools; cosine divided by .07",
    "tfidf": "train-source views only; word unigrams/bigrams, sublinear TF, lowercase, L2 norm"}


def source_items(docs, bindings):
    index = {doc["id"]: i for i, doc in enumerate(docs)}
    available = {doc["id"] for doc in docs if doc["status"] == "encodable"}
    items, excluded = [], []
    seen_sources = set()
    for source in bindings:
        if source["source_id"] in seen_sources or source["split"] not in {"source_train", "source_validation"}:
            raise ValueError("duplicate source or invalid internal split")
        seen_sources.add(source["source_id"])
        candidates = {c["id"]: c for c in source["candidates"]}
        queries = []
        for query in source["queries"]:
            if (not query["positive_ids"] or not set(query["positive_ids"]) <= set(query["candidate_ids"])
                    or not set(query["candidate_ids"]) <= set(candidates)):
                raise ValueError("positive/pool/source pointer mismatch")
            if query["document_id"] not in available or any(candidates[cid]["document_id"] not in available for cid in query["candidate_ids"]):
                excluded.append({"source_id": source["source_id"], "split": source["split"],
                    "query_id": query["id"], "status": "whole_query_pool_unavailable_length"})
                continue
            queries.append(query)
        if not queries:
            continue
        used = {cid for q in queries for cid in q["candidate_ids"]}
        cs = [c for c in source["candidates"] if c["id"] in used]
        ci = {c["id"]: i for i, c in enumerate(cs)}
        valid = torch.zeros((len(queries), len(cs)), dtype=torch.bool)
        positive = torch.zeros_like(valid)
        for i, query in enumerate(queries):
            if any(candidates[cid]["value_type"] != query["value_type"] for cid in query["candidate_ids"]):
                raise ValueError("query pool crosses broad source value types")
            valid[i, [ci[cid] for cid in query["candidate_ids"]]] = True
            positive[i, [ci[cid] for cid in query["positive_ids"]]] = True
        items.append({"source_id": source["source_id"], "split": source["split"], "queries": queries,
            "candidates": cs, "qidx": [index[q["document_id"]] for q in queries],
            "cidx": [index[c["document_id"]] for c in cs], "valid": valid, "positive": positive,
            "types": torch.tensor([TYPES.index(q["value_type"]) for q in queries], dtype=torch.long)})
    return items, excluded


def batch_masks(items):
    qtotal, ctotal = sum(len(i["queries"]) for i in items), sum(len(i["candidates"]) for i in items)
    valid = torch.zeros((qtotal, ctotal), dtype=torch.bool)
    positive = torch.zeros_like(valid)
    qidx, cidx, types, qpos, cpos = [], [], [], 0, 0
    for item in items:
        nq, nc = item["valid"].shape
        valid[qpos:qpos + nq, cpos:cpos + nc] = item["valid"]
        positive[qpos:qpos + nq, cpos:cpos + nc] = item["positive"]
        qidx.extend(item["qidx"])
        cidx.extend(item["cidx"])
        types.append(item["types"])
        qpos += nq
        cpos += nc
    return qidx, cidx, torch.cat(types), valid, positive


def row_records(scores, item, method):
    records = []
    beams = owner_beam(scores, topk=5)
    for i, query in enumerate(item["queries"]):
        order = torch.argsort(scores[i], descending=True, stable=True)
        order = [int(j) for j in order if torch.isfinite(scores[i, j])]
        rank = min(j + 1 for j, k in enumerate(order) if item["positive"][i, k])
        nce = float(owner_nce(scores[i:i + 1], item["positive"][i:i + 1]))
        records.append({"source_id": item["source_id"], "split": item["split"], "query_id": query["id"],
            "method": method, "nce": nce, "rank_first_positive": rank, "top1": int(rank == 1),
            "recall5": int(rank <= 5), "reciprocal_rank": 1 / rank,
            "has_homograph_negative": bool(query["homograph_negative_ids"]),
            "has_other_record_same_field": bool(query["other_record_same_field_ids"]),
            "candidate_count": len(order), "positive_count": len(query["positive_ids"]),
            "top5_candidate_ids": [item["candidates"][j]["id"] for j in beams[i]["candidate_indices"]],
            "supervision_scope": PROTOCOL["task"]})
    return records


def summarize(records, excluded):
    groups = defaultdict(list)
    for record in records:
        for subset, allowed in (("all", True), ("homograph", record["has_homograph_negative"]),
                                ("other_record_same_field", record["has_other_record_same_field"])):
            if allowed:
                groups[record["split"] + ":" + subset].append(record)
    result = {}
    for group, values in groups.items():
        by_source = defaultdict(list)
        for r in values:
            by_source[r["source_id"]].append(r)
        result[group] = {"queries": len(values), "sources": len(by_source),
            **{k: float(np.mean([r[k] for r in values])) for k in ("nce", "top1", "recall5", "reciprocal_rank")},
            "source_macro_top1": float(np.mean([np.mean([r["top1"] for r in rs]) for rs in by_source.values()]))}
    return {"groups": result, "excluded_query_count": len(excluded), "task": PROTOCOL["task"], "natural_ownership_accuracy": None}


def lexical_scores(docs, bindings, items):
    train_indices = {j for item in items if item["split"] == "source_train" for key in ("qidx", "cidx") for j in item[key]}
    train_docs = {docs[j]["id"] for j in train_indices}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm="l2", lowercase=True)
    vectorizer.fit([d["text"] for d in docs if d["id"] in train_docs])
    matrix = vectorizer.transform([d["text"] for d in docs])
    records = []
    for item in items:
        scores = torch.from_numpy((matrix[item["qidx"]] @ matrix[item["cidx"]].T).toarray()).float() / PROTOCOL["temperature"]
        scores = scores.masked_fill(~item["valid"], -torch.inf)
        records.extend(row_records(scores, item, "tfidf_train_source_fit"))
    return records, {"vocabulary_size": len(vectorizer.vocabulary_), "vocabulary_digest": digest(vectorizer.vocabulary_),
                     "fit_document_count": len(train_docs), "fit_document_ids_digest": digest(sorted(train_docs)),
                     "fit_uses_source_validation": False, "fit_uses_unavailable_or_excluded_pools": False}


def evaluate(items, vectors, model=None):
    records = []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for item in items:
            q, c = vectors[item["qidx"]], vectors[item["cidx"]]
            if model is None:
                scores = (F.normalize(q, dim=-1) @ F.normalize(c, dim=-1).T / PROTOCOL["temperature"]).masked_fill(~item["valid"], -torch.inf)
            else:
                scores = model(q, c, item["types"], valid=item["valid"])
            records.extend(row_records(scores, item, "frozen_hidden_cosine" if model is None else "source_rel_mini"))
    return records


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh training/baseline output required")
    if args.phase == "lexical":
        parent, docs, bindings = verify_prepared(args.features)
        matrix = None
    else:
        parent, docs, bindings, matrix = load_features(args.features)
    if parent["parent"]["kind"] != "source_reconstruction":
        raise ValueError("natural candidate outputs cannot be used as training supervision")
    if args.sanity != parent["parent"]["sanity_only"] or not 1 <= args.epochs <= 20:
        raise ValueError("sanity identity mismatch or epoch limit outside frozen plan")
    items, excluded = source_items(docs, bindings)
    if not all(any(i["split"] == split for i in items) for split in ("source_train", "source_validation")):
        raise ValueError("both disjoint source partitions need available queries")
    paths = list((GRAPH / "route_graph").glob("*.py")) + [Path(__file__).with_name(n) for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "source_relation_data.py", "source_relation_features.py",
         "source_relation_model.py", "source_relation_train.py")]
    settings = {"protocol": PROTOCOL, "phase": args.phase, "epochs": args.epochs, "sanity_only": args.sanity,
        "features_path": str(args.features.resolve()), "prepare_manifest_sha256": file_sha256(args.features / "prepare_manifest.json"),
        "features_manifest_sha256": file_sha256(args.features / "manifest.json") if matrix is not None else None,
        "code_sha256": freeze_code(args.output, paths), "internal_split": {i["source_id"]: i["split"] for i in items}}
    write_json_once(args.output / "settings.json", settings)
    write_json_once(args.output / "excluded_queries.json", excluded)
    lexical, lex_receipt = lexical_scores(docs, bindings, items)
    write_json_once(args.output / "tfidf_predictions.json", lexical)
    summaries = {"tfidf": {**summarize(lexical, excluded), "fit_receipt": lex_receipt}}
    history = []
    if matrix is not None:
        torch.manual_seed(PROTOCOL["seed"])
        vectors = torch.from_numpy(np.array(matrix, copy=True))
        baseline = evaluate(items, vectors)
        write_json_once(args.output / "frozen_predictions.json", baseline)
        summaries["frozen_cosine"] = summarize(baseline, excluded)
        model = SourceRelMini(input_dim=vectors.shape[1], hidden_dim=PROTOCOL["hidden_dim"],
            num_types=len(TYPES), temperature=PROTOCOL["temperature"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"])
        train = [i for i in items if i["split"] == "source_train"]
        validation = [i for i in items if i["split"] == "source_validation"]
        rng = torch.Generator().manual_seed(PROTOCOL["seed"])
        best, best_state, best_epoch = float("inf"), None, None
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train), generator=rng).tolist()
            weighted_loss, query_count = 0., 0
            for begin in range(0, len(order), PROTOCOL["batch_sources"]):
                batch = [train[k] for k in order[begin:begin + PROTOCOL["batch_sources"]]]
                qidx, cidx, types, valid, positive = batch_masks(batch)
                optimizer.zero_grad(set_to_none=True)
                scores = model(vectors[qidx], vectors[cidx], types, valid=valid)
                loss = owner_nce(scores, positive)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("nonfinite head gradient")
                optimizer.step()
                weighted_loss += float(loss.detach()) * len(qidx)
                query_count += len(qidx)
            val = summarize(evaluate(validation, vectors, model), excluded)["groups"]["source_validation:all"]
            history.append({"epoch": epoch, "training_query_mean_nce": weighted_loss / query_count, "validation": val})
            if val["nce"] < best:
                best, best_epoch = val["nce"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(json.dumps(history[-1], allow_nan=False), flush=True)
        model.load_state_dict(best_state)
        predictions = evaluate(items, vectors, model)
        write_json_once(args.output / "trained_predictions.json", predictions)
        summaries["source_rel_mini"] = summarize(predictions, excluded)
        torch.save({"state_dict": best_state, "input_dim": vectors.shape[1], "hidden_dim": PROTOCOL["hidden_dim"],
            "types": TYPES, "temperature": PROTOCOL["temperature"], "best_epoch": best_epoch,
            "settings_object_digest": digest(settings), "task": PROTOCOL["task"]}, args.output / "checkpoint.pt")
    write_json_once(args.output / "history.json", history)
    write_json_once(args.output / "summary.json", {"status": "complete", "sanity_only": args.sanity,
        "methods": summaries, "labels_read": False, "natural_ownership_accuracy": None,
        "native_forward_calls": 0, "complete_detector": False})
    verify_code(args.output, settings)
    verify_prepared(args.features)
    if matrix is not None and file_sha256(args.features / "manifest.json") != settings["features_manifest_sha256"]:
        raise ValueError("upstream feature manifest changed")
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_file_sha256": file_sha256(args.output / "settings.json"),
        "artifacts": {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file() and p.name != "settings.json"}})
    print(json.dumps({"status": "complete", "methods": summaries}, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["lexical", "train"])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sanity", action="store_true")
    run(parser.parse_args())
