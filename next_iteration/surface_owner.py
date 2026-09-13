"""Masked-context occurrence retrieval; scores are proposals, never support.

Original response values and their hidden states do not enter owner ranking.
Auxiliary masked-text encodings are explicitly distinct from native replay.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np

from next_iteration.surface_graph import _check
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest

OWNER_PROTOCOL = {
    "version": "masked-owner@1", "layers": [7, 15, 23], "topk": 4,
    "weights": {"masked_context_cosine": .65, "masked_context_lexical": .25, "path_lexical": .10},
    "near_tie_cost_gap": .03, "type_policy": "exact_surface_type",
    "target_surface_used_in_ranking": False, "training": "none",
    "features": "auxiliary_frozen_masked_text_post_block_last_token_not_native_trace",
}
_LEX_STOP = frozenset(["a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "to", "and", "or", "it", "its", "has", "have", "with", "for"])


def document(text):
    rendered = "Context:\n" + text + "\nEnd of context."
    return {"document_id": digest(rendered), "text": rendered, "context": text,
            "scope": "auxiliary_masked_context_not_native_generation_state"}


def feature_requests(response_graph, sources):
    _check(response_graph)
    _check(sources)
    docs, bindings = {}, []
    for kind, nodes, id_key in (("target_owner", response_graph["slots"], "slot_id"),
                                ("source_owner", sources["occurrences"], "occurrence_id")):
        for node in nodes:
            doc = document(node["masked_context"])
            docs[doc["document_id"]] = doc
            bindings.append({"kind": kind, "node_id": node[id_key], "document_id": doc["document_id"]})
    return {"documents": list(docs.values()), "bindings": bindings,
            "response_graph_sha256": response_graph["sha256"], "source_graph_sha256": sources["sha256"]}


def _words(text):
    text = re.sub(r"<[A-Z_]+>", " ", text)
    words = re.findall(r"[a-z]+", text.replace("_", " ").lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _LEX_STOP}


def _lexical(a, b):
    a, b = _words(a), _words(b)
    return 1 - len(a & b) / max(1, len(a | b))


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("owner features require matching finite three-layer vectors")
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denom == 0):
        raise ValueError("zero owner vector")
    return float(np.mean((1 - np.clip(np.sum(a * b, axis=1) / denom, -1, 1)) / 2))


def _array_sha(vector):
    return hashlib.sha256(np.asarray(vector).tobytes(order="C")).hexdigest()


def verify_capture(vectors, capture, documents, expected_model_identity):
    if not capture or not expected_model_identity:
        raise ValueError("dense features require capture records and frozen model identity")
    _check(capture)
    if (capture["schema"] != "masked-owner-capture@1" or capture["model_identity"] != expected_model_identity
            or capture["code_sha256"] != file_sha256(Path(__file__))
            or capture["representation"] != OWNER_PROTOCOL["features"]
            or capture["layers"] != OWNER_PROTOCOL["layers"]):
        raise ValueError("dense feature capture identity differs")
    records = {r["document"]["document_id"]: r for r in capture["records"]}
    if len(records) != len(capture["records"]):
        raise ValueError("duplicate captured document identity")
    for doc in documents:
        did = doc["document_id"]
        if did not in records or did not in vectors:
            raise ValueError("all exact masked documents need captured owner features")
        record, vector = records[did], np.asarray(vectors[did])
        if (record["document"] != doc or record["input_ids_sha256"] != digest(record["input_ids"])
                or not record["input_ids"] or any(type(t) is not int or t < 0 for t in record["input_ids"])
                or record["model_identity"] != expected_model_identity or record["layers"] != capture["layers"]
                or record["representation"] != capture["representation"]
                or vector.dtype != np.float32 or vector.shape != (3, capture["hidden_size"])
                or not np.isfinite(vector).all() or record["array_sha256"] != _array_sha(vector)):
            raise ValueError("dense feature array/document/identity differs from capture")


def match_owners(response_graph, sources, vectors, *, mode="masked_dense", capture=None, expected_model_identity=None):
    """All compatible occurrences contribute to denominators, including ties.

    Lexical-only mode is a named preflight/baseline; it cannot impersonate a
    completed high-dimensional feature stage. No finite truth is inferred here.
    """
    requests = feature_requests(response_graph, sources)
    if mode not in {"masked_dense", "lexical_only_preflight"}:
        raise ValueError("unknown matcher mode")
    if mode == "masked_dense":
        verify_capture(vectors, capture, requests["documents"], expected_model_identity)
    occurrences = sources["occurrences"]
    source_values = Counter(o["display_surface"].casefold().strip() for o in occurrences)
    matches = []
    for slot in response_graph["slots"]:
        compatible = [o for o in occurrences if o["surface_type"] == slot["surface_type"]]
        target_doc = document(slot["masked_context"])["document_id"]
        candidates = []
        for occ in compatible:
            source_doc = document(occ["masked_context"])["document_id"]
            lexical = _lexical(slot["masked_context"], occ["masked_context"])
            path = " ".join(str(x) for x in occ["field_path"] if type(x) is str)
            path_lexical = _lexical(slot["masked_context"], path) if path else lexical
            dense = _cosine(vectors[target_doc], vectors[source_doc]) if mode == "masked_dense" else None
            cost = (.65 * dense + .25 * lexical + .10 * path_lexical if dense is not None
                    else (.25 * lexical + .10 * path_lexical) / .35)
            candidates.append({"occurrence_id": occ["occurrence_id"], "parent_id": occ["parent_id"],
                "cost": cost, "features": {"masked_context_cosine": dense,
                "masked_context_lexical": lexical, "path_lexical": path_lexical},
                "source_document_id": source_doc,
                "same_source_value_occurrences": source_values[occ["display_surface"].casefold().strip()],
                "edit_policy": occ["edit_policy"]})
        candidates.sort(key=lambda c: (c["cost"], c["occurrence_id"]))
        selected = candidates[:OWNER_PROTOCOL["topk"]]
        ties = [c for c in candidates if selected and c["cost"] - selected[0]["cost"] <= OWNER_PROTOCOL["near_tie_cost_gap"]]
        matches.append({"slot_id": slot["slot_id"], "target_document_id": target_doc,
            "candidate_count": len(candidates), "selected": selected,
            "unsearched_count": max(0, len(candidates) - len(selected)),
            "near_tie_occurrence_ids": [c["occurrence_id"] for c in ties],
            "owner_ambiguous": len({c["parent_id"] for c in ties}) > 1,
            "within_parent_occurrence_ambiguous": len(ties) > 1,
            "null_or_unknown_always_available": True, "support_status": "not_assessed"})
    result = {"schema": "masked-owner-proposals@1", "mode": mode, "protocol": OWNER_PROTOCOL,
        "response_graph_sha256": response_graph["sha256"], "source_graph_sha256": sources["sha256"],
        "matches": matches, "labels_used": False, "native_forward_calls": 0,
        "dense_feature_stage_executed": mode == "masked_dense",
        "capture_sha256": capture["sha256"] if mode == "masked_dense" else None}
    return {**result, "sha256": digest(result)}


def capture_masked_contexts(model, tokenizer, documents, *, model_identity, batch_size=4, max_tokens=4096):
    """Pool frozen last-token states for exact masked documents, no truncation.

    Caller freezes model/code/input files and persists returned arrays/records.
    This encoder does not pretend the masked input is the original trace.
    """
    import torch

    if model.training or not model_identity or type(batch_size) is not int or batch_size < 1:
        raise ValueError("frozen eval model, identity and positive batch size required")
    if len({d["document_id"] for d in documents}) != len(documents):
        raise ValueError("deduplicate identical documents before capture")
    for doc in documents:
        if doc != document(doc["context"]):
            raise ValueError("masked document changed")
    layers = OWNER_PROTOCOL["layers"]
    if max(layers) >= len(model.model.layers):
        raise ValueError("frozen feature layer exceeds model")
    result, records, forwards = {}, [], 0
    for begin in range(0, len(documents), batch_size):
        batch = documents[begin:begin + batch_size]
        ids = [tokenizer.encode(d["text"], add_special_tokens=True) for d in batch]
        if any(not row or len(row) > max_tokens for row in ids):
            raise ValueError("masked context exceeds frozen limit; no silent truncation")
        lengths = [len(row) for row in ids]
        width = max(lengths)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        inputs = torch.tensor([row + [pad] * (width - len(row)) for row in ids], device=model.device)
        mask = torch.tensor([[1] * n + [0] * (width - n) for n in lengths], device=model.device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        vectors = np.empty((len(batch), len(layers), model.config.hidden_size), dtype=np.float32)
        seen, handles = [], []
        count = len(batch)

        def hook(column, *, batch_count=count, batch_lengths=lengths, target=vectors, visits=seen):
            def save(module, args, output):
                state = output[0] if isinstance(output, tuple) else output
                selected = state[torch.arange(batch_count, device=state.device),
                                 torch.tensor(batch_lengths, device=state.device) - 1]
                target[:, column] = selected.float().cpu().numpy()
                visits.append(column)
            return save

        for column, layer in enumerate(layers):
            handles.append(model.model.layers[layer].register_forward_hook(hook(column)))
        try:
            with torch.no_grad():
                model.model(input_ids=inputs, attention_mask=mask, position_ids=positions, use_cache=False)
            forwards += 1
        finally:
            for handle in handles:
                handle.remove()
        if sorted(seen) != list(range(len(layers))) or not np.isfinite(vectors).all():
            raise ValueError("masked feature capture incomplete or nonfinite")
        for doc, row, vector in zip(batch, ids, vectors, strict=True):
            result[doc["document_id"]] = vector
            records.append({"document": doc, "input_ids": row, "input_ids_sha256": digest(row),
                "model_identity": model_identity, "layers": layers,
                "representation": OWNER_PROTOCOL["features"], "batch_size": len(batch),
                "batch_padded_length": width, "array_sha256": _array_sha(vector)})
    receipt = {"schema": "masked-owner-capture@1", "records": records,
        "model_identity": model_identity, "layers": layers, "representation": OWNER_PROTOCOL["features"],
        "hidden_size": model.config.hidden_size, "code_sha256": file_sha256(Path(__file__)),
        "feature_forward_calls": forwards, "native_forward_calls": 0}
    return result, {**receipt, "sha256": digest(receipt)}
