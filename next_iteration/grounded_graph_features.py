"""Bind complete source graphs and pre-token states to exact frozen inputs."""

from pathlib import Path

import numpy as np
import torch

from next_iteration.surface_graph import _check
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest

NODE_KINDS = ["literal_field_owner", "source_document_owner", "text_context", "surface_component", "word_atom",
    "field_provenance_bundle", "literal_record_provenance_bundle", "weekday_key_inventory", "text_context_provenance_bundle"]
EDGE_KINDS = ["component_of_source_parent", "member_of_provenance_bundle",
    "source_parent_to_component", "provenance_bundle_to_member"]
REPRESENTATION = "llama.model.last_hidden_state_after_final_norm; source_prompt_only; pretoken_query"


def _source_nodes(inventory, source_offsets):
    entries = [*inventory["fields"], *inventory["contexts"], *inventory["components"], *inventory["bundles"]]
    if len({n["id"] for n in entries}) != len(entries):
        raise ValueError("source inventory has duplicate node IDs")
    by_id = {n["id"]: n for n in entries}
    memberships = {}
    for entry in entries:
        if "member_ids" not in entry:
            a, b = entry["raw_span"]
            memberships[entry["id"]] = {i for i, (left, right) in enumerate(source_offsets)
                if left < b and right > a and right > left}
    pending = [n for n in entries if "member_ids" in n]
    while pending:
        available = [n for n in pending if all(x in memberships for x in n["member_ids"])]
        if not available:
            raise ValueError("cyclic or missing source bundle membership")
        for entry in available:
            memberships[entry["id"]] = set().union(*(memberships[x] for x in entry["member_ids"]))
        done = {n["id"] for n in available}
        pending = [n for n in pending if n["id"] not in done]
    nodes = [{"id": n["id"], "kind": n["kind"], "type": NODE_KINDS.index(n["kind"]),
        "prompt_token_indices": sorted(memberships[n["id"]]), "available": bool(memberships[n["id"]]),
        "raw_span": n.get("raw_span"), "member_ids": n.get("member_ids", []),
        "value_status": n.get("value_status", "not_semantic_truth")} for n in entries]
    indices = {n["id"]: i for i, n in enumerate(nodes)}
    edges = []
    for edge in inventory["edges"]:
        if edge["from"] not in by_id or edge["to"] not in by_id:
            raise ValueError("source graph edge refers to absent inventory node")
        kind = EDGE_KINDS.index(edge["kind"])
        if kind > 1:
            raise ValueError("raw inventory unexpectedly contains model reverse edges")
        a, b = indices[edge["from"]], indices[edge["to"]]
        edges.extend([[a, b, kind], [b, a, kind + 2]])
    return nodes, edges


def prepare_example(prompt_record, inventory, text, targets, tokenizer):
    """Targets are optional weak-training metadata; they never enter LM input.

    target item: {target_span:[a,b], source_owner_id:id}. Only the FIRST token
    overlapping the value is pointer-supervised. Generation uses every token.
    A natural response calls this with targets=[] and the same input schema.
    """
    _check(inventory)
    if (prompt_record["source_id"] != inventory["source_id"]
            or prompt_record["prompt"][slice(*prompt_record["source_span"])] != inventory["text"]):
        raise ValueError("source graph does not belong to original prompt")
    prompt = tokenizer(prompt_record["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = [tokenizer.bos_token_id] + prompt["input_ids"]
    if prompt_ids != prompt_record["prompt_token_ids"] or len(prompt_ids) != prompt_record["prompt_length"]:
        raise ValueError("original observer prompt/BOS tokenization differs")
    response = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not response["input_ids"]:
        raise ValueError("empty response/reconstruction text")
    offsets = [[0, 0], *[list(x) for x in prompt["offset_mapping"]]]
    a, b = prompt_record["source_span"]
    source_offsets = [[max(0, left - a), min(b - a, right - a)]
        if left < b and right > a else [0, 0] for left, right in offsets]
    nodes, edges = _source_nodes(inventory, source_offsets)
    node_indices = {n["id"]: i for i, n in enumerate(nodes)}
    pointer, target_records = {}, []
    for target in targets:
        target = {k: target[k] for k in ("target_span", "source_owner_id")}
        left, right = target["target_span"]
        if not 0 <= left < right <= len(text):
            raise ValueError("weak target escapes exact reconstructed text")
        selected = [i for i, (lo, hi) in enumerate(response["offset_mapping"]) if lo < right and hi > left and hi > lo]
        if not selected or target["source_owner_id"] not in node_indices:
            raise ValueError("weak target or its source owner has no aligned coordinate")
        owner = node_indices[target["source_owner_id"]]
        if not nodes[owner]["available"]:
            raise ValueError("weak source owner has no mapped source token")
        first = selected[0]
        if first in pointer and owner not in pointer[first]:
            raise ValueError("different weak target owners share one first token")
        pointer.setdefault(first, []).append(owner)
        target_records.append({**target, "token_indices": selected, "first_token": first, "owner_node_index": owner})
    ids = prompt_ids + response["input_ids"]
    plen = len(prompt_ids)
    packet = {"schema": "grounded-graph-feature-input@1", "representation": REPRESENTATION,
        "source_id": inventory["source_id"], "inventory_sha256": inventory["sha256"],
        "prompt_record": prompt_record, "response_text": text, "source_offsets": source_offsets,
        "response_offsets": [list(x) for x in response["offset_mapping"]], "input_ids": ids[:-1],
        "target_token_ids": response["input_ids"], "query_positions": list(range(plen - 1, len(ids) - 1)),
        "target_positions": list(range(plen, len(ids))), "nodes": nodes, "edges": edges,
        "pointer_supervision": [{"query_index": i, "owner_node_indices": sorted(set(o))} for i, o in sorted(pointer.items())],
        "weak_targets": target_records, "node_kind_vocabulary": NODE_KINDS, "edge_kind_vocabulary": EDGE_KINDS,
        "no_response_token_in_source_pool": True, "labels_read": False}
    return {**packet, "sha256": digest(packet)}


def validate_packet(packet, inventory, tokenizer):
    targets = [{k: t[k] for k in ("target_span", "source_owner_id")} for t in packet["weak_targets"]]
    rebuilt = prepare_example(packet["prompt_record"], inventory, packet["response_text"], targets, tokenizer)
    if rebuilt != packet:
        raise ValueError("source/query representation receipt differs from exact reconstruction")
    if packet["query_positions"] != [t - 1 for t in packet["target_positions"]]:
        raise ValueError("queries are not genuine pre-token states")
    if any(i >= packet["prompt_record"]["prompt_length"] for n in packet["nodes"] for i in n["prompt_token_indices"]):
        raise ValueError("source pool contains response/future token")


@torch.no_grad()
def capture(model, tokenizer, inventory, packet, model_identity):
    validate_packet(packet, inventory, tokenizer)
    if model.training or any(p.requires_grad for p in model.parameters()) or model.config.model_type != "llama":
        raise ValueError("capture requires a fully frozen eval Llama observer")
    if set(model_identity) != {"model_files", "tokenizer_files", "capture_code_sha256"}:
        raise ValueError("exact observer/tokenizer/capture identities required")
    if model_identity["capture_code_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("capture code identity differs from executed file")
    seen = []
    hook = model.model.norm.register_forward_hook(lambda module, inputs, output: seen.append(output))
    try:
        ids = torch.tensor([packet["input_ids"]], device=model.device)
        states = model.model(input_ids=ids, use_cache=False, output_hidden_states=False,
            output_attentions=False, return_dict=True).last_hidden_state
        if len(seen) != 1 or states.data_ptr() != seen[0].data_ptr():
            raise ValueError("returned features are not the actual final-norm output")
        h = states[0].float()
        if h.shape[0] != len(packet["input_ids"]) or not torch.isfinite(h).all():
            raise ValueError("observer returned nonfinite or misaligned token states")
        weights = torch.zeros((len(packet["nodes"]), packet["prompt_record"]["prompt_length"]), device=h.device)
        for index, node in enumerate(packet["nodes"]):
            keys = node["prompt_token_indices"]
            if keys:
                weights[index, keys] = 1. / len(keys)
        source = (weights @ h[:weights.shape[1]]).cpu().numpy()
        query = h[packet["query_positions"]].cpu().numpy()
    finally:
        hook.remove()
    if not np.isfinite(source).all() or not np.isfinite(query).all():
        raise ValueError("pooled frozen features are nonfinite")
    import hashlib
    receipt = {"schema": "grounded-graph-feature-capture@1", "packet_sha256": packet["sha256"],
        "representation": REPRESENTATION, "model_identity": model_identity, "actual_observer_forwards": 1,
        "observer_dtype": str(model.dtype), "attention": model.config._attn_implementation,
        "input_ids_sha256": digest(packet["input_ids"]), "final_norm_hook_executions": len(seen),
        "source_array_shape": list(source.shape), "query_array_shape": list(query.shape), "dtype": "float32",
        "source_array_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "query_array_sha256": hashlib.sha256(query.tobytes()).hexdigest(),
        "pooling": "unique prompt source-token mean; bundle uses token union, no duplicate membership weighting",
        "native_route_claim": False, "labels_read": False}
    return {"source": source, "query": query}, {**receipt, "sha256": digest(receipt)}
