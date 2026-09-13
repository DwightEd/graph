"""Stream frozen residual span features with an input/coordinate capture record."""

import hashlib
import json

import numpy as np

from route_graph.source_event_graph import _digest, validate_inventory


def span_mapping(inventory, catalog, input_ids, token_offsets, input_context):
    """Bind raw text, whole model input, and exact weighted token memberships.

    Offsets are inventory-relative or None for tokens outside that inventory.
    The caller must obtain them from the same frozen tokenizer as input_ids.
    Full prompt/context text remains in the record; spans do not truncate it.
    Pooling gives unit mass to non-whitespace characters only, regardless of
    whether the tokenizer attaches whitespace to a neighboring token.
    """
    validate_inventory(inventory)
    if catalog["sha256"] != _digest({k: v for k, v in catalog.items() if k != "sha256"}):
        raise ValueError("capture catalog integrity mismatch")
    if catalog["inventory_sha256"] != inventory["sha256"]:
        raise ValueError("capture catalog belongs to a different inventory")
    if set(input_context) != {"prompt", "response", "source_span"}:
        raise ValueError("capture requires original prompt/response/source coordinates")
    prompt, response = input_context["prompt"], input_context["response"]
    a, b = input_context["source_span"]
    if (not isinstance(prompt, str) or not isinstance(response, str)
            or type(a) is not int or type(b) is not int or not 0 <= a < b <= len(prompt)):
        raise ValueError("invalid original input context")
    if inventory["side"] == "source":
        if response or prompt[a:b] != inventory["text"]:
            raise ValueError("source feature capture must precede any response")
    elif response != inventory["text"]:
        raise ValueError("response capture differs from inventory text")
    if (len(input_ids) != len(token_offsets) or not input_ids
            or any(type(i) is not int or i < 0 for i in input_ids)):
        raise ValueError("capture input IDs and offsets do not align")
    text = inventory["text"]
    previous = -1
    for span in token_offsets:
        if span is None:
            continue
        if (not isinstance(span, list) or len(span) != 2
                or any(type(i) is not int for i in span)
                or not 0 <= span[0] < span[1] <= len(text) or span[0] < previous):
            raise ValueError("capture offsets must follow raw coordinate order")
        previous = span[0]
    character_counts = np.zeros(len(text), dtype=np.int32)
    for span in token_offsets:
        if span is not None:
            character_counts[slice(*span)] += 1
    character_mass = np.divide(1.0, character_counts, out=np.zeros(len(text)), where=character_counts > 0)
    character_mass[[i for i, char in enumerate(text) if char.isspace()]] = 0
    nodes = [*catalog["nodes"], *catalog["events"]]
    memberships = []
    for node in nodes:
        start, end = node["span"]
        members = []
        covered = set()
        for index, span in enumerate(token_offsets):
            if span is None:
                continue
            lo, hi = max(start, span[0]), min(end, span[1])
            if lo < hi:
                weight = float(character_mass[lo:hi].sum())
                if weight > 0:
                    members.append({"token_index": index, "weight": weight})
                covered.update(range(lo, hi))
        if not members or any(not text[i].isspace() and i not in covered for i in range(start, end)):
            raise ValueError("capture token mapping does not cover a catalog span")
        memberships.append({"id": node["id"], "span": node["span"], "members": members,
                            "raw_span_characters": end - start,
                            "non_whitespace_characters": sum(not text[i].isspace() for i in range(start, end)),
                            "pooled_character_mass": sum(m["weight"] for m in members)})
    result = {
        "schema": "span-feature-input-mapping@1", "inventory": inventory,
        "catalog_sha256": catalog["sha256"], "input_ids": input_ids,
        "input_sha256": _digest(input_ids), "token_offsets": token_offsets,
        "input_context": input_context, "input_context_sha256": _digest(input_context),
        "feature_memberships": memberships,
        "pooling": "non_whitespace_character_mass_split_between_overlapping_tokens; subword_states_include_context",
    }
    result = json.loads(json.dumps(result, allow_nan=False))
    return {**result, "sha256": _digest(result)}


def capture_record(catalog, mapping, vectors, encoder):
    """Bind the mapping, model identities and array bytes after capture."""
    inventory = mapping["inventory"]
    rebuilt = span_mapping(inventory, catalog, mapping["input_ids"], mapping["token_offsets"], mapping["input_context"])
    if rebuilt != mapping:
        raise ValueError("capture mapping integrity mismatch")
    expected_input = encoder[inventory["side"] + "_input_sha256"]
    if expected_input != mapping["input_sha256"]:
        raise ValueError("encoder input differs from captured token IDs")
    array = np.asarray(vectors)
    if (array.dtype != np.float32 or array.ndim != 3 or not np.isfinite(array).all()
            or array.shape[:2] != (len(mapping["feature_memberships"]), len(encoder["layers"]))):
        raise ValueError("captured feature shape/dtype differs from mapping")
    identity = {k: encoder[k] for k in (
        "model_sha256", "tokenizer_sha256", "layers", "representation", "capture_code_sha256",
    )}
    result = {"schema": "span-feature-capture@1", "mapping": mapping,
              "encoder": identity, "array_shape": list(array.shape),
              "array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    return {**result, "sha256": _digest(result)}


def capture_post_block(model, catalog, mapping, encoder):
    """One frozen forward; pool each layer on device and release token states.

    The returned arrays are high-dimensional residual features, not origins or
    attention attribution. Model loading and exact tokenizer preflight belong
    to the executing runner and must be frozen in its execution manifest.
    """
    import torch

    if model.training or encoder["representation"] != "post_block_residual_mean":
        raise ValueError("capture requires eval mode and post-block residual representation")
    layers = model.model.layers
    if any(i >= len(layers) or i < 0 for i in encoder["layers"]):
        raise ValueError("feature layer escapes model")
    # Validate the bound mapping before touching the model.
    expected = span_mapping(mapping["inventory"], catalog, mapping["input_ids"], mapping["token_offsets"], mapping["input_context"])
    if expected != mapping:
        raise ValueError("capture mapping integrity mismatch")
    result = np.empty((len(mapping["feature_memberships"]), len(encoder["layers"]), model.config.hidden_size), dtype=np.float32)
    handles, seen = [], []
    memberships = mapping["feature_memberships"]
    weights = np.zeros((len(memberships), len(mapping["input_ids"])), dtype=np.float32)
    for row, membership in enumerate(memberships):
        total = sum(item["weight"] for item in membership["members"])
        for item in membership["members"]:
            weights[row, item["token_index"]] = item["weight"] / total
    pool_weights = torch.tensor(weights, device=model.device)
    def hook_for(column):
        def save(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            state = state[0].float()
            pooled = pool_weights.to(state.device) @ state
            result[:, column] = pooled.cpu().numpy()
            seen.append(column)
        return save
    for column, layer in enumerate(encoder["layers"]):
        handles.append(layers[layer].register_forward_hook(hook_for(column)))
    try:
        with torch.no_grad():
            ids = torch.tensor([mapping["input_ids"]], device=model.device)
            model.model(input_ids=ids, use_cache=False, output_hidden_states=False, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()
    if sorted(seen) != list(range(len(encoder["layers"]))):
        raise ValueError("model did not execute each captured layer exactly once")
    return result, capture_record(catalog, mapping, result, encoder)
