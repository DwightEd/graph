"""Coordinate-based owner views; no global substring replacement of values.

The caller supplies semantically appropriate masks and protected constraints.
The compiler preserves all other bytes and records exact output/input segments.
It cannot discover undisclosed value dependencies or validate semantic roles.
"""

import hashlib

from next_iteration.surface_graph import _check, _sealed


def masked_view(text, *, document_id, masks, protected=()):
    """Build a payload-safe text view without raw target values in metadata.

    mask: {id, span:[start,end], role:target_value|candidate_value|linked_value}.
    Protected spans represent non-target conditions/identities that may contain
    identical strings. If such a constraint shares a masked coordinate, abstain.
    """
    if not isinstance(text, str) or not isinstance(masks, list):
        raise TypeError("require raw text and an explicit mask roster")
    def span(pair):
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair)
                or not 0 <= pair[0] < pair[1] <= len(text)):
            raise ValueError("invalid original mask/protected span")
        return pair
    protections = [span(p) for p in protected]
    ordered = sorted(masks, key=lambda m: span(m["span"]))
    ids = set()
    for index, mask in enumerate(ordered):
        if not isinstance(mask["id"], str) or not mask["id"] or mask["id"] in ids:
            raise ValueError("distinct mask identities required")
        ids.add(mask["id"])
        a, b = span(mask["span"])
        if mask["role"] not in ("target_value", "candidate_value", "linked_value"):
            raise ValueError("unknown mask role")
        if index and ordered[index - 1]["span"][1] > a:
            raise ValueError("overlapping masks need one explicit shared-value member")
        if any(a < d and c < b for c, d in protections):
            raise ValueError("mask overlaps an independently required non-target constraint")
    pieces, segments, cursor, out_cursor = [], [], 0, 0
    for index, mask in enumerate(ordered):
        a, b = mask["span"]
        if a > cursor:
            pieces.append(text[cursor:a])
            segments.append({"kind": "preserved", "input": [cursor, a], "output": [out_cursor, out_cursor + a - cursor]})
            out_cursor += a - cursor
        placeholder = f"<VALUE_{index}>"
        pieces.append(placeholder)
        segments.append({"kind": "masked", "input": [a, b], "output": [out_cursor, out_cursor + len(placeholder)],
            "mask_index": index, "role": mask["role"]})
        out_cursor += len(placeholder)
        cursor = b
    if cursor < len(text):
        pieces.append(text[cursor:])
        segments.append({"kind": "preserved", "input": [cursor, len(text)],
                         "output": [out_cursor, out_cursor + len(text) - cursor]})
    rendered = "".join(pieces)
    # The output carries opaque indices. Human/model-derived IDs and quotations
    # remain only in the caller's separate provenance object, never this payload.
    return _sealed({"schema": "masked-owner-view@1", "document_id": str(document_id),
        "original_sha256": hashlib.sha256(text.encode()).hexdigest(), "original_length": len(text),
        "text": rendered, "segments": segments, "mask_count": len(ordered),
        "mask_roster_sha256": _sealed({"masks": masks, "protected": protections})["sha256"],
        "dependency_completeness": "caller_obligation_not_inferred_from_text", "labels_used": False})


def validate_view(text, view, *, masks, protected=()):
    _check(view)
    expected = masked_view(text, document_id=view["document_id"], masks=masks, protected=protected)
    if expected != view:
        raise ValueError("masked view differs from original coordinates or mask roster")


def owner_payload(response_view, source_views):
    """Allowlist the reader interface; never serialize original graphs or IDs.

    Use separate fidelity requests for the unmasked assertion. The downstream
    producer must bind this exact request to its relation-table receipt.
    """
    _check(response_view)
    if not any(s.get("role") == "target_value" for s in response_view["segments"]):
        raise ValueError("owner query needs an explicitly masked response target")
    for view in source_views:
        _check(view)
        if not any(s.get("role") in ("candidate_value", "linked_value") for s in view["segments"]):
            raise ValueError("each owner source view needs explicit candidate value masks")
    return {"response_context": response_view["text"],
        "sources": [{"source_index": i, "context": v["text"]} for i, v in enumerate(source_views)],
        "value_holes": [{"mask_index": s["mask_index"], "role": s["role"]}
                        for s in response_view["segments"] if s["kind"] == "masked"],
        "view_identities": {"response": response_view["sha256"], "sources": [v["sha256"] for v in source_views]}}
