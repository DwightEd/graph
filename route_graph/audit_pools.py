"""Freeze structurally matched candidates before any semantic control labels."""

import numpy as np
import torch

from route_graph.audit_alignment import key_view, span_keys, text_nodes
from route_graph.frozen_reader import digest


def visibility(keys, queries):
    return (
        float((np.asarray(keys)[None, :] <= np.asarray(queries)[:, None]).mean())
        if keys
        else 0.0
    )


def ratio_matches(left, right):
    return left > 0 and right > 0 and 0.5 <= left / right <= 2


def control_pool(oracle, group, role_keys, limit=16):
    """Only position, length, baseline attention and visibility enter matching."""
    selected = set(group["keys"])
    count = len(selected)
    original_mass = oracle.mass(group)
    original_visibility = visibility(group["keys"], group["queries"])
    possible = {}
    for width in sorted({max(1, count // 2), count, count * 2}):
        for start in range(0, len(role_keys) - width + 1, max(1, width // 4)):
            keys = role_keys[start : start + width]
            if selected & set(keys):
                continue
            mass = oracle.mass(group, keys)
            visible = visibility(keys, group["queries"])
            if not (
                ratio_matches(len(keys), count)
                and ratio_matches(mass, original_mass)
                and abs(visible - original_visibility) <= 0.1
            ):
                continue
            candidate = {**group, "keys": keys}
            cid = digest(candidate)[:20]
            possible[cid] = {
                "id": cid,
                "group": candidate,
                "matching": {
                    "key_ratio": len(keys) / count,
                    "mass_ratio": mass / original_mass,
                    "visibility_difference": visible - original_visibility,
                },
            }
    return [possible[key] for key in sorted(possible)[:limit]]


def freeze_pools(oracle, search, row, alignment, question=None):
    prompt = oracle.contrast.prompt_length
    roles = {
        "source": alignment["source_keys"],
        "history": list(range(prompt, len(oracle.contrast.prefix))),
    }
    pools, finalists = {}, []
    families = [("content", "source"), ("content", "history"), ("route", None)]
    for kind, role in families:
        candidates = [
            item
            for item in search["ranked"]
            if item["group"]["kind"] == kind
            and (role is None or item["group"]["role"] == role)
        ]
        if not candidates:
            continue
        for item in candidates:
            pools[item["id"]] = control_pool(
                oracle, item["group"], roles[item["group"]["role"]]
            )
        # A structurally untestable root does not suppress testable descendants.
        chosen = next(
            (item for item in candidates if len(pools[item["id"]]) >= 2), candidates[0]
        )
        finalists.append({**chosen, "controls": pools[chosen["id"]]})
    mlp = next(
        (item for item in search["ranked"] if item["group"]["kind"] == "mlp"), None
    )
    views = {}
    for item in finalists:
        for candidate in [item] + item["controls"]:
            group = candidate["group"]
            view = key_view(row, alignment, group["role"], group["keys"])
            vid = digest(view)[:20]
            candidate["view_id"] = vid
            views[vid] = {**view, "id": vid, "node_type": "native_position"}
    nodes = text_nodes(row, alignment, len(oracle.contrast.prefix))
    for node in nodes:
        views.setdefault(node["id"], {**node, "node_type": "raw_text_unit"})
    # Frozen high-dimensional embeddings remain in the observer; norms only
    # match input interventions and never become a semantic/true-answer score.
    with torch.no_grad():
        ids = torch.tensor(oracle.contrast.prefix, device=oracle.model.device)
        norms = (
            oracle.model.get_input_embeddings()(ids).float().norm(dim=-1).cpu().tolist()
        )
    origin_pools = {}
    for item in finalists:
        group = item["group"]
        if group["kind"] != "content":
            continue
        variants = [group["keys"]]
        if group["role"] == "history" and question:
            spans = [
                q["answer_span"] for q in question.get("previous_answer_candidates", [])
            ]
            if question.get("previous_claim_span") is not None:
                spans.append(question["previous_claim_span"])
            variants.extend(
                span_keys(alignment, "history", span, len(oracle.contrast.prefix))
                for span in spans
            )
        for origin in variants:
            if not origin:
                continue
            oid = origin_pool_id(origin, group)
            pool = raw_origin_pool(
                origin, group, roles[group["role"]], norms, row, alignment
            )
            origin_pools[oid] = pool
            for entry in pool:
                views.setdefault(
                    entry["id"],
                    {
                        **entry["view"],
                        "id": entry["id"],
                        "node_type": "raw_origin_control",
                    },
                )
    return {
        "finalists": finalists,
        "mlp": mlp,
        "views": list(views.values()),
        "text_node_ids": [node["id"] for node in nodes],
        "input_embedding_norms": norms,
        "origin_control_pools": origin_pools,
        "selection": "native_effect_then_structural_control_feasibility_before_semantics",
        "control_pool_limit": 16,
        "candidate_control_audit": [
            {
                "id": item["id"],
                "delta": item["delta"],
                "pool_count": len(pools[item["id"]]),
                "controls": pools[item["id"]],
                "selected": item["id"] in {f["id"] for f in finalists},
                "status": "structurally_matchable"
                if len(pools[item["id"]]) >= 2
                else "no_two_structural_controls",
            }
            for item in search["ranked"]
            if item["id"] in pools
        ],
    }


def select_controls(item, semantic):
    """Select once using blinded semantics, before any control intervention."""
    candidates = []
    for control in item["controls"]:
        label = semantic.get(control["view_id"], {})
        if label.get("relation") == "unrelated":
            candidates.append(control)
    return sorted(candidates, key=lambda c: c["id"])[:2]


def origin_pool_id(origin, group):
    return digest(
        {"origin": list(origin), "role": group["role"], "target": group["keys"]}
    )


def raw_origin_pool(origin, group, role_keys, norms, row, alignment, limit=16):
    selected, target = set(origin), np.asarray(group["keys"])
    count = len(origin)
    origin_norm = float(np.mean([norms[key] for key in origin]))
    reach = float((np.asarray(origin)[:, None] <= target[None, :]).mean())
    candidates = {}
    for width in sorted({max(1, count // 2), count, count * 2}):
        for start in range(0, len(role_keys) - width + 1, max(1, width // 4)):
            keys = role_keys[start : start + width]
            if selected & set(keys):
                continue
            control_norm = float(np.mean([norms[key] for key in keys]))
            control_reach = float((np.asarray(keys)[:, None] <= target[None, :]).mean())
            if not (
                ratio_matches(len(keys), count)
                and ratio_matches(control_norm, origin_norm)
                and abs(control_reach - reach) <= 0.1
            ):
                continue
            view = key_view(row, alignment, group["role"], keys)
            vid = digest(view)[:20]
            candidates[vid] = {
                "id": vid,
                "keys": keys,
                "role": group["role"],
                "view": view,
                "matching": {
                    "key_ratio": len(keys) / count,
                    "embedding_norm_ratio": control_norm / origin_norm,
                    "causal_reach_difference": control_reach - reach,
                },
            }
    return [candidates[key] for key in sorted(candidates)[:limit]]


def origin_controls(origin, group, frozen, semantic, origin_role):
    if origin_role != group["role"]:
        raise ValueError("origin and input control roles must match")
    pool = frozen["origin_control_pools"].get(origin_pool_id(origin, group), [])
    return [
        {k: v for k, v in entry.items() if k != "view"}
        for entry in pool
        if entry["role"] == origin_role
        and semantic.get(entry["id"], {}).get("relation") == "unrelated"
    ][:2]
