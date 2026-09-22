"""Human-readable observed writes and separately named local sensitivities."""

import numpy as np


def identity(manifest):
    context, panel = manifest["context"], manifest["panel"]
    return {
        "case_id": context["case_id"],
        "side": context["side"],
        "panel": panel["name"],
        "natural": panel["natural"],
    }


def query_row(manifest, trace):
    observed = int(trace["observed_id"]) >= 0
    numeric = (
        "attention_reconstruction_error",
        "attention_add_roundoff",
        "mlp_add_roundoff",
        "norm_roundoff",
        "unembedding_roundoff",
    )
    return dict(
        **identity(manifest),
        query=int(trace["query"]),
        decision_offset=int(trace["decision_offset"]),
        logit_gap=float(trace["logit_gap"]),
        entropy=float(trace["entropy"]),
        surprisal=-float(trace["observed_logp"]) if observed else None,
        observed_id=int(trace["observed_id"]) if observed else None,
        first_candidate_id=int(trace["candidate_ids"][0]),
        second_candidate_id=int(trace["candidate_ids"][1]),
        ledger_error=float(trace["ledger_error"]),
        numeric_absolute_sum=sum(float(np.abs(trace[name]).sum()) for name in numeric),
        capture_seconds=float(trace["capture_seconds"]),
    )


def ledger_rows(manifest, trace):
    rows = []
    for layer, (before, middle, after) in enumerate(trace["residual_scores"]):
        for kind, left, right, update, rounding in (
            (
                "attention",
                before,
                middle,
                trace["attention_score"][layer],
                trace["attention_add_roundoff"][layer],
            ),
            (
                "ffn",
                middle,
                after,
                trace["mlp_score"][layer],
                trace["mlp_add_roundoff"][layer],
            ),
        ):
            rows.append(
                dict(
                    **identity(manifest),
                    query=int(trace["query"]),
                    layer=layer,
                    sublayer=kind,
                    before=float(left),
                    after=float(right),
                    write=float(update),
                    add_roundoff=float(rounding),
                    positive_to_negative=bool(left > 0 and right < 0),
                    negative_to_positive=bool(left < 0 and right > 0),
                )
            )
    return rows


def head_rows(manifest, trace):
    rows = []
    direction_norm = np.linalg.norm(trace["direction"])
    for layer, head, group in np.ndindex(trace["group_route_mass"].shape):
        norm = float(trace["group_write_norm"][layer, head, group])
        score = float(trace["group_logit_write"][layer, head, group])
        keys = trace["group_ids"] == group
        rows.append(
            dict(
                **identity(manifest),
                query=int(trace["query"]),
                layer=layer,
                head=head,
                group=str(trace["group_names"][group]),
                route_mass=float(trace["group_route_mass"][layer, head, group]),
                write_norm=norm,
                logit_write=score,
                positive=float(trace["group_positive"][layer, head, group]),
                negative=float(trace["group_negative"][layer, head, group]),
                edge_energy_sum=float(trace["group_value_energy"][layer, head, group]),
                direction_cosine=score / (norm * direction_norm)
                if norm > 0 and direction_norm > 0
                else None,
                final_margin_sensitivity=float(
                    trace["edge_margin_sensitivity"][layer, head, keys].sum()
                ),
            )
        )
    return rows


def dependency_rows(manifest, trace, per_receiver=32):
    """CSV displays top absolute dependencies; full signed arrays remain in every NPZ."""
    rows = []
    for receiver, kind in enumerate(trace["receiver_kind"]):
        common = dict(
            **identity(manifest),
            query=int(trace["query"]),
            receiver_kind=str(kind),
            receiver_layer=int(trace["receiver_layer"][receiver]),
            receiver_head=int(trace["receiver_head"][receiver]),
            receiver_value=float(trace["receiver_value"][receiver]),
        )
        effect = trace["dependency_head_effect"][receiver]
        order = np.argsort(-np.abs(effect).ravel(), kind="stable")[:per_receiver]
        for flat in order:
            layer, head, group = np.unravel_index(flat, effect.shape)
            if not trace["dependency_head_observed"][receiver, layer, head]:
                continue
            rows.append(
                dict(
                    **common,
                    sender_kind="head_message",
                    sender_layer=layer,
                    sender_head=head,
                    source_group=str(trace["group_names"][group]),
                    local_sensitivity=float(effect[layer, head, group]),
                )
            )
        for layer in np.flatnonzero(trace["dependency_mlp_observed"][receiver]):
            rows.append(
                dict(
                    **common,
                    sender_kind="ffn_write",
                    sender_layer=int(layer),
                    sender_head=-1,
                    source_group="all",
                    local_sensitivity=float(
                        trace["dependency_mlp_effect"][receiver, layer]
                    ),
                )
            )
    return rows


def source_edge_rows(manifest, trace, top_heads=12, top_edges=4):
    """Show actual source words for high-magnitude reviewed-role writes, both signs."""
    candidates = trace["group_logit_write"][..., :3]
    order = np.argsort(-np.abs(candidates).ravel(), kind="stable")[:top_heads]
    tokens = manifest["context"]["token_text"]
    rows = []
    for flat in order:
        layer, head, group = np.unravel_index(flat, candidates.shape)
        keys = np.flatnonzero(trace["group_ids"] == group)
        ranked = keys[
            np.argsort(
                -np.abs(trace["edge_logit_write"][layer, head, keys]), kind="stable"
            )
        ]
        for key in ranked[:top_edges]:
            rows.append(
                dict(
                    **identity(manifest),
                    layer=layer,
                    head=head,
                    group=str(trace["group_names"][group]),
                    source=int(key),
                    token=tokens[key],
                    attention=float(trace["attention"][layer, head, key]),
                    logit_write=float(trace["edge_logit_write"][layer, head, key]),
                    local_sensitivity=float(
                        trace["edge_margin_sensitivity"][layer, head, key]
                    ),
                )
            )
    return rows
