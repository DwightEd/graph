"""Saved budget updates and labelled graph endpoints; no graph recomputation."""

import numpy as np

from .routes import EPS
from .transport_state import budget_risk


def role_budgets(budget, source_count):
    """Sum only for auditing the final readout: [token, layer, source/history/other]."""
    return np.stack((budget[..., :source_count].sum(axis=(-1, -2)),
                     budget[..., source_count].sum(axis=-1),
                     budget[..., source_count + 1].sum(axis=-1)), axis=-1)


def state_measurements(state, source_count, strength, labels, valid):
    observed = np.asarray(state["observed_budget"], dtype=np.float64)
    inferred = state["inferred_budget"]
    edges = state["edge_weight"].astype(np.float64)
    weights = edges + edges.T
    degree = weights.sum(axis=1)
    reuse = state["reuse_weight"]
    reuse_degree = reuse.sum(axis=0) + reuse.sum(axis=1)
    old_roles, new_roles = (role_budgets(value, source_count) for value in (observed, inferred))
    old_share, new_share = (value / np.maximum(value.sum(axis=-1, keepdims=True), EPS)
                            for value in (old_roles, new_roles))
    # The linear state equation commutes with role sums. Check its readout
    # coordinates cheaply; this does not verify every individual budget entry.
    residual = ((1 + strength * degree[:, None, None]) * new_roles
                - strength * np.einsum("ts,slg->tlg", weights, new_roles) - old_roles)
    denominator = np.maximum(np.linalg.norm(observed.reshape(len(labels), -1), axis=1), EPS)
    delta = (inferred - observed).reshape(len(labels), -1)
    return {"degree": degree, "reuse_degree": reuse_degree,
        "kernel_mass_fraction": np.divide(degree, reuse_degree, out=np.full(len(labels), np.nan), where=reuse_degree > 0),
        "retention_weight": state["retention_weight"], "posterior_variance": state["posterior_variance"],
        "relative_budget_change": np.linalg.norm(delta, axis=1) / denominator,
        "max_abs_budget_change": np.abs(delta).max(axis=1),
        "relative_role_residual": np.linalg.norm(residual.reshape(len(labels), -1), axis=1)
            / np.maximum(np.linalg.norm(old_roles.reshape(len(labels), -1), axis=1), EPS),
        "source_share": old_share[..., 0].mean(axis=1), "history_share": old_share[..., 1].mean(axis=1),
        "delta_source_share": (new_share[..., 0] - old_share[..., 0]).mean(axis=1),
        "delta_history_share": (new_share[..., 1] - old_share[..., 1]).mean(axis=1),
        "raw_budget_readout": budget_risk(observed, source_count),
        "state_budget_readout": budget_risk(inferred, source_count),
        **neighbor_labels(weights, labels, valid)}


def neighbor_labels(weights, labels, valid):
    """Labels describe target tokens attached to query-state nodes, not their keys."""
    known_mass = weights @ valid.astype(float)
    error_mass = weights @ (valid & (labels == 1)).astype(float)
    total_mass = weights.sum(axis=1)
    return {"error_neighbor_fraction": np.divide(error_mass, known_mass,
                out=np.full(len(labels), np.nan), where=known_mass > 0),
            "unknown_neighbor_mass": total_mass - known_mass,
            "opposite_label_mass": np.where(labels == 1, known_mass - error_mass, error_mass)}


def edge_rows(state, labels, valid):
    """Report both query-target and historical-key labels; neither proves causality."""
    reuse, edges = state["reuse_weight"], state["edge_weight"]
    receivers, donors = np.nonzero(reuse)
    query_label = np.where(valid, labels, -1)
    # Donor state s is query P+s-1, so its answer key is target s-1.
    key_label = np.r_[-1, query_label[:-1]]
    result = []
    for endpoint, donor_labels in (("query_target", query_label), ("answer_key", key_label)):
        for receiver_label in (-1, 0, 1):
            for donor_label in (-1, 0, 1):
                selected = ((query_label[receivers] == receiver_label) & (donor_labels[donors] == donor_label))
                row, column = receivers[selected], donors[selected]
                raw_mass = float(reuse[row, column].sum())
                mass = float(edges[row, column].sum())
                result.append({"donor_label_alignment": endpoint, "receiver_label": receiver_label,
                    "donor_label": donor_label, "observed_edges": int(selected.sum()),
                    "conditioned_edges": int(np.count_nonzero(edges[row, column])),
                    "reuse_mass": raw_mass, "conditioned_mass": mass,
                    "retained_fraction": mass / raw_mass if raw_mass else None,
                    "mean_query_distance": float(np.dot(edges[row, column], row - column) / mass) if mass else None})
    return result


def state_groups(tokens):
    fields = ("degree", "reuse_degree", "kernel_mass_fraction", "retention_weight", "posterior_variance",
              "relative_budget_change", "relative_role_residual", "delta_raw_route", "abs_delta_raw_route",
              "delta_source_share", "delta_history_share", "error_neighbor_fraction", "opposite_label_mass")
    partitions = {"phase": [row["phase"] for row in tokens],
                  "label_half": [f'{row["label"]}_{row["half"]}' for row in tokens],
                  "response_id": [row["response_id"] for row in tokens]}
    result = []
    for partition, groups in partitions.items():
        groups = np.asarray(groups)
        for group in np.unique(groups):
            selected = groups == group
            for field in fields:
                values = np.asarray([row[field] for row in tokens], dtype=float)[selected]
                finite = values[np.isfinite(values)]
                result.append({"partition": partition, "group": str(group), "field": field,
                    "tokens": int(selected.sum()), "observed": len(finite),
                    "mean": float(finite.mean()) if len(finite) else None,
                    "median": float(np.median(finite)) if len(finite) else None,
                    "q90": float(np.quantile(finite, .9)) if len(finite) else None})
    return result
