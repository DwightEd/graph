"""One constrained detector, with fixed removals of its actual components."""

import numpy as np

from ..message_carriers.token_representation import aggregate_units
from .calibration import channels
from .graph import dependence_laplacian, solve_residual

PRIMARY = "unified"
NEW_METHODS = (PRIMARY, "unified_no_graph", "unified_local_anchor", "unified_random_graph",
               "source_consensus", "flat_fusion")


def score_record(record, edges, scales, strength):
    observed = channels(record, scales)
    route, anchor = observed["route"], observed["source_anchor"]
    units, count = record["views"]["units"], len(route)
    laplacian, adjacency = dependence_laplacian(edges, count)
    residual, centered, diagnostic = solve_residual(route, units, laplacian, strength)
    random_laplacian, _ = dependence_laplacian(edges, count, random_endpoints=True)
    random_residual, _, _ = solve_residual(route, units, random_laplacian, strength)
    scores = {name: value.copy() for name, value in record["scores"].items()}
    scores.update(unified=anchor + residual, unified_no_graph=anchor + centered,
        unified_local_anchor=observed["local_anchor"] + residual,
        unified_random_graph=anchor + random_residual, source_consensus=anchor,
        flat_fusion=.5 * (anchor + route))
    for name in NEW_METHODS:
        # These unit means are exact model constraints, not rounded numerical solves.
        if name == "flat_fusion":
            mean = aggregate_units(scores[name], units)
        else:
            mean = observed["local_anchor"] if name == "unified_local_anchor" else anchor
        scores[name + "_unit_mean"] = mean.copy()
    components = dict(**observed, route_centered=centered, graph_residual=residual,
        random_graph_residual=random_residual, target=scores["target"], unit_id=scores["unit_id"])
    diagnostic.update(response_id=record["response"]["id"], tokens=count,
        measured_head_edges=len(edges["target"]), graph_nonzeros=adjacency.nnz,
        graph_weight_sum=float(adjacency.sum()), max_graph_correction=float(abs(residual - centered).max()),
        source_view_disagreement=float(abs(observed["local_anchor"] - observed["carrier_anchor"]).mean()))
    return scores, components, diagnostic
