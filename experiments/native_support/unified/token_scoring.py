"""Target-level source observations, routing, soft anchors and TV continuity."""

from ..message_carriers.token_representation import aggregate_units
from .graph import dependence_laplacian
from .token_solver import solve_tokens

PRIMARY = "unified_token"
NEW_METHODS = (PRIMARY, "unified_token_no_graph", "unified_token_no_anchor",
               "unified_token_no_continuity", "unified_token_no_token_source",
               "unified_token_random_graph", "token_observation")


def add_token_scores(record, edges, scores, components, graph_strength,
                     anchor_strength, continuity_strength):
    units = record["views"]["units"]
    source, observed = components["source_token"], components["token_observation"]
    laplacian, _ = dependence_laplacian(edges, len(source))
    random_laplacian, _ = dependence_laplacian(edges, len(source), random_endpoints=True)
    variants = {
        PRIMARY: (observed, source, laplacian, anchor_strength, graph_strength, continuity_strength),
        "unified_token_no_graph": (observed, source, laplacian, anchor_strength, 0, continuity_strength),
        "unified_token_no_anchor": (observed, source, laplacian, 0, graph_strength, continuity_strength),
        "unified_token_no_continuity": (observed, source, laplacian, anchor_strength, graph_strength, 0),
        "unified_token_no_token_source": (components["route"], components["source_anchor"], laplacian,
                                    anchor_strength, graph_strength, continuity_strength),
        "unified_token_random_graph": (observed, source, random_laplacian,
                                       anchor_strength, graph_strength, continuity_strength),
    }
    diagnostics = {}
    for name, (target, baseline, graph, alpha, gamma, strength) in variants.items():
        scores[name], diagnostics[name] = solve_tokens(target, baseline, components["source_anchor"],
                                                     units, graph, alpha, gamma, strength)
    scores["token_observation"] = observed.copy()
    for name in NEW_METHODS:
        scores[name + "_unit_mean"] = aggregate_units(scores[name], units)
    components["token_source_correction"] = scores[PRIMARY] - source
    return diagnostics
