"""Frozen unsupervised token anomaly scores."""

import numpy as np


def score_graph(model, graph):
    reconstruction = model.reconstruct(graph)
    score = reconstruction["node_error"].copy()
    if len(graph.edge_index.T):
        target = graph.edge_index[1]
        edge_count = np.bincount(target, minlength=len(score))
        edge_sum = np.bincount(target, weights=reconstruction["edge_error"], minlength=len(score))
        score += np.divide(edge_sum, edge_count, out=np.zeros_like(edge_sum), where=edge_count > 0)
    return {"response_id": graph.response_id, "score": score,
            "node_error": reconstruction["node_error"],
            "edge_error": reconstruction["edge_error"]}


def calibrate_source_budget(scores, records, budget):
    maxima = {}
    for result, record in zip(scores, records):
        maxima.setdefault(record["source_id"], []).append(float(np.max(result["score"])))
    source_maxima = np.array([max(values) for values in maxima.values()])
    return {"threshold": float(np.quantile(source_maxima, 1 - budget, method="higher")),
            "budget": budget, "sources": len(source_maxima)}
