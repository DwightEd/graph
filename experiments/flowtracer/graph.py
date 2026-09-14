"""Token DAG construction with source/sink sentinels."""


def build_token_dag(attention, token_ids=None, threshold=0.0):
    import numpy as np
    matrix = np.asarray(attention, dtype=float)
    n = matrix.shape[0]
    edges = []
    for target in range(n):
        for source in range(target):
            weight = float(matrix[target, source])
            if weight > threshold:
                edges.append({"source": source, "target": target, "weight": weight})
    return {"num_tokens": n, "token_ids": token_ids, "nodes": list(range(n)), "edges": edges}


def add_super_source(graph, weights=None):
    result = {**graph, "nodes": list(graph["nodes"]), "edges": list(graph["edges"])}
    source = graph["num_tokens"]
    values = weights or {node: 1.0 for node in graph["nodes"] if not any(e["target"] == node for e in graph["edges"])}
    result["nodes"].append(source)
    result["edges"].extend({"source": source, "target": int(node), "weight": float(weight)} for node, weight in values.items())
    result["super_source"] = source
    return result


def add_super_sink(graph, targets, weights=None):
    result = {**graph, "nodes": list(graph["nodes"]), "edges": list(graph["edges"])}
    sink = max(result["nodes"]) + 1 if result["nodes"] else 0
    values = weights or {int(node): 1.0 for node in targets}
    result["nodes"].append(sink)
    result["edges"].extend({"source": int(node), "target": sink, "weight": float(weight)} for node, weight in values.items())
    result["super_sink"] = sink
    result["targets"] = [int(node) for node in targets]
    return result
