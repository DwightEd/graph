def compute_reachability_potential(graph, targets):
    target_set = {int(node) for node in targets}
    outgoing = {node: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        outgoing[edge["source"]].append(edge)
    potential = {node: (1.0 if node in target_set else 0.0) for node in graph["nodes"]}
    for node in sorted(graph["nodes"], reverse=True):
        if node not in target_set:
            potential[node] = sum(edge["weight"] * potential[edge["target"]] for edge in outgoing[node])
    return potential


def reweight_edges(graph, potential):
    edges, totals = [], {}
    for edge in graph["edges"]:
        value = float(edge["weight"]) * float(potential.get(edge["target"], 0.0))
        if not value:
            continue
        totals[edge["source"]] = totals.get(edge["source"], 0.0) + value
        edges.append({**edge, "flow_weight": value})
    for edge in edges:
        edge["flow_weight"] /= totals[edge["source"]]
    return edges


def propagate_flow(graph, targets, edge_weights=None):
    edges = edge_weights or graph["edges"]
    incoming = {node: [] for node in graph["nodes"]}
    for edge in edges:
        incoming[edge["target"]].append(edge)
    flow = {node: (1.0 if node in targets else 0.0) for node in graph["nodes"]}
    edge_flow = []
    for node in sorted(graph["nodes"], reverse=True):
        total = sum(edge.get("flow_weight", edge["weight"]) for edge in incoming[node])
        for edge in incoming[node]:
            weight = edge.get("flow_weight", edge["weight"])
            amount = flow[node] * weight / total
            edge_flow.append({**edge, "flow": amount})
            flow[edge["source"]] = flow.get(edge["source"], 0.0) + amount
    return flow, edge_flow


def compute_edge_flow(graph, targets):
    potential = compute_reachability_potential(graph, targets)
    weighted = reweight_edges(graph, potential)
    _, edge_flow = propagate_flow(graph, targets, weighted)
    return {"potential": potential, "edges": edge_flow}


def compute_node_throughput(graph, edge_flow):
    throughput = {node: 0.0 for node in graph["nodes"]}
    for edge in edge_flow:
        throughput[edge["source"]] += float(edge["flow"])
        throughput[edge["target"]] += float(edge["flow"])
    return throughput


def extract_weighted_paths(graph, targets, top_k=10):
    """Return the highest-product source-to-target paths in the token DAG."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    incoming = {node: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        incoming[edge["target"]].append(edge)
    paths = []
    for target in targets:
        candidates = [(1.0, [int(target)])]
        while candidates:
            score, path = candidates.pop(0)
            previous = [edge for edge in incoming[path[0]] if edge["source"] not in path]
            if not previous:
                paths.append({"nodes": path, "weight": float(score), "target": int(target)})
                continue
            for edge in previous:
                candidates.append((score * float(edge["weight"]), [edge["source"]] + path))
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:top_k]
    return sorted(paths, key=lambda item: item["weight"], reverse=True)[:top_k]
