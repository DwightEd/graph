def plot_token_graph(graph, edge_flow=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, graph["num_tokens"] / 2), 4))
    flow_by_pair = {(e["source"], e["target"]): e.get("flow", 0.0) for e in (edge_flow or [])}
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        width = 0.5 + 4.0 * flow_by_pair.get((source, target), 0.0)
        ax.plot([source, target], [0.05, 0.05], alpha=min(1.0, width / 4), linewidth=width, color="tab:blue")
    ax.scatter(graph["nodes"], [0.05] * len(graph["nodes"]), color="black", zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("token index")
    ax.set_title("target-related token flow")
    return ax


def plot_token_throughput(throughput, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    x = sorted(throughput)
    ax.bar(x, [throughput[i] for i in x], color="tab:orange")
    ax.set_xlabel("token index")
    ax.set_ylabel("throughput")
    ax.set_title("node throughput")
    return ax
