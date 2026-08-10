import matplotlib.pyplot as plt
import torch

from stats import channel_stats


def plot_layer_head(sample, metric="concentration", support_mass=0.8, ax=None):
    """Heatmap showing where a structural signal appears across layers and heads."""
    values = channel_stats(sample, support_mass)[metric].cpu()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(values, aspect="auto", origin="lower")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(metric)
    plt.colorbar(image, ax=ax)
    return ax


def plot_token_graph(graph, max_edges=300, ax=None):
    """Plot causal edges above token positions; prompt/response boundary is explicit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    edge_index = graph["edge_index"]
    if "edge_weight" in graph:
        score = graph["edge_weight"].float()
    else:
        score = graph["edge_attr"].float().amax(dim=1)

    if len(score) > max_edges:
        chosen = torch.topk(score, max_edges).indices
    else:
        chosen = torch.arange(len(score))

    for edge in chosen.tolist():
        source = int(edge_index[0, edge])
        target = int(edge_index[1, edge])
        height = max((target - source) / 2, 1)
        center = (source + target) / 2
        x = torch.linspace(source, target, 40)
        y = height * (1 - ((x - center) / height).square()).clamp_min(0)
        style = "-" if int(graph["edge_type"][edge]) == 0 else ":"
        ax.plot(x.numpy(), y.numpy(), linestyle=style, linewidth=0.5 + 2 * float(score[edge]))

    n = len(graph["token_ids"])
    ax.scatter(range(n), [0] * n, s=8)
    ax.axvline(graph["response_idx"] - 0.5, linestyle="--")
    ax.set_xlim(-1, n)
    ax.set_yticks([])
    ax.set_xlabel("Token position")
    ax.set_title("Prompt→Response (solid), Response→Response (dotted)")
    return ax
