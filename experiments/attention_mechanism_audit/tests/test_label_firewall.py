from pathlib import Path


def test_graph_construction_uses_neither_labels_nor_proxy_gnn_layers():
    root = Path(__file__).parents[1]
    source = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("capture.py", "graph.py", "collect.py", "data.py")
    )
    for forbidden in (
        ".labels(",
        "response_labels(",
        "prepare_evaluation_labels",
        "retain_embedded_labels=True",
        "GRUCell",
        "GATConv",
        "GraphConv",
        "HeadTransition",
    ):
        assert forbidden not in source
