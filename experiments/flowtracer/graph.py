"""Token DAG model for target-related attention flow."""

from dataclasses import dataclass, field


@dataclass
class TokenGraph:
    num_tokens: int
    token_ids: list | None = None
    edges: list[dict] = field(default_factory=list)
    super_source: int | None = None
    super_sink: int | None = None
    targets: list[int] = field(default_factory=list)

    @property
    def nodes(self):
        return list(range(self.num_tokens))

    @classmethod
    def from_attention(cls, attention, token_ids=None, threshold=0.0):
        import numpy as np
        matrix = np.asarray(attention, dtype=float)
        edges = [
            {"source": source, "target": target, "weight": float(matrix[target, source])}
            for target in range(matrix.shape[0])
            for source in range(target)
            if matrix[target, source] > threshold
        ]
        return cls(matrix.shape[0], token_ids, edges)

    def add_super_source(self, weights=None):
        source = self.num_tokens
        roots = {node for node in self.nodes if not any(edge["target"] == node for edge in self.edges)}
        values = weights or {node: 1.0 for node in roots}
        self.edges.extend({"source": source, "target": int(node), "weight": float(weight)} for node, weight in values.items())
        self.super_source = source
        return self

    def add_super_sink(self, targets, weights=None):
        sink = max(self.nodes) + 1
        values = weights or {int(node): 1.0 for node in targets}
        self.edges.extend({"source": int(node), "target": sink, "weight": float(weight)} for node, weight in values.items())
        self.super_sink, self.targets = sink, list(map(int, targets))
        return self

    def as_dict(self):
        return {"num_tokens": self.num_tokens, "token_ids": self.token_ids,
                "nodes": self.nodes, "edges": self.edges,
                "super_source": self.super_source, "super_sink": self.super_sink,
                "targets": self.targets}


def build_token_dag(attention, token_ids=None, threshold=0.0):
    return TokenGraph.from_attention(attention, token_ids, threshold).as_dict()


def add_super_source(graph, weights=None):
    model = TokenGraph(graph["num_tokens"], graph.get("token_ids"), list(graph["edges"]))
    return model.add_super_source(weights).as_dict()


def add_super_sink(graph, targets, weights=None):
    model = TokenGraph(graph["num_tokens"], graph.get("token_ids"), list(graph["edges"]))
    return model.add_super_sink(targets, weights).as_dict()
