"""Build the canonical signed constraint-control graph."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from control_graph.data import FactorialEvent

EDGE_ORDER = (
    "source_onset",
    "source_followup",
    "prefix_followup",
    "source_prefix_coupling",
)
EDGE_LAYOUT = (
    ("source_onset", ("source_constraint",), "onset_choice"),
    ("source_followup", ("source_constraint",), "followup_choice"),
    ("prefix_followup", ("generated_prefix",), "followup_choice"),
    (
        "source_prefix_coupling",
        ("source_constraint", "generated_prefix"),
        "followup_choice",
    ),
)
GRAPH_SCHEMA = "control-graph/constraint-control@1"


@dataclass(frozen=True)
class ControlEdge:
    kind: str
    sources: tuple[str, ...]
    target: str
    weight: float


@dataclass(frozen=True)
class ControlGraph:
    """A fixed-role graph for one factual commitment event."""

    event_id: str
    source_id: str
    split: str
    relation: str
    edges: tuple[ControlEdge, ...]

    def signature(self) -> dict[str, float]:
        return {edge.kind: edge.weight for edge in self.edges}

    def to_record(self) -> dict:
        return {
            "schema": GRAPH_SCHEMA,
            "event_id": self.event_id,
            "source_id": self.source_id,
            "split": self.split,
            "relation": self.relation,
            "edges": [
                {
                    "kind": edge.kind,
                    "sources": list(edge.sources),
                    "target": edge.target,
                    "weight": edge.weight,
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_record(cls, record: dict) -> ControlGraph:
        expected = {"schema", "event_id", "source_id", "split", "relation", "edges"}
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("control graph record has an invalid field set")
        if record["schema"] != GRAPH_SCHEMA or not isinstance(record["edges"], list):
            raise ValueError("control graph record has an unsupported schema")
        edge_fields = {"kind", "sources", "target", "weight"}
        if any(not isinstance(edge, dict) or set(edge) != edge_fields for edge in record["edges"]):
            raise ValueError("control graph record has an invalid edge field set")
        if any(
            not isinstance(edge["kind"], str)
            or not isinstance(edge["sources"], list)
            or not all(isinstance(source, str) for source in edge["sources"])
            or not isinstance(edge["target"], str)
            or type(edge["weight"]) not in {int, float}
            or not isfinite(edge["weight"])
            for edge in record["edges"]
        ):
            raise ValueError("control graph record has invalid edge values")
        edges = tuple(
            ControlEdge(
                kind=edge["kind"],
                sources=tuple(edge["sources"]),
                target=edge["target"],
                weight=float(edge["weight"]),
            )
            for edge in record["edges"]
        )
        graph = cls(
            event_id=record["event_id"],
            source_id=record["source_id"],
            split=record["split"],
            relation=record["relation"],
            edges=edges,
        )
        layout = tuple((edge.kind, edge.sources, edge.target) for edge in graph.edges)
        if layout != EDGE_LAYOUT:
            raise ValueError("control graph record has a non-canonical edge schema")
        return graph


class ControlGraphBuilder:
    """Convert factorial margins into four orthogonal causal graph edges."""

    def build(self, event: FactorialEvent) -> ControlGraph:
        m = event.margins
        source_followup = 0.25 * (
            m.world_a_after_a
            + m.world_a_after_b
            - m.world_b_after_a
            - m.world_b_after_b
        )
        prefix_followup = 0.25 * (
            m.world_a_after_a
            - m.world_a_after_b
            + m.world_b_after_a
            - m.world_b_after_b
        )
        interaction = 0.25 * (
            m.world_a_after_a
            - m.world_a_after_b
            - m.world_b_after_a
            + m.world_b_after_b
        )
        edges = (
            ControlEdge(
                "source_onset",
                ("source_constraint",),
                "onset_choice",
                0.5 * (m.onset_a - m.onset_b),
            ),
            ControlEdge(
                "source_followup",
                ("source_constraint",),
                "followup_choice",
                source_followup,
            ),
            ControlEdge(
                "prefix_followup",
                ("generated_prefix",),
                "followup_choice",
                prefix_followup,
            ),
            ControlEdge(
                "source_prefix_coupling",
                ("source_constraint", "generated_prefix"),
                "followup_choice",
                abs(interaction),
            ),
        )
        return ControlGraph(
            event.event_id,
            event.source_id,
            event.split,
            event.relation,
            edges,
        )
