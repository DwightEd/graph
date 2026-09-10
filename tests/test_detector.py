from control_graph.detector import GraphAnomalyDetector
from control_graph.graph import ControlEdge, ControlGraph


def graph(event_id: str, values: tuple[float, float, float, float]) -> ControlGraph:
    kinds = (
        ("source_onset", ("source_constraint",), "onset_choice"),
        ("source_followup", ("source_constraint",), "followup_choice"),
        ("prefix_followup", ("generated_prefix",), "followup_choice"),
        (
            "source_prefix_coupling",
            ("source_constraint", "generated_prefix"),
            "followup_choice",
        ),
    )
    return ControlGraph(
        event_id=event_id,
        source_id=f"source-{event_id}",
        split="train",
        relation="temporal",
        edges=tuple(
            ControlEdge(kind, sources, target, value)
            for (kind, sources, target), value in zip(kinds, values, strict=True)
        ),
    )


def test_detector_scores_control_graph_deviation_without_labels() -> None:
    fit_graphs = [
        graph(str(index), (1.0 + index * 0.01, 0.9, 0.2, 0.1))
        for index in range(12)
    ]
    detector = GraphAnomalyDetector().fit(fit_graphs)

    near, anomalous = detector.score(
        [
            graph("near", (1.05, 0.91, 0.21, 0.11)),
            graph("anomalous", (1.05, -2.0, 3.0, 2.0)),
        ]
    )

    assert anomalous.score > near.score
    assert anomalous.dominant_edge in anomalous.contributions
    assert set(anomalous.contributions) == {
        "source_onset",
        "source_followup",
        "prefix_followup",
        "source_prefix_coupling",
    }


def test_fit_requires_enough_graphs_to_define_a_reference_distribution() -> None:
    detector = GraphAnomalyDetector()

    try:
        detector.fit([graph("one", (1.0, 1.0, 0.0, 0.0))])
    except ValueError as error:
        assert "at least four" in str(error)
    else:
        raise AssertionError("fit accepted an undefined one-graph reference")
