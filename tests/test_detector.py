from control_graph.detector import GraphAnomalyDetector
from control_graph.graph import ControlEdge, ControlGraph


def graph(
    event_id: str,
    values: tuple[float, float, float, float],
    *,
    relation: str = "temporal",
) -> ControlGraph:
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
        relation=relation,
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
    assert anomalous.dominant_edge in anomalous.edge_deviations
    assert anomalous.edge_deviations["source_followup"] < 0
    assert anomalous.edge_deviations["prefix_followup"] > 0
    assert set(anomalous.edge_deviations) == {
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


def test_detector_calibrates_each_relation_separately() -> None:
    temporal = [
        graph(f"temporal-{index}", (1.0, 1.0, 0.1, 0.1))
        for index in range(4)
    ]
    numerical = [
        graph(
            f"numerical-{index}",
            (10.0, 10.0, 5.0, 5.0),
            relation="numerical",
        )
        for index in range(4)
    ]

    detector = GraphAnomalyDetector().fit(temporal + numerical)
    score = detector.score(
        [graph("same-numerical", (10.0, 10.0, 5.0, 5.0), relation="numerical")]
    )[0]

    assert score.score == 0.0


def test_detector_rejects_a_relation_without_a_reference_profile() -> None:
    detector = GraphAnomalyDetector().fit(
        [graph(str(index), (1.0, 1.0, 0.1, 0.1)) for index in range(4)]
    )

    try:
        detector.score([graph("unknown", (1.0, 1.0, 0.1, 0.1), relation="causal")])
    except ValueError as error:
        assert "no fitted reference" in str(error)
    else:
        raise AssertionError("detector silently reused another relation's profile")
