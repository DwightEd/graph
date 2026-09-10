from control_graph.data import FactorialEvent, FactorialMargins
from control_graph.graph import ControlGraphBuilder


def event_with(margins: FactorialMargins) -> FactorialEvent:
    return FactorialEvent(
        event_id="event-1",
        source_id="source-1",
        split="train",
        relation="temporal",
        margins=margins,
    )


def test_builder_recovers_factorial_control_edges() -> None:
    graph = ControlGraphBuilder().build(
        event_with(
            FactorialMargins(
                onset_a=3.0,
                onset_b=-1.0,
                world_a_after_a=3.25,
                world_a_after_b=0.75,
                world_b_after_a=1.75,
                world_b_after_b=-3.75,
            )
        )
    )

    assert graph.signature() == {
        "source_onset": 2.0,
        "source_followup": 1.5,
        "prefix_followup": 2.0,
        "source_prefix_coupling": 0.75,
    }


def test_graph_signature_is_invariant_to_swapping_a_and_b() -> None:
    margins = FactorialMargins(
        onset_a=3.0,
        onset_b=-1.0,
        world_a_after_a=3.25,
        world_a_after_b=0.75,
        world_b_after_a=1.75,
        world_b_after_b=-3.75,
    )
    swapped = FactorialMargins(
        onset_a=-margins.onset_b,
        onset_b=-margins.onset_a,
        world_a_after_a=-margins.world_b_after_b,
        world_a_after_b=-margins.world_b_after_a,
        world_b_after_a=-margins.world_a_after_b,
        world_b_after_b=-margins.world_a_after_a,
    )

    builder = ControlGraphBuilder()
    assert builder.build(event_with(margins)).signature() == builder.build(
        event_with(swapped)
    ).signature()
