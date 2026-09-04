from experiments.reanchor_flow.analyze import _select_mechanism_records


def test_mechanism_sampling_is_deterministic_and_source_diverse():
    records = [
        ("QA", "source-a", "qa-1"),
        ("QA", "source-a", "qa-2"),
        ("QA", "source-b", "qa-3"),
        ("QA", "source-c", "qa-4"),
        ("Summary", "source-d", "summary-1"),
        ("Summary", "source-e", "summary-2"),
    ]
    first = _select_mechanism_records(records, 2)
    second = _select_mechanism_records(list(reversed(records)), 2)
    assert first == second
    assert len(first & {"qa-1", "qa-2", "qa-3", "qa-4"}) == 2
    selected_sources = {
        source for _, source, sample in records if sample in first and sample.startswith("qa")
    }
    assert len(selected_sources) == 2
    assert {"summary-1", "summary-2"} <= first
