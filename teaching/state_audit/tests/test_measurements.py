import numpy as np

from state_audit.analysis.annotations import label_report, match_normal, span_row, token_spans
from state_audit.analysis.measurements import measure_heads, reanchor_candidates, source_messages
from state_audit.analysis.roles import probe_rows


def example_trace():
    answer = dict(
        token_ids=[0, 1, 2, 3],
        prompt_length=3,
        key_sources=[-1, 0, 1],
        special_token_ids=[0],
        evidence=[dict(id="a"), dict(id="b")],
    )
    attention = np.array([[[0.8, 0.1, 0.1, 0.0]]])
    trace = dict(attention=attention, value=np.ones((1, 4, 2)))
    return answer, trace


def test_special_removal_is_only_an_observational_view():
    answer, trace = example_trace()
    original = trace["attention"].copy()
    measured = measure_heads(trace, np.eye(2), answer)
    np.testing.assert_allclose(measured["ordinary_mass"], [[0.2]])
    np.testing.assert_allclose(measured["source_mass"], [[[0.5, 0.5]]])
    # Raw A·V retains its original 0.1 mass; audit normalization must not change messages.
    message = source_messages(trace, np.eye(2), np.array([False, True, False, False]))
    np.testing.assert_allclose(message, [[[0.1, 0.1]]])
    np.testing.assert_array_equal(trace["attention"], original)
    assert np.isnan(measured["evidence_history_cosine"]).all()


def test_reanchor_uses_same_source_past_only():
    masses = np.zeros((10, 2, 2))
    masses[:4, 0, 0] = 0.1
    masses[4:, 0, 0] = 0.8
    nodes = reanchor_candidates(masses, 4, 0.5, 0.2)
    assert nodes[0]["target"] == 4
    assert nodes[0]["source"] == 0
    assert np.isclose(nodes[0]["rise"], 0.7)
    changed = masses.copy()
    changed[7:] = 1
    old = [node for node in nodes if node["target"] < 7]
    new = [node for node in reanchor_candidates(changed, 4, 0.5, 0.2) if node["target"] < 7]
    assert old == new


def test_uniform_attention_does_not_identify_a_head_role():
    before = np.array([[[0.1, 0.1]]])
    row = probe_rows(before, before, 2, 0, 0)[0]
    assert np.isclose(row["positional"], 1)
    assert np.isclose(row["symbolic"], 1)
    assert not row["identifiable"]
    before = np.array([[[0.1, 0.8]]])
    row = probe_rows(before, before[..., ::-1], 2, 0, 0)[0]
    assert row["symbolic"] > row["positional"]


def test_span_overlap_merges_but_adjacent_annotations_stay_distinct():
    answer = dict(
        response="a b c d",
        response_ids=[1, 2, 3, 4],
        special_token_ids=[],
        response_offsets=[[0, 1], [2, 3], [4, 5], [6, 7]],
        evidence=[dict(id="fact")],
        labels=[dict(start=0, end=3), dict(start=2, end=3), dict(start=4, end=5)],
    )
    assert token_spans(answer) == [(0, 2), (2, 3)]
    _, report = label_report(answer, [], 1)
    assert report["error_tokens"] == 3 and report["continuation_tokens"] == 1
    answer["labels"] = None
    assert label_report(answer, [], 1)[1]["error_tokens"] is None


def test_normal_controls_match_length_repetition_and_are_disjoint():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    offsets, cursor = [], 0
    for word in words:
        offsets.append([cursor, cursor + len(word)])
        cursor += len(word) + 1
    answer = dict(
        response=text, response_ids=list(range(100)), response_offsets=offsets, special_token_ids=[]
    )
    controls = match_normal(answer, [(30, 34), (60, 64)], 4)
    assert all(control is not None and control[1] - control[0] == 4 for control in controls)
    assert set(range(*controls[0])).isdisjoint(range(*controls[1]))


def test_warmup_is_not_reported_as_negative_pre_onset_evidence():
    answer = dict(
        response="abcdefghijkl",
        response_offsets=[[i, i + 1] for i in range(12)],
        evidence=[dict(id="fact")],
    )
    early = span_row(answer, [], (4, 5), "hallucination", 0, window=4)
    assert early["onset_observable"]
    assert not early["pre_onset_observable"]
    assert early["pre_onset_queries_observed"] == 0
    later = span_row(answer, [], (8, 9), "hallucination", 0, window=4)
    assert later["pre_onset_observable"]
    assert later["pre_onset_queries_observed"] == 4
