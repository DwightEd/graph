from experiments.reanchor_flow.claims import (
    FORCED_CHUNK,
    NATURAL_BOUNDARY,
    RESPONSE_START,
    split_claims,
)


class TinyTokenizer:
    pieces = {
        1: "Alpha",
        2: " beta",
        3: ".",
        4: " Gamma",
        5: " delta",
        6: "!",
        7: " tail",
    }

    def decode(self, ids, **_):
        return "".join(self.pieces[int(index)] for index in ids)


def test_sentence_boundaries_are_absolute_and_label_free():
    spans = split_claims(TinyTokenizer(), [1, 2, 3, 4, 5, 6, 7], 0)
    assert [(span.start, span.stop) for span in spans] == [(0, 3), (3, 6)]
    assert [span.boundary_kind for span in spans] == [
        RESPONSE_START,
        NATURAL_BOUNDARY,
    ]


def test_short_trailing_filler_does_not_move_completed_span_end():
    spans = split_claims(
        TinyTokenizer(), [1, 2, 3, 7], 0, min_tokens=2
    )
    assert [(span.start, span.stop) for span in spans] == [(0, 3)]


def test_max_length_forces_a_boundary():
    spans = split_claims(
        TinyTokenizer(),
        [1, 2, 4, 5, 7],
        0,
        min_tokens=1,
        max_tokens=2,
    )
    assert [(span.start, span.stop) for span in spans] == [(0, 2), (2, 4), (4, 5)]
    assert [span.boundary_kind for span in spans] == [
        RESPONSE_START,
        FORCED_CHUNK,
        FORCED_CHUNK,
    ]
