import numpy as np
import pytest

from route_graph.detector import SourceReference


def point(source, value, token=0, task="QA"):
    return {
        "source_id": source,
        "response_id": source,
        "token_index": token,
        "prompt_tokens": 10,
        "task": task,
        "generator": "fixture",
        **{
            view: [value, value * 2]
            for view in ("residual", "observed", "null", "signal")
        },
    }


def test_reference_distances_are_label_free_source_balanced_and_local():
    rows = [point(f"train-{i}", i * 0.1) for i in range(4)]
    reference = SourceReference(neighbors=2, per_source=2).fit(rows)
    near = reference.score(point("test", 0.15))
    far = reference.score(point("test", 4))
    assert far["residual"] > near["residual"]
    duplicated = SourceReference(neighbors=2, per_source=2).fit(
        rows + [point("train-0", 0)] * 20
    )
    assert duplicated.score(point("test", 0.15)) == near
    with pytest.raises(ValueError, match="source-disjoint"):
        reference.score(point("train-0", 0.15))
    with pytest.raises(ValueError, match="reference stratum"):
        reference.score(point("test", 0.15, token=20))
    with pytest.raises(ValueError, match="finite"):
        SourceReference().fit([point("bad", np.nan)])


def test_constant_reference_coordinates_do_not_amplify_unseen_roundoff():
    rows = [point(f"train-{i}", i * 0.1) for i in range(4)]
    for row in rows:
        for view in ("residual", "observed", "null", "signal"):
            row[view][1] = 0
    reference = SourceReference(neighbors=2).fit(rows)
    query = point("test", 0.15)
    for view in ("residual", "observed", "null", "signal"):
        query[view][1] = 0
    original = reference.score(query)
    for view in ("residual", "observed", "null", "signal"):
        query[view][1] = 0.001
    assert reference.score(query) == original
