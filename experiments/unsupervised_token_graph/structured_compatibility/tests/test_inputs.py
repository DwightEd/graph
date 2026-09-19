import numpy as np

from experiments.unsupervised_token_graph.structured_compatibility.inputs import (
    row_routes,
)


def test_row_routes_keeps_source_and_history_roles_separate():
    keys = np.array([0, 1, 4, 7, 8])
    weights = np.array([.2, .1, .3, .15, .2])
    lookup = np.array([True, False, False, False])
    values = row_routes(
        keys,
        weights,
        prompt_length=4,
        query=8,
        source_lookup=lookup,
        recent_window=2,
    )
    np.testing.assert_allclose(
        values,
        [.2, .1, .15, .3, .2],
    )
