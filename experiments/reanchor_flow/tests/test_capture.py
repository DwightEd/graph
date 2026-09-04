import numpy as np

from experiments.reanchor_flow.capture import all_layer_gate


def test_all_layer_gate_uses_prediction_to_query_offset():
    edges = np.zeros((6, 6), dtype=bool)
    edges[0, 3] = True
    edges[3, 4] = True
    gate = all_layer_gate(edges, layer_count=4)
    assert gate.split_layer == 4
    assert gate.late_edges is None
    assert gate.source_mask is None
    assert gate.early_edges[2, 0]
    assert gate.early_edges[3, 3]
