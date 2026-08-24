from experiments.attention_holonomy_audit.graph import build_attention_event_graph
from experiments.attention_holonomy_audit.tests.helpers import make_sample


def test_event_graph_has_depth_relay_query_and_diamonds():
    graph = build_attention_event_graph(make_sample("s0", "g0"))
    assert graph.event_head_value.shape[1] == 2
    assert graph.depth_edge_index.shape[1] > 0
    assert graph.relay_edge_index.shape[1] > 0
    assert graph.diamond_index.shape[1] > 0
    assert graph.query_ptr[-1].item() == len(graph.query_event_index)
    graph.validate()
