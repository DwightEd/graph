"""Regression: keep all prediction scores, not all full graph tensors."""

from test_charm_structure_audit import fixture_files, model


def test_prediction_records_do_not_retain_graph_tensors(tmp_path):
    from experiments.charm_structure_audit.data import load_graph, read_json
    from experiments.charm_structure_audit.train import predict

    root, prepared = fixture_files(tmp_path)
    record = next(r for r in read_json(prepared/'index.json') if r['split']=='test')
    graph, sample = load_graph(record, prepared)
    assert 'edge_attr' in graph and 'gold' not in graph
    assert not {'x', 'edge_index', 'edge_attr', 'edge_mark'} & sample.keys()
    predicted = predict(model(), [record], prepared, 'full', 0)
    assert not {'x', 'edge_index', 'edge_attr', 'edge_mark'} & predicted[0].keys()
    assert predicted[0]['score'].shape == predicted[0]['gold'].shape
