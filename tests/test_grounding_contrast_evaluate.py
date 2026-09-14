import importlib.util
from pathlib import Path
import numpy as np
import pytest
import hashlib
import json
import subprocess
import sys


def module():
    path = Path(__file__).with_name('grounding_contrast_evaluate.py')
    if not path.exists():
        path = Path(__file__).parents[1] / 'next_iteration' / 'grounding_contrast_evaluate.py'
    spec = importlib.util.spec_from_file_location('p1_eval', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_perfect_reverse_tied():
    m = module()
    assert m.weighted_metrics([0, 1], [0, 1]) == dict(auroc=1., auprc=1.)
    assert m.weighted_metrics([0, 1], [1, 0]) == dict(auroc=0., auprc=.5)
    assert m.weighted_metrics([0, 1], [1, 1]) == dict(auroc=.5, auprc=.5)


def test_zero_weight_is_omitted_numerically():
    m = module()
    assert m.weighted_metrics([0, 1, 0], [1, 0, 2], [1, 1, 0]) == dict(auroc=0., auprc=.5)


def test_initial_errors_and_adjacent_spans():
    m = module()
    y, onset, through = m.annotation_targets([[0, 2], [2, 4], [4, 6], [6, 8]],
                                            [dict(start=0, end=4), dict(start=4, end=6)])
    np.testing.assert_array_equal(y, [1, 1, 1, 0])
    np.testing.assert_array_equal(onset, [1, 0, 1, 0])
    np.testing.assert_array_equal(through, [1, 0, 0, 0])


def test_empty_class_reported():
    assert module().weighted_metrics([0, 0], [1, 0])['auroc'] is None


def test_bad_alignment_fails():
    with pytest.raises(ValueError):
        module().annotation_targets([[0, 2]], [dict(start=3, end=4)])


def test_agrees_with_sklearn():
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = module()
    rng = np.random.default_rng(1)
    y = rng.integers(2, size=100)
    s = rng.integers(5, size=100)
    w = rng.random(100)
    metrics = m.weighted_metrics(y, s, w)
    assert metrics['auroc'] == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert metrics['auprc'] == pytest.approx(average_precision_score(y, s, sample_weight=w))


@pytest.mark.parametrize('has_error', [True, False])
def test_complete_evaluator_integration(tmp_path, has_error):
    """Tiny program-defined fixtures test plumbing, not reported ML results."""
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    annotations = tmp_path / 'annotations.jsonl'
    annotation_rows = []
    hashes = {}
    for rid in ['fixture1', 'fixture2']:
        response = 'ab'
        annotation_rows.append(dict(id=rid, source_id=rid, response=response,
                                    labels=[dict(start=1, end=2)] if has_error else []))
        record = dict(id=rid, source_id=rid, task='fixture', token_ids=[1, 2], offsets=[[0, 1], [1, 2]],
                      response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                      scores=dict(source_risk=[-1, 1], native_nll=[1, -1],
                                  native_entropy=[1, -1], native_negative_margin=[1, -1]))
        path = predictions / f'response_{rid}.json'
        path.write_text(json.dumps(record))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    annotations.write_text('\n'.join(map(json.dumps, annotation_rows)))
    (predictions / 'manifest.json').write_text(json.dumps(dict(
        complete=True, planned_ids=['fixture1', 'fixture2'], completed_ids=['fixture1', 'fixture2'],
        settings=dict(phase='development'), output_sha256=hashes)))
    script = Path(module().__file__)
    output = tmp_path / 'evaluation.json'
    subprocess.run([sys.executable, str(script), '--predictions', str(predictions),
                    '--annotations', str(annotations), '--output', str(output), '--bootstrap', '10'],
                   check=True, capture_output=True)
    result = json.loads(output.read_text())
    assert result['all_token']['tokens'] == 4
    assert result['all_token']['micro']['source_risk']['auroc'] == (1. if has_error else None)
    if not has_error:
        assert result['paired_source_bootstrap_micro']['native_nll']['auroc']['difference'] is None
