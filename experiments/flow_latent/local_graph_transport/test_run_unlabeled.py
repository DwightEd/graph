"""Score-stage label isolation, raw row alignment and immutable evaluation gates."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import run_unlabeled
from .run_capture import write_json


class ForbiddenLabels:
    def __getitem__(self, index):
        raise AssertionError('Label-free scoring accessed natural hallucination labels')


def toy_pack():
    pack = dict(context=np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
        observations=np.zeros((4, 5)), development=np.array([False, False, True, True]),
        source_index=np.array([0, 0, 1, 1]), token_id=np.arange(10, 14),
        target=np.array([0, 2, 0, 1]), labels=ForbiddenLabels())
    records = [dict(id='a', source_id='s0', partition='fit', tokens=3,
                   directory='responses/a', packed_start=0, packed_stop=2),
               dict(id='b', source_id='s1', partition='dev', tokens=2,
                   directory='responses/b', packed_start=2, packed_stop=4)]
    return pack, dict(records=records, sources=['s0', 's1'], split='train')


def test_reference_uses_original_exact_fit_values_and_never_indexes_labels(tmp_path):
    train, metadata = toy_pack()
    directory = tmp_path / metadata['records'][0]['directory']
    directory.mkdir(parents=True)
    raw = np.array([0., 99., 1.])
    np.savez(directory / 'observations.npz', token_id=[10, 99, 11],
             source_local=raw, source_full=raw, raw_route=raw)
    reference = run_unlabeled.fit_scalar_reference(train, metadata, tmp_path)
    np.testing.assert_array_equal(reference['source_local_values'], [0., 1.])
    changed = dict(train)
    changed['context'] = train['context'].copy()
    changed['context'][train['development']] = 9999
    repeated = run_unlabeled.fit_scalar_reference(changed, metadata, tmp_path)
    for name in reference:
        np.testing.assert_array_equal(reference[name], repeated[name])
    # Even a train-pack reconstruction ulp cannot change the original tied CDF.
    changed['context'][0, 0] = np.nextafter(0., 1.)
    repeated = run_unlabeled.fit_scalar_reference(changed, metadata, tmp_path)
    np.testing.assert_array_equal(repeated['source_local_values'], [0., 1.])


def test_raw_all_token_graph_is_scored_before_packed_valid_filter(tmp_path):
    pack, metadata = toy_pack()
    for record, ids in zip(metadata['records'], ([10, 99, 11], [12, 13])):
        directory = tmp_path / record['directory']
        directory.mkdir(parents=True)
        value = np.arange(len(ids), dtype=float)
        np.savez(directory / 'observations.npz', token_id=ids, source_local=value,
                 source_full=value, raw_route=value)
        captured = tmp_path / 'capture' / record['id']
        captured.mkdir(parents=True)
        attention = np.zeros((2, len(ids) + 1, 2, 8), dtype=np.float16)
        attention[0, 1:, :, 0] = .1
        np.savez(captured / 'arrays.npz', answer_ids=ids, local_attention=attention)
    reference = run_unlabeled.fit_scalar_reference(pack, metadata, tmp_path)
    scores, ranks, diagnostics = run_unlabeled.score_partition(
        reference, pack, metadata, tmp_path, tmp_path / 'capture')
    assert len(scores) == 8
    assert diagnostics['invalid_tokens'] == 1
    assert diagnostics['answers'][0]['answer_tokens'] == 3
    assert diagnostics['answers'][0]['evaluated_tokens'] == 2
    assert all(value.shape == (4,) for value in scores.values())
    assert all(np.isfinite(value).all() for value in ranks.values())


def test_evaluation_rejects_changed_frozen_predictions_before_loading_labels(tmp_path, monkeypatch):
    directory = tmp_path / 'unlabeled'
    directory.mkdir()
    prediction = directory / 'test_scores.npz'
    np.savez(prediction, source_unary=[.2, .8])
    digest = run_unlabeled.file_hash(prediction)
    write_json(directory / 'FREEZE.json', dict(hashes={'test_scores.npz': digest}, pack_hashes={}))
    np.savez(prediction, source_unary=[.8, .2])
    monkeypatch.setattr(run_unlabeled, 'evaluation_labels', lambda *args: pytest.fail(
        'Test labels were requested before rejecting altered predictions'))
    with pytest.raises(ValueError, match='changed before evaluation'):
        run_unlabeled.evaluate(SimpleNamespace(output=tmp_path, cache=tmp_path, bootstrap=2))


def test_reference_threshold_is_fit_mixture_weighted_not_normal_label_calibrated():
    values = np.array([0., 10., 10., 10.])
    weights = np.array([.5, 1 / 6, 1 / 6, 1 / 6])
    assert run_unlabeled.weighted_reference_threshold(values, weights, .5) == 0.
    assert run_unlabeled.weighted_reference_threshold(values, weights, .95) == 10.
