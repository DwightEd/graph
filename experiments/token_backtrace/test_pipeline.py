"""Disk-stage integration keeps missing work and every original token visible."""

import json

import numpy as np
import pytest

from .pipeline import score_document, write_new


def document():
    # No-event is an actual coverage outcome, not a label saying 'normal'.
    return dict(schema='measured_token_graph_v1', key='synthetic', source_id='source',
        token_ids=[3, 4, 5], character_offsets=[[0, 1], [0, 1], [1, 2]],
        labels_used=False, proposals_complete=True, events=[],
        thresholds=dict(direct=0., graph=0., reference_id='independent-B'))


def test_no_event_round_trip_keeps_all_tokens_and_unresolved_evidence(tmp_path):
    result = score_document(document())
    path = tmp_path / 'scores.json'
    write_new(path, result)
    saved = json.loads(path.read_text())
    assert saved['token_ids'] == [3, 4, 5]
    assert saved['scores']['low_evidence'] == [True, True, True]
    assert saved['scores']['risk'] == [-.5, -.5, -.5]
    assert saved['character_offsets'][:2] == [[0, 1], [0, 1]]
    with pytest.raises(FileExistsError):
        write_new(path, result)


def test_unfinished_proposals_cannot_be_published_as_complete():
    source = document()
    source['proposals_complete'] = False
    result = score_document(source)
    assert result['status'] == 'incomplete'
    np.testing.assert_array_equal(result['scores']['alarm'], [False, False, False])


def test_natural_label_used_contract_rejected():
    source = document()
    source['labels_used'] = True
    with pytest.raises(ValueError, match='truth-label'):
        score_document(source)


def test_unknown_proposal_completion_cannot_be_truthy():
    source = document()
    source['proposals_complete'] = 'unknown'
    with pytest.raises(ValueError, match='must be a boolean'):
        score_document(source)
