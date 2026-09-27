"""Development-only deployment selection must finish before test scoring."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'teaching/state_audit/src'))
from experiments.probabilistic_detection.iteration import freeze_detector


def test_detector_selection_uses_development_auc_and_its_own_threshold(tmp_path):
    directory = tmp_path / 'QA'
    directory.mkdir()
    methods = {'logistic': .7, 'logistic_matched': .72, 'trees': .77, 'selected_gaussian': .74}
    values = {key: dict(auroc=value, threshold=index) for index, (key, value) in enumerate(methods.items())}
    (directory / 'development_metrics.json').write_text(json.dumps(values))
    (directory / 'readout_development_metrics.json').write_text(json.dumps({'selected_readout': dict(auroc=.8, threshold=9)}))
    freeze_detector(SimpleNamespace(output=tmp_path), 'QA')
    selected = json.loads((directory / 'detector_selection.json').read_text())
    assert selected['selected'] == 'selected_readout' and selected['threshold'] == 9
    assert selected['test_labels_used'] is False
    with pytest.raises(FileExistsError):
        freeze_detector(SimpleNamespace(output=tmp_path), 'QA')


def test_selection_refuses_to_start_after_test_predictions_exist(tmp_path):
    directory = tmp_path / 'QA'
    directory.mkdir()
    (directory / 'test_scores.npz').touch()
    with pytest.raises(FileExistsError):
        freeze_detector(SimpleNamespace(output=tmp_path), 'QA')
