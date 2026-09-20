import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from state_audit.audit import audit_run
from state_audit.capture import capture_run, capture_sample
from state_audit.datasets import load_examples
from state_audit.generation import generate_run, make_answer
from state_audit.storage import read_arrays, read_json


def test_full_offline_audit_and_labels(tiny_run):
    _, _, root, *_ = tiny_run
    settings = dict(window=2, minimum_mass=0.0, minimum_rise=0.0, roles=True)
    result = audit_run(root, root / "audit", settings)
    assert result["labeled_answers"] == 2
    assert result["error_tokens"] == 2
    assert result["continuation_tokens"] == 1
    assert result["continuation_fraction"] == 0.5
    assert result["purpose"] == "observational_audit_not_detector_evaluation"
    roles = (root / "audit" / "000000" / "roles_000.csv").read_text()
    assert "left_keys" in roles and len(roles.splitlines()) > 1


def test_generation_preserves_ids_and_does_not_inherit_labels(tiny_run, tmp_path):
    model, tokenizer, _, data, options, _ = tiny_run
    options = dict(options, mode="generate")
    example = load_examples(data)[1]
    answer = make_answer(model, tokenizer, example, options, seed=5)
    assert answer["token_ids"] == answer["prompt_ids"] + answer["response_ids"]
    assert answer["labels"] is None
    capture_sample(model, answer, tmp_path / "trace", [0])
    readout = read_arrays(tmp_path / "trace" / "readout.npz")
    assert len(readout["target_logp"]) == len(answer["response_ids"])
    assert np.isfinite(readout["logit_entropy"]).all()
    repeated = make_answer(model, tokenizer, example, options, seed=5)
    assert repeated["response_ids"] == answer["response_ids"]


def test_resume_preserves_files_and_rejects_changed_settings(tiny_run):
    model, tokenizer, root, data, options, settings = tiny_run
    path = root / "samples" / "000000" / "trace" / "layer_000.npz"
    timestamp = path.stat().st_mtime_ns
    generate_run(model, tokenizer, data, root, options, settings, resume=True)
    capture_run(model, root, None, resume=True)
    assert path.stat().st_mtime_ns == timestamp
    with pytest.raises(ValueError, match="settings changed"):
        generate_run(model, tokenizer, data, root, dict(options, seed=8), settings, resume=True)


def test_labels_do_not_change_observations(tiny_run):
    _, _, root, *_ = tiny_run
    settings = dict(window=2, minimum_mass=0.0, minimum_rise=0.0, roles=True)
    audit_run(root, root / "before", settings)
    answer_path = root / "samples" / "000001" / "answer.json"
    answer = copy.deepcopy(read_json(answer_path))
    answer["labels"] = []
    answer_path.write_text(json.dumps(answer))
    audit_run(root, root / "after", settings)
    for name in ("nodes.csv", "heads_000.csv", "roles_000.csv"):
        assert (root / "before" / "000001" / name).read_bytes() == (
            root / "after" / "000001" / name
        ).read_bytes()


def test_offline_import_does_not_load_torch():
    script = (
        "import sys; from state_audit.audit import audit_run; assert 'torch' not in sys.modules"
    )
    source = str(Path(__file__).parents[1] / "src")
    subprocess.run(
        [sys.executable, "-c", script], check=True, env=dict(os.environ, PYTHONPATH=source)
    )
