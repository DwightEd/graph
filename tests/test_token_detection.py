"""The default detector needs neither a reference cohort nor a mechanism experiment."""

from unittest.mock import patch

import numpy as np
import pytest
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from test_native_support import response

from experiments.native_support.routes import ROUTE_SCORES
from experiments.native_support.run import main
from experiments.native_support.token_detection import DIRECTORY, run_detection


def write_compact_fixture(output):
    responses, annotations = [], {}
    random = np.random.default_rng(4)
    for index in range(3):
        item = {**response(), "id": str(index), "source_id": f"source{index}"}
        values = {name: random.uniform(-.6, .6, 6) for name in ROUTE_SCORES}
        values.update(entropy=random.uniform(0, 3, 6), target=np.arange(6), query=np.arange(6, 12),
                      token_id=np.array(item["token_ids"][7:]), all_token_ids=np.array(item["token_ids"]),
                      source_mask=np.empty(0, bool), ledger_error=np.zeros(6))
        write_arrays(output / "route_filter_v3/features" / f"{index:04d}.npz", **values)
        responses.append(item)
        annotations[item["id"]] = {"token_ids": item["token_ids"][7:], "labels": [0, 1, 1, 0, 0, 0]}
    settings = {"model": "unused", "responses": responses}
    write_json(output / "settings.json", settings)
    write_json(output / "annotations.json", annotations)
    return settings, annotations


def test_cached_tokens_keep_baseline_and_ignore_labels_and_state_models(tmp_path, capsys):
    settings, annotations = write_compact_fixture(tmp_path)
    original = tmp_path / "route_filter_v3/features/0000.npz"
    original_bytes = original.read_bytes()
    with patch("experiments.native_support.token_features.extract_features", side_effect=AssertionError("raw extraction forbidden")), \
         patch("state_audit.model.load_model", side_effect=AssertionError("model loading forbidden")):
        main(["--stage", "score", "--output", str(tmp_path)])
    destination = tmp_path / DIRECTORY
    summary = read_json(destination / "summary.json")
    assert summary["reference_required"] is False
    assert summary["scored_tokens"] == 18
    scores = read_arrays(destination / "responses/0000/scores.npz")
    cached = read_arrays(original)
    for method in summary["methods"]:
        np.testing.assert_array_equal(scores[method], cached[method])
    np.testing.assert_array_equal(scores["risk"], cached[summary["primary_method"]])
    annotations["0"]["labels"] = [1, 0, 0, 1, 1, 1]
    write_json(tmp_path / "annotations.json", annotations)
    run_detection(tmp_path, settings)
    changed = read_arrays(destination / "responses/0000/scores.npz")
    for name in scores:
        np.testing.assert_array_equal(scores[name], changed[name])
    assert original.read_bytes() == original_bytes
    # Even with old stage outputs present, evaluate selects the new completed detector.
    write_json(tmp_path / "risk_fusion_v6/w16/summary.json", {"historical": True})
    main(["--stage", "evaluate", "--output", str(tmp_path)])
    assert "attention_displacement" in capsys.readouterr().out
    assert (destination / "report.html").is_file()


def test_single_response_needs_no_reference_and_bad_alignment_is_rejected(tmp_path):
    settings, _annotations = write_compact_fixture(tmp_path)
    settings["responses"] = settings["responses"][:1]
    write_json(tmp_path / "settings.json", settings)
    summary = run_detection(tmp_path, settings)
    assert summary["responses"] == 1
    assert summary["evaluation"]["status"] == "evaluated"
    settings["responses"][0]["token_ids"][-1] += 1
    with pytest.raises(ValueError, match="token IDs differ"):
        run_detection(tmp_path, settings)
