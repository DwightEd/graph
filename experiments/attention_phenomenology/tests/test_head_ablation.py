import json

import numpy as np

from experiments.attention_phenomenology.head_ablation import compare_head_model_runs


def _write_predictions(path, score):
    np.savez_compressed(
        path,
        sample_id=np.asarray(["a", "a", "a", "b", "b", "b"]),
        source_id=np.asarray(["sa", "sa", "sa", "sb", "sb", "sb"]),
        task_type=np.asarray(["QA"] * 6),
        token_index=np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
        token_label=np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int8),
        current_probability=np.asarray(score, dtype=np.float32),
        forecast_probability=np.asarray(score, dtype=np.float32),
    )


def test_comparison_is_paired_by_sample_and_saves_current_and_forecast(tmp_path):
    reuse = tmp_path / "reuse.npz"
    no_reuse = tmp_path / "no_reuse.npz"
    output = tmp_path / "comparison.json"
    _write_predictions(reuse, [0.1, 0.2, 0.9, 0.1, 0.8, 0.7])
    _write_predictions(no_reuse, [0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

    result = compare_head_model_runs(
        reuse,
        no_reuse,
        output=output,
        bootstrap_replicates=20,
        seed=7,
    )

    assert result["current"]["reuse"]["auroc"] == 1.0
    assert result["current"]["delta_reuse_minus_no_reuse"]["auprc"] > 0
    assert result["forecast_1"]["tokens"] == 4
    assert result["bootstrap_unit"] == "complete sample_id"
    assert json.loads(output.read_text())["bootstrap_replicates"] == 20


def test_comparison_rejects_unmatched_token_rows(tmp_path):
    reuse = tmp_path / "reuse.npz"
    no_reuse = tmp_path / "no_reuse.npz"
    _write_predictions(reuse, [0.1, 0.2, 0.9, 0.1, 0.8, 0.7])
    _write_predictions(no_reuse, [0.1, 0.2, 0.9, 0.1, 0.8, 0.7])
    with np.load(no_reuse) as artifact:
        changed = {name: artifact[name].copy() for name in artifact.files}
    changed["token_index"][1] = 9
    np.savez_compressed(no_reuse, **changed)

    try:
        compare_head_model_runs(reuse, no_reuse, output=tmp_path / "out.json")
    except ValueError as error:
        assert "token_index" in str(error)
    else:
        raise AssertionError("unmatched token rows were accepted")
