"""Report exports must preserve readouts and distinguish missing from null."""

import json
import tarfile

import numpy as np
import pandas as pd

from experiments.path_conflict.paired_exports import adaptation_coverage, export_readouts
from experiments.path_conflict.paired_report import write_report


def test_report_exports_record_without_activation_arrays(tmp_path):
    directory = tmp_path / "pairs" / "claim" / "supported" / "onset"
    (directory / "worlds").mkdir(parents=True)
    context = dict(candidates=[[1, 2], [3, 4, 5]], readout="sequence_margin", position=4)
    (directory / "context.json").write_text(json.dumps(context))
    record = dict(measure=-1.5, correct_token_logps=[-.5, -2.], mean_margin=.25)
    for path in (directory / "baseline.npz", directory / "worlds" / "cut.npz"):
        np.savez(path, record=json.dumps(record), prefix=np.ones((3, 20)))

    summary = export_readouts(tmp_path)
    assert summary == dict(phases=1, baseline_records=1, missing_baseline_records=0, world_records=1)
    assert json.loads((directory / "baseline.json").read_text()) == record
    assert json.loads((directory / "worlds" / "cut.json").read_text()) == record
    write_report(tmp_path, summary)
    with tarfile.open(tmp_path / "paired_review.tar.gz") as archive:
        names = archive.getnames()
    assert "pairs/claim/supported/onset/baseline.json" in names
    assert not any(name.endswith(".npz") for name in names)


def test_missing_readout_and_same_layer_plan_are_not_zero_effects(tmp_path):
    directory = tmp_path / "pairs" / "claim" / "unsupported" / "onset"
    directory.mkdir(parents=True)
    (directory / "context.json").write_text(json.dumps(
        dict(candidates=[[1], [2]], readout="sequence_margin", position=3)))
    heads = [dict(layer=12, head=1), dict(layer=12, head=2)]
    pairs = [dict(left="L12H1_head_total", right="L12H2_head_total", relation="different_heads")]
    (directory.parents[1] / "plan.json").write_text(json.dumps(dict(heads=heads, pairs=pairs)))

    assert export_readouts(tmp_path)["missing_baseline_records"] == 1
    assert not (directory / "baseline.json").exists()
    coverage = adaptation_coverage(tmp_path, pd.DataFrame()).iloc[0]
    assert coverage.status == "not_measured_no_cross_layer_pair"
    assert coverage.measured_rows == 0
