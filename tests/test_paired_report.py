"""CSV report regressions; these checks need no LLM or torch installation."""

import pandas as pd

from experiments.path_conflict.paired_report import collect, control_counts, report_pairs


def save_phase(root, phase, name, frame):
    directory = root / "pairs" / "claim" / "supported" / phase
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / (name + ".csv"), index=False)


def test_object_booleans_count_failures_without_bitwise_negation():
    frame = pd.DataFrame({"numeric_ok": pd.Series([True, False, None], dtype=object)})
    tables = dict.fromkeys(("effects", "interactions", "adaptation", "persistence"), frame)
    assert control_counts(tables)["adaptation"] == dict(rows=3, passed=1, failed=1, missing=1)


def test_report_empty_phases_do_not_create_negative_control_counts(tmp_path):
    pd.DataFrame([dict(source_id="source", case_id="claim")]).to_csv(
        tmp_path / "pair_inventory.csv", index=False)
    for name, count in (("adaptation", 32), ("persistence", 40)):
        empty = pd.DataFrame(columns=["case_id", "source_id", "side", "phase"])
        save_phase(tmp_path, "before_claim", name, empty)
        rows = []
        for index in range(count):
            rows.append(dict(case_id="claim", source_id="source", side="supported", phase="back_half",
                readout="observed_logp", numeric_ok=True, early=index, late=index + 1,
                site="message", restoration_gain=.1, layer=index, head=0,
                selection="paired_gradient", support=.1))
        save_phase(tmp_path, "back_half", name, pd.DataFrame(rows))
        assert str(collect(tmp_path, name).numeric_ok.dtype) == "boolean"
    summary = report_pairs(tmp_path, repeats=10)
    assert summary["failed_control_rows"] == dict(effects=0, interactions=0, adaptation=0, persistence=0)
    assert summary["control_rows"]["adaptation"] == dict(rows=32, passed=32, failed=0, missing=0)
    assert summary["control_rows"]["persistence"] == dict(rows=40, passed=40, failed=0, missing=0)
    assert (tmp_path / "paired_review.tar.gz").is_file()
