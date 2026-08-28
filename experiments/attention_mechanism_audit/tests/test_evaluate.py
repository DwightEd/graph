import numpy as np

from experiments.attention_mechanism_audit.audit import (
    AuditArtifact,
    AuditRow,
    RawMargins,
)
from experiments.attention_mechanism_audit.evaluate import (
    evaluate,
    source_mean_bootstrap,
)


def test_source_bootstrap_weights_sources_not_rows():
    summary = source_mean_bootstrap(
        np.asarray([1.0, 3.0, 10.0]),
        np.asarray(["a", "a", "b"]),
        replicates=101,
        seed=7,
    )

    assert summary["available"] is True
    assert summary["samples"] == 3
    assert summary["sources"] == 2
    assert summary["source_equal_mean"] == 6.0
    assert summary["ci_low"] <= 6.0 <= summary["ci_high"]


def test_evaluation_reports_only_the_three_fixed_mechanisms():
    artifact = AuditArtifact.from_rows(
        [
            AuditRow(
                "a1",
                "a",
                10,
                11,
                True,
                RawMargins(-1.0, -2.0, -1.0, -2.0, 0.0, 0.0, 0.0),
            ),
            AuditRow(
                "a2",
                "a",
                10,
                11,
                False,
                RawMargins(-1.0, -2.0, 1.0, 2.0, 1.0, 0.0, 0.0),
            ),
            AuditRow(
                "b1",
                "b",
                20,
                21,
                True,
                RawMargins(-2.0, -3.0, 2.0, 1.0, 2.0, 1.0, 1.0),
            ),
        ]
    )

    report = evaluate(artifact, bootstrap_replicates=101, seed=11)

    assert report["labels_used"] is False
    assert set(report["mechanisms"]) == {"select", "relay", "override"}
    assert "auroc" not in str(report).lower()
    assert "probe" not in str(report).lower()
    assert (
        report["mechanisms"]["select"]["success_rate"]["source_equal_mean"]
        == 0.75
    )
    relay = report["mechanisms"]["relay"]
    assert relay["self_lock_rate"]["source_equal_mean"] == 0.5
    override = report["mechanisms"]["override"]
    assert override["eligible_samples"] == 2
    assert override["eligible_sources"] == 2
    assert override["capture_failure_rate"]["source_equal_mean"] == 0.5


def test_relay_and_override_are_unavailable_without_select_success():
    artifact = AuditArtifact.from_rows(
        [
            AuditRow(
                "sample",
                "source",
                10,
                11,
                True,
                RawMargins(-1.0, -1.0, -0.5, 0.0, 0.0, 0.0, -0.5),
            )
        ]
    )

    mechanisms = evaluate(artifact, bootstrap_replicates=11)["mechanisms"]
    relay = mechanisms["relay"]
    override = mechanisms["override"]

    assert relay["eligible_samples"] == 0
    assert relay["history_prior_support"]["available"] is False
    assert relay["history_evidence_relay"]["available"] is False
    assert relay["self_lock_rate"]["available"] is False
    assert override["eligible_samples"] == 0
    assert override["question_prior_strength"]["available"] is False
    assert override["prior_capture"]["available"] is False
    assert override["capture_failure_rate"]["available"] is False
