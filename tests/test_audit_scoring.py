import json

import numpy as np

from control_graph.audit import AttentionAuditScorer, AuditScoreConfig
from control_graph.audit_evaluation import (
    AttentionAuditEvaluator,
    AuditEvaluationConfig,
)

GROUPS = np.array(
    ["special", "evidence", "other_prompt", "history_far", "history_local", "self"]
)


def write_audit_fixture(root) -> None:
    samples = []
    for sample_number in range(4):
        sample_id = f"sample-{sample_number}"
        relative = f"train/QA/{sample_id}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        hallucinated = sample_number >= 2
        message_mass = np.zeros((1, 2, 4, len(GROUPS)), dtype=np.float32)
        message_mass[..., 1] = 0.8
        message_mass[..., 3] = 0.05
        message_mass[..., 4] = 0.05
        message_mass[..., 5] = 0.05
        if hallucinated:
            message_mass[:, :, 1:3, 1] = 0.1
            message_mass[:, :, 1:3, 3] = 0.3
            message_mass[:, :, 1:3, 4] = 0.3
            message_mass[:, :, 1:3, 5] = 0.25
        message_ordinary_mass = message_mass[..., 1:].sum(axis=-1)
        mass = message_mass.copy()
        ordinary_mass = message_ordinary_mass.copy()
        observed_margin = np.array([2.0, -1.0, -1.0, np.nan] if hallucinated else [2.0] * 3 + [np.nan])
        np.savez_compressed(
            path,
            audit_schema=np.array(3),
            labels_used_for_capture=np.array(False),
            sample_id=np.array(sample_id),
            source_id=np.array(f"source-{sample_number}"),
            task_type=np.array("QA"),
            token_ids=np.arange(8),
            response_start=np.array(5),
            special_mask=np.zeros(8, dtype=bool),
            group_names=GROUPS,
            message_mass=message_mass,
            message_ordinary_mass=message_ordinary_mass,
            mass=mass,
            ordinary_mass=ordinary_mass,
            head_margin=np.ones((1, 2, 4), dtype=np.float32),
            observed_margin=observed_margin,
        )
        samples.append(
            {
                "split": "train",
                "task_type": "QA",
                "sample_id": sample_id,
                "source_id": f"source-{sample_number}",
                "path": relative,
                "response_tokens": 3,
                "response_start": 5,
                "resumed": True,
            }
        )
    samples.append(
        {
            "split": "test",
            "task_type": "QA",
            "sample_id": "missing",
            "source_id": "missing-source",
            "path": "test/QA/missing.npz",
            "response_tokens": 3,
            "response_start": 5,
            "resumed": False,
        }
    )
    (root / "index.json").write_text(
        json.dumps(
            {
                "audit_schema": 3,
                "labels_used_for_capture": False,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def test_audit_scoring_uses_completed_traces_without_opening_labels(tmp_path) -> None:
    audit_root = tmp_path / "audit"
    output = tmp_path / "scores"
    audit_root.mkdir()
    write_audit_fixture(audit_root)
    (audit_root / "train/QA/sample-0.labels.npz").write_bytes(b"not an npz")

    summary = AttentionAuditScorer(
        AuditScoreConfig(audit_root, output, completed_only=True), progress=False
    ).run()

    records = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    assert summary == {
        "schema": "control-graph/audit-score-summary@1",
        "planned_samples": 5,
        "completed_samples": 4,
        "skipped_samples": 1,
        "available_response_tokens": 12,
        "special_target_tokens": 0,
        "unscorable_tokens": 0,
        "scored_tokens": 12,
        "sources": 4,
        "labels_used": False,
        "output": str(output / "scores.jsonl"),
    }
    assert len(records) == 12
    assert all("label" not in record for record in records)
    assert set(records[0]["mechanism_edges"]) == {
        "evidence_support",
        "other_prompt_support",
        "far_history_support",
        "local_history_support",
    }
    normal = next(record for record in records if record["sample_id"] == "sample-0")
    hallucination = next(
        record
        for record in records
        if record["sample_id"] == "sample-2" and record["response_index"] == 1
    )
    assert hallucination["scores"]["constraint_displacement"] > normal["scores"][
        "constraint_displacement"
    ]


def test_audit_evaluation_separates_onset_from_span_continuation(tmp_path) -> None:
    audit_root = tmp_path / "audit"
    score_output = tmp_path / "scores"
    audit_root.mkdir()
    write_audit_fixture(audit_root)
    for sample_number in range(4):
        labels = np.array([0, 1, 1] if sample_number >= 2 else [0, 0, 0], dtype=np.int8)
        np.savez_compressed(
            audit_root / f"train/QA/sample-{sample_number}.labels.npz", labels=labels
        )
    AttentionAuditScorer(
        AuditScoreConfig(audit_root, score_output, completed_only=True), progress=False
    ).run()

    report = AttentionAuditEvaluator(
        AuditEvaluationConfig(
            audit_root,
            score_output / "scores.jsonl",
            tmp_path / "evaluation.json",
            bootstrap=20,
            seed=7,
        ),
        progress=False,
    ).run()

    assert report["labels_used_stage"] == "evaluation_only"
    assert report["tokens_joined"] == 12
    assert set(report["cohorts"]) == {"all", "onset", "continuation"}
    for cohort in report["cohorts"].values():
        result = cohort["scores"]["constraint_displacement"]
        assert result["auroc"] == 1.0
        assert result["auprc"] == 1.0
    assert set(report["cohorts"]["all"]["scores"]) == {
        "constraint_displacement",
        "attention_displacement",
        "negative_margin",
        "relative_position",
    }
