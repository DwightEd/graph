import json

import numpy as np
import pytest

from main import main
from onset_analysis.analysis import OnsetChoiceAudit, OnsetChoiceConfig

GROUPS = np.array(
    ["special", "evidence", "other_prompt", "history_far", "history_local", "self"]
)


def write_onset_fixture(root, *, contaminate_first_control: bool = False) -> None:
    samples = []
    for number in range(4):
        sample_id = f"sample-{number}"
        relative = f"train/QA/{sample_id}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        response_tokens = 5
        response_start = 5
        query_count = response_tokens + 1
        mass = np.zeros((1, 2, query_count, len(GROUPS)), dtype=np.float32)
        mass[..., 1] = 0.1
        mass[..., 2] = 0.05
        mass[..., 3] = 0.05
        mass[..., 4] = 0.5
        mass[..., 5] = 0.3
        evidence_gain = 0.2 + 0.1 * number
        mass[:, :, 3, 1] += evidence_gain
        mass[:, :, 3, 4] -= evidence_gain
        entropy = np.full((1, 2, query_count), 0.6, dtype=np.float32)
        entropy[:, :, 3] = 0.4
        top1 = np.full((1, 2, query_count), 0.4, dtype=np.float32)
        top1[:, :, 3] = 0.6
        instability = 0.1 + 0.1 * number
        observed_margin = np.array([1.0, 1.0, 1.0, -instability, 1.0, np.nan])
        predictor_entropy = np.array([1.0, 1.0, 1.0, 1.5 + number, 1.0, np.nan])
        response_text = [" Normal", " answer", ",", " Wrong", "."]
        if number == 0 and contaminate_first_control:
            response_text = [" Normal", ",", ",", " Wrong", " After"]
        np.savez_compressed(
            path,
            audit_schema=np.array(3),
            labels_used_for_capture=np.array(False),
            sample_id=np.array(sample_id),
            source_id=np.array(f"source-{number}"),
            task_type=np.array("QA"),
            token_ids=np.arange(response_start + response_tokens),
            token_text=np.array(["prompt"] * response_start + response_text),
            response_start=np.array(response_start),
            special_mask=np.zeros(response_start + response_tokens, dtype=bool),
            group_names=GROUPS,
            mass=mass,
            ordinary_mass=mass[..., 1:].sum(axis=-1),
            entropy_normalized=entropy,
            top1=top1,
            observed_margin=observed_margin,
            predictor_entropy=predictor_entropy,
        )
        np.savez_compressed(
            path.with_suffix(".labels.npz"),
            labels=np.array([0, 0, 1, 1, 0], dtype=np.int8),
        )
        samples.append(
            {
                "split": "train",
                "task_type": "QA",
                "sample_id": sample_id,
                "source_id": f"source-{number}",
                "path": relative,
                "response_tokens": response_tokens,
                "response_start": response_start,
                "resumed": True,
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


def test_onset_audit_skips_punctuation_and_tests_instability_with_lookback(
    tmp_path,
) -> None:
    audit_root = tmp_path / "audit"
    output = tmp_path / "onsets"
    audit_root.mkdir()
    write_onset_fixture(audit_root)

    report = OnsetChoiceAudit(
        OnsetChoiceConfig(
            audit_root,
            output,
            pre_window=3,
            match_window=8,
            bootstrap=20,
            seed=3,
        ),
        progress=False,
    ).run()

    records = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    first = records[0]
    assert report["hallucination_spans"] == 4
    assert report["content_onsets"] == 4
    assert report["matched_pairs"] == 4
    assert report["unmatched_onsets"] == 0
    assert first["onset"]["response_index"] == 3
    assert first["onset"]["token_text"].strip() == "Wrong"
    assert first["onset"]["lookback_mode"] == "focused_evidence"
    assert first["control"]["response_index"] == 1
    assert report["classification"]["instability"]["auroc"] == 1.0
    assert report["classification"]["remote_gain"]["auroc"] == 1.0
    assert report["joint"]["onset_instability_vs_remote_gain"]["estimate"] == 1.0


def test_onset_audit_command_runs_the_same_workflow(tmp_path, capsys) -> None:
    audit_root = tmp_path / "audit"
    output = tmp_path / "onsets"
    audit_root.mkdir()
    write_onset_fixture(audit_root)

    main(
        [
            "onset-audit",
            "--audit",
            str(audit_root),
            "--output",
            str(output),
            "--pre-window",
            "3",
            "--match-window",
            "8",
            "--bootstrap",
            "0",
        ]
    )

    assert json.loads(capsys.readouterr().out)["matched_pairs"] == 4
    assert (output / "events.jsonl").is_file()


def test_normal_control_window_cannot_include_hallucinated_tokens(tmp_path) -> None:
    audit_root = tmp_path / "audit"
    output = tmp_path / "onsets"
    audit_root.mkdir()
    write_onset_fixture(audit_root, contaminate_first_control=True)

    report = OnsetChoiceAudit(
        OnsetChoiceConfig(
            audit_root, output, pre_window=3, match_window=8, bootstrap=0
        ),
        progress=False,
    ).run()

    assert report["matched_pairs"] == 3
    assert report["unmatched_onsets"] == 1
    events = (output / "events.jsonl").read_text()
    assert "sample-0" not in events


@pytest.mark.parametrize("early_error", [False, True])
def test_first_error_is_not_replaced_by_a_later_measurable_onset(tmp_path, early_error):
    audit = tmp_path / "audit"
    audit.mkdir()
    write_onset_fixture(audit)
    if early_error:
        index = json.loads((audit / "index.json").read_text())
        for entry in index["samples"]:
            path = audit / entry["path"]
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            # Two extra normal tokens leave an uncontaminated control before the later error.
            queries = [0, 1, 1, 1, 2, 3, 4, 5]
            for name in ("mass", "ordinary_mass", "entropy_normalized", "top1"):
                arrays[name] = arrays[name][:, :, queries]
            for name in ("observed_margin", "predictor_entropy"):
                arrays[name] = arrays[name][queries]
            for name in ("token_ids", "token_text", "special_mask"):
                arrays[name] = np.concatenate(
                    (
                        arrays[name][:6],
                        np.repeat(arrays[name][6:7], 2),
                        arrays[name][6:],
                    )
                )
            entry["response_tokens"] = 7
            np.savez_compressed(path, **arrays)
            np.savez_compressed(
                path.with_suffix(".labels.npz"),
                labels=np.array([1, 0, 0, 0, 1, 1, 0], dtype=np.int8),
            )
        (audit / "index.json").write_text(json.dumps(index))
    output = tmp_path / "onsets"
    main(
        [
            "onset-audit",
            "--audit",
            str(audit),
            "--output",
            str(output),
            "--pre-window",
            "1",
            "--bootstrap",
            "0",
        ]
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["cohorts"]["first_error"]["matched_pairs"] == (
        0 if early_error else 4
    )
    assert report["cohorts"]["later_onsets"]["matched_pairs"] == (
        4 if early_error else 0
    )
    assert report["coverage"]["first_error_spans"] == 4
    assert report["coverage"]["first_errors_without_prior_predictor"] == (
        4 if early_error else 0
    )
    assert report["span_structure"]["hallucination_tokens"] == (
        12 if early_error else 8
    )
    assert report["span_structure"]["continuation_tokens"] == 4
    assert report["interpretation"]["margin"] == "observer_token_compatibility"


def test_joint_analysis_retains_differences_between_events_from_one_source(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    write_onset_fixture(audit)
    index = json.loads((audit / "index.json").read_text())
    for entry in index["samples"]:
        entry["source_id"] = "shared-source"
        path = audit / entry["path"]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["source_id"] = np.array("shared-source")
        np.savez_compressed(path, **arrays)
    (audit / "index.json").write_text(json.dumps(index))
    report = OnsetChoiceAudit(
        OnsetChoiceConfig(audit, tmp_path / "onsets", bootstrap=0), progress=False
    ).run()
    joint = report["joint"]["onset_instability_vs_remote_gain"]
    assert joint["estimate"] == pytest.approx(1.0)
    assert joint["events"] == 4
    assert joint["sources"] == 1
    assert joint["confidence_interval"] is None


def test_normal_control_also_requires_a_normal_baseline_predictor(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    write_onset_fixture(audit)
    for path in audit.glob("train/QA/*.labels.npz"):
        np.savez_compressed(path, labels=np.array([1, 0, 1, 1, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="matched normal control"):
        OnsetChoiceAudit(
            OnsetChoiceConfig(audit, tmp_path / "onsets", pre_window=1, bootstrap=0),
            progress=False,
        ).run()
