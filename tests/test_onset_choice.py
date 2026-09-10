import json

import numpy as np

from control_graph.onset_choice import OnsetChoiceAudit, OnsetChoiceConfig
from main import main

GROUPS = np.array(
    ["special", "evidence", "other_prompt", "history_far", "history_local", "self"]
)


def write_onset_fixture(root) -> None:
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
        np.savez_compressed(
            path,
            audit_schema=np.array(3),
            labels_used_for_capture=np.array(False),
            sample_id=np.array(sample_id),
            source_id=np.array(f"source-{number}"),
            task_type=np.array("QA"),
            token_ids=np.arange(response_start + response_tokens),
            token_text=np.array(
                ["prompt"] * response_start + [" Normal", " answer", ",", " Wrong", "."]
            ),
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


def test_onset_audit_skips_punctuation_and_tests_instability_with_lookback(tmp_path) -> None:
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

    records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
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
