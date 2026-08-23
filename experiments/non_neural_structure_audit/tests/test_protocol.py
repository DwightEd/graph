import json

from experiment_protocol import file_sha256
from experiments.non_neural_structure_audit.artifacts import write_json
from experiments.non_neural_structure_audit.config import EvaluationConfig
from experiments.non_neural_structure_audit.protocol import (
    freeze_confirmation,
    load_confirmation_plan,
    load_split_plan,
    method_sha256,
    prepare_split_plan,
    tokenizer_sha256,
)


def test_split_and_confirmation_plans_bind_disjoint_sources_and_config(tmp_path):
    score_dir = tmp_path / "scores"
    manifest_path = score_dir / "manifest.json"
    rows = [
        {"sample_id": f"sample-{index}", "source_id": f"source-{index}"}
        for index in range(4)
    ]
    write_json(
        manifest_path,
        {
            "schema": "non-neural-structure-manifest-v2",
            "labels_read": False,
            "dataset_manifest_sha256": "a" * 64,
            "reference_sha256": "b" * 64,
            "method_sha256": method_sha256(),
            "samples": rows,
        },
    )
    split_path = tmp_path / "split.json"
    split = prepare_split_plan(
        score_dir=score_dir,
        output=split_path,
        discovery_fraction=0.5,
        seed=7,
    )

    assert set(split["discovery_source_ids"]).isdisjoint(
        split["confirmation_source_ids"]
    )
    assert set(split["discovery_sample_ids"]) | set(
        split["confirmation_sample_ids"]
    ) == {row["sample_id"] for row in rows}

    discovery_path = tmp_path / "discovery.json"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    discovery_path.write_text(
        json.dumps(
            {
                "scope": "discovery",
                "selected_sample_ids": split["discovery_sample_ids"],
                "score_manifest_sha256": file_sha256(manifest_path),
                "method_sha256": method_sha256(),
                "tokenizer_sha256": tokenizer_sha256(tokenizer),
                "decisions": [{"audit": "A0", "status": "PASS"}],
            }
        ),
        encoding="utf-8",
    )
    config = EvaluationConfig(scope="confirmation", bootstrap_replicates=11)
    confirmation_path = tmp_path / "confirmation.json"
    freeze_confirmation(
        split_plan=split_path,
        discovery_evaluation=discovery_path,
        output=confirmation_path,
        tokenizer_path=tokenizer,
        config=config,
    )

    loaded = load_confirmation_plan(
        confirmation_path,
        score_dir=score_dir,
        tokenizer_path=tokenizer,
        config=config,
    )
    assert loaded["confirmation_sample_ids"] == split["confirmation_sample_ids"]

    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery["decisions"][0]["status"] = "INCONCLUSIVE_A0_CONTROLS_MISSING"
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="A0 is incomplete"):
        freeze_confirmation(
            split_plan=split_path,
            discovery_evaluation=discovery_path,
            output=tmp_path / "blocked-confirmation.json",
            tokenizer_path=tokenizer,
            config=config,
        )


def test_split_plan_rejects_sample_ids_that_do_not_match_their_frozen_sources(
    tmp_path,
):
    score_dir = tmp_path / "scores"
    manifest_path = score_dir / "manifest.json"
    rows = [
        {"sample_id": f"sample-{index}", "source_id": f"source-{index}"}
        for index in range(4)
    ]
    write_json(
        manifest_path,
        {
            "schema": "non-neural-structure-manifest-v2",
            "labels_read": False,
            "dataset_manifest_sha256": "a" * 64,
            "reference_sha256": "b" * 64,
            "method_sha256": method_sha256(),
            "samples": rows,
        },
    )
    split_path = tmp_path / "split.json"
    split = prepare_split_plan(score_dir=score_dir, output=split_path, seed=7)
    split["discovery_sample_ids"].append(split["confirmation_sample_ids"][0])
    write_json(split_path, split)

    import pytest

    with pytest.raises(ValueError, match="sample groups do not match"):
        load_split_plan(split_path, score_dir=score_dir)
