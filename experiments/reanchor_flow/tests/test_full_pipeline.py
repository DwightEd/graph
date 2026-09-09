"""Real tiny-Llama capture, post-hoc evaluation, and resumable refinement.

Only the external dataset adapter and text-to-unit alignment are fixtures.
Model forwards, event selection, gradients, artifacts, validation, and reports
all use production implementations. Synthetic annotations have no factual
meaning and this test cannot establish model-size performance or detection.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("matplotlib")

from experiments.reanchor_flow import dataset as dataset_module, subset, subset_report
from experiments.reanchor_flow.cohort_plot import GROUPS
from experiments.reanchor_flow.route_plan import RouteBudget
from experiments.reanchor_flow.sample_scan import render_sample_scan
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model


class FixtureDataset:
    def __init__(self, annotations, token_ids, metadata, access, *, allow_labels):
        self.spec = {}
        self.manifest = {"split": "test"}
        self.sample_ids = tuple(annotations)
        self.annotations = annotations
        self.token_ids = token_ids
        self.records = metadata
        self.access = access
        self.allow_labels = allow_labels

    def metadata(self, sample_id):
        return self.records[sample_id]

    def __getitem__(self, sample_id):
        return SimpleNamespace(
            sample_id=sample_id,
            attention=lambda: SimpleNamespace(
                token_ids=self.token_ids[sample_id], response_idx=4
            ),
            release_attention=lambda: None,
        )

    def prepare_evaluation_labels(self, sample_ids):
        assert self.allow_labels, "capture attempted to open annotations"
        self.access.append(tuple(sample_ids))
        return SimpleNamespace(
            response_labels=lambda sample: torch.tensor(
                self.annotations[sample.sample_id], dtype=torch.long
            )
        )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_scan_labels_resume_and_functional_refinement(tmp_path, monkeypatch):
    annotations = {
        "mixed": [0, 1, 0],
        "clean": [0, 0, 0],
        "hallucinated": [1, 1, 1],
    }
    token_ids = {
        name: torch.tensor([1, 2, 3, 4, *response])
        for name, response in zip(
            annotations, ([5, 6, 7], [8, 9, 10], [12, 13, 14]), strict=True
        )
    }
    metadata = {
        name: {
            "source_id": f"source-{name}",
            "task_type": "Summary" if name == "hallucinated" else "QA",
            "generator_model": "synthetic-tiny-llama",
        }
        for name in annotations
    }
    label_access = []

    def open_fixture(_root, *, retain_embedded_labels, **_kwargs):
        return FixtureDataset(
            annotations,
            token_ids,
            metadata,
            label_access,
            allow_labels=retain_embedded_labels,
        )

    monkeypatch.setattr(dataset_module, "open_research_dataset", open_fixture)
    monkeypatch.setattr(
        dataset_module, "build_source_units", lambda *_args: paired_world().units
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    source_info = tmp_path / "source_info.jsonl"
    source_info.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata.values()), encoding="utf-8"
    )
    output = tmp_path / "output"
    tokenizer = SimpleNamespace(
        name_or_path="tiny-llama",
        convert_ids_to_tokens=lambda ids: [f"synthetic_{token}" for token in ids],
    )
    model = tiny_model()
    config = subset.SubsetRunConfig(
        model_id="tiny-llama",
        model_dtype="float32",
        tokenizer_id="tiny-llama",
        cohort=subset.CohortPlan(("QA", "Summary"), 0, (), 2026),
        targets=subset.TargetPlan(3, "reanchor-window", None),
        mechanism=subset.MechanismPlan(
            signal=subset.FlowSignal.MESSAGE,
            carrier_scope="response",
            coverage=0.9,
            query_chunk=2,
            route_budget=RouteBudget(max_rows=2),
            local_window=1,
            saved_edges=64,
        ),
        scan_only=True,
    )
    corpus = dataset_module.RagTruthAuditCorpus.open(
        cache,
        source_info,
        split="test",
        model_id=config.model_id,
        tokenizer=tokenizer,
    )

    counts = subset.SubsetAuditRunner(model, tokenizer, corpus, output, config).run()
    assert counts == {"samples": 3, "targets": 0, "resumed": 0, "confirmed": 0}
    assert not label_access
    manifest = _read_json(output / "run_manifest.json")
    assert manifest["analysis_complete"]
    assert manifest["analysis_scope"] == "structure_only"
    assert not manifest["audits"]
    assert set(manifest["samples"]) == set(annotations)
    saved_files = []
    for name, entry in manifest["samples"].items():
        scan_path = output / entry["scan"]
        saved_files.extend((scan_path, output / entry["world"]))
        with np.load(scan_path, allow_pickle=False) as scan:
            assert scan["route_row_position"].tolist() == [3, 4, 5]
            assert scan["reanchor_bucket_transport"].shape == (3, 4, 3, 4)
            assert scan["reanchor_score"].shape == (3, 4, 3)
            assert not bool(scan["labels_used_for_capture"])
            np.testing.assert_array_equal(scan["token_ids"], token_ids[name].numpy())

    scanned_report = subset_report.evaluate_subset_split(cache, output, plot=True)
    assert scanned_report["analysis_scope"] == "structure_only"
    assert scanned_report["targets"] == []
    assert len(label_access) == 1
    assert set(label_access[0]) == set(annotations)
    cohort = _read_json(output / "cohort_summary.json")
    coverage = cohort["groups"]["ALL"]["coverage"]
    assert coverage["scanned_samples"] == 3
    assert coverage["scanned_response_tokens"] == 9
    assert coverage["scanned_nonhallucinated_tokens"] == 5
    assert coverage["scanned_hallucinated_tokens"] == 4
    assert coverage["mixed_scanned_samples"] == 1
    assert coverage["response_token_coverage"] == 1
    evidence = cohort["groups"]["QA"]["structure"]["metrics"]["prompt_evidence"]
    paired = evidence["within_sample_hallucinated_minus_nonhallucinated"]
    mixed_scan = output / manifest["samples"]["mixed"]["scan"]
    with np.load(mixed_scan, allow_pickle=False) as scan:
        transport = scan["reanchor_bucket_transport"]
        fractions = transport[..., 0] / transport.sum(axis=-1)
        # q=4 predicts response token 1, the sole positive token in this sample.
        expected = fractions[:, :, 1] - fractions[:, :, [0, 2]].mean(axis=-1)
    np.testing.assert_allclose(paired["mean"], expected, atol=1e-7)
    np.testing.assert_array_equal(paired["sample_count"], np.ones((3, 4)))
    for plot in scanned_report["cohort_plots"]:
        assert (output / plot).stat().st_size > 10_000
    timeline = render_sample_scan(mixed_scan, output / "synthetic_mixed.png", tokenizer)
    assert timeline.stat().st_size > 10_000

    mtimes = {path: path.stat().st_mtime_ns for path in saved_files}
    assert (
        subset.SubsetAuditRunner(model, tokenizer, corpus, output, config).run()[
            "targets"
        ]
        == 0
    )
    assert len(label_access) == 1
    assert mtimes == {path: path.stat().st_mtime_ns for path in saved_files}

    refined = subset.SubsetAuditRunner(
        model,
        tokenizer,
        corpus,
        output,
        replace(config, scan_only=False),
    ).run()
    assert refined["samples"] == 3
    assert refined["targets"] == 9
    assert refined["resumed"] == 0
    assert len(label_access) == 1
    assert mtimes == {path: path.stat().st_mtime_ns for path in saved_files}
    final_report = subset_report.evaluate_subset_split(cache, output, plot=True)
    assert final_report["analysis_scope"] == "structure_and_selected_target_function"
    assert len(final_report["targets"]) == 9
    assert final_report["full_scan_coverage"]["ALL"] == coverage
    for row in final_report["targets"]:
        relative = row["query_position"] + 1 - 4
        assert row["hallucination_label"] == annotations[row["sample_id"]][relative]
    final_cohort = _read_json(output / "cohort_summary.json")
    functional = final_cohort["groups"]["ALL"]["functional"]
    assert functional["selected_targets"] == dict(zip(GROUPS, [5, 4], strict=True))
    assert functional["bucket_immediate_action"]["sample_count"] == 3
    competition = final_report["groups"]["ALL"]["raw_axis_evaluation"][
        "route_origin_competition"
    ]
    assert competition["evaluated_targets"] == 9
    assert competition["auroc"] is not None
    assert competition["auprc"] is not None

    manifest = _read_json(output / "run_manifest.json")
    artifacts = [output / entry["result"] for entry in manifest["audits"].values()]
    assert len(artifacts) == 9
    for artifact in artifacts:
        with np.load(artifact, allow_pickle=False) as stored:
            assert stored["route_head_transport"].shape[:2] == (3, 4)
            query = int(stored["query_position"])
            assert int(stored["prediction_position"]) == query + 1
            assert len(stored["route_row_position"]) <= 2
            assert int(stored["route_row_position"][-1]) == query
    artifact_mtimes = {path: path.stat().st_mtime_ns for path in artifacts}
    resumed = subset.SubsetAuditRunner(
        model,
        tokenizer,
        corpus,
        output,
        replace(config, scan_only=False),
    ).run()
    assert resumed["targets"] == resumed["resumed"] == 9
    assert artifact_mtimes == {path: path.stat().st_mtime_ns for path in artifacts}
    assert len(label_access) == 2
    (output / "SYNTHETIC_FIXTURE.txt").write_text(
        "Synthetic tiny random Llama: 3 layers, 4 heads, hidden size 32.\n"
        "Three invented samples and invented labels; no empirical hallucination claim.\n"
        "Production full scan, per-target VJP/root cut, resume, label join, cohort plots.\n"
        "This does not test Llama-3.1-8B GPU memory or detection performance.\n",
        encoding="utf-8",
    )
