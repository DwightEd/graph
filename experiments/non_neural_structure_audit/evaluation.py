"""Label-aware audit orchestration run only after structure scores are frozen."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiment_protocol import file_sha256
from research_dataset import open_research_dataset

from .artifacts import read_json, write_csv, write_json
from .config import EvaluationConfig
from .decisions import gate_decisions
from .evaluation_data import load_frozen_samples
from .joint_form import evaluate_joint_forms
from .protocol import (
    load_confirmation_plan,
    load_split_plan,
    method_sha256,
    tokenizer_sha256,
    validate_score_binding,
)
from .relation_audit import relation_rows
from .temporal_audit import (
    align_query_to_next_token,
    pre_onset_slope,
    temporal_rows,
)

__all__ = [
    "StructureEvaluator",
    "align_query_to_next_token",
    "pre_onset_slope",
]


class StructureEvaluator:
    """Open labels after score freezing and produce model-authorization gates."""

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = EvaluationConfig() if config is None else config

    def run(
        self,
        *,
        split_root,
        score_dir,
        output_dir,
        tokenizer_path=None,
        split_plan=None,
        confirmation_plan=None,
    ) -> None:
        score_dir = Path(score_dir)
        manifest_path = score_dir / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest["labels_read"] is not False:
            raise ValueError("evaluation requires label-free frozen scores")
        if manifest["trace_alignment"] != "post_token_query_at_same_position":
            raise ValueError("unexpected cached trace alignment")
        if manifest["evaluation_alignment"] != "query_t_to_response_token_t_plus_1":
            raise ValueError("unexpected score/label alignment")
        if self.config.scope != "smoke" and tokenizer_path is None:
            raise ValueError("discovery and confirmation require a local tokenizer")

        split_plan_sha256 = None
        confirmation_plan_sha256 = None
        if self.config.scope == "smoke":
            selected_sample_ids = [row["sample_id"] for row in manifest["samples"]]
        elif self.config.scope == "discovery":
            if split_plan is None:
                raise ValueError("discovery requires a frozen split plan")
            split = load_split_plan(split_plan, score_dir=score_dir)
            selected_sample_ids = split["discovery_sample_ids"]
            split_plan_sha256 = file_sha256(split_plan)
        else:
            if confirmation_plan is None:
                raise ValueError("confirmation requires a frozen confirmation plan")
            confirmation = load_confirmation_plan(
                confirmation_plan,
                score_dir=score_dir,
                tokenizer_path=tokenizer_path,
                config=self.config,
            )
            selected_sample_ids = confirmation["confirmation_sample_ids"]
            confirmation_plan_sha256 = file_sha256(confirmation_plan)
            split_plan_sha256 = confirmation["split_plan_sha256"]

        dataset = open_research_dataset(split_root, device="cpu")
        validate_score_binding(
            manifest=manifest,
            score_dir=score_dir,
            dataset=dataset,
            selected_sample_ids=selected_sample_ids,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = load_frozen_samples(
            score_dir=score_dir,
            manifest=manifest,
            scratch_dir=output_dir / ".scratch",
            selected_sample_ids=selected_sample_ids,
            dataset=dataset,
            tokenizer_path=tokenizer_path,
        )
        with bundle:
            relations = relation_rows(
                bundle,
                set(manifest["response_endpoint_null_relations"]),
                self.config,
            )
            samples = bundle.samples
            temporal = temporal_rows(samples, self.config)
            joint = (
                []
                if self.config.scope == "confirmation"
                else evaluate_joint_forms(samples, self.config)
            )
            positive_responses = sum(
                bool(sample.labels[1:][sample.eligible[1:]].any()) for sample in samples
            )
            a0_components = {
                "artifact_binding_verified": True,
                "gold_alignment_verified": False,
                "pipeline_label_permutation_verified": False,
            }
            decisions = gate_decisions(
                relations,
                scope=self.config.scope,
                samples=len(samples),
                source_groups=len({sample.source_id for sample in samples}),
                positive_responses=positive_responses,
                artifact_binding_verified=a0_components["artifact_binding_verified"],
                full_a0_verified=all(a0_components.values()),
                config=self.config,
            )
            if self.config.scope == "smoke":
                scientific_status = "ENGINEERING_SMOKE_ONLY"
            elif decisions[0]["status"] != "PASS":
                scientific_status = "BLOCKED_BY_A0"
            elif self.config.scope == "discovery":
                scientific_status = "EXPLORATORY_DISCOVERY"
            else:
                scientific_status = "FORMAL_CONFIRMATION"
            for rows in (relations, temporal, joint):
                for row in rows:
                    row["scope"] = self.config.scope
                    row["scientific_status"] = scientific_status

            write_csv(output_dir / "relation_metrics.csv", relations)
            write_csv(output_dir / "temporal_audit.csv", temporal)
            write_csv(output_dir / "joint_form_cv.csv", joint)
            write_csv(output_dir / "decision_table.csv", decisions)
            labels = np.concatenate(
                [sample.labels[1:][sample.eligible[1:]] for sample in samples]
            )
            write_json(
                output_dir / "evaluation.json",
                {
                    "schema": "non-neural-structure-evaluation-v1",
                    "labels_read": True,
                    "scope": self.config.scope,
                    "scientific_status": scientific_status,
                    "evaluation_config": asdict(self.config),
                    "method_sha256": method_sha256(),
                    "score_manifest_sha256": file_sha256(manifest_path),
                    "split_plan_sha256": split_plan_sha256,
                    "confirmation_plan_sha256": confirmation_plan_sha256,
                    "selected_sample_ids": [sample.sample_id for sample in samples],
                    "selected_source_ids": sorted(
                        {sample.source_id for sample in samples}
                    ),
                    "token_scope": "content_alphanumeric"
                    if tokenizer_path is not None
                    else "all_tokens",
                    "tokenizer_path": str(Path(tokenizer_path).resolve())
                    if tokenizer_path is not None
                    else None,
                    "tokenizer_sha256": tokenizer_sha256(tokenizer_path)
                    if tokenizer_path is not None
                    else None,
                    "samples": len(samples),
                    "source_groups": len({sample.source_id for sample in samples}),
                    "positive_responses": positive_responses,
                    "evaluated_tokens": len(labels),
                    "positive_tokens": int(labels.sum()),
                    "prevalence": float(labels.mean()),
                    "a0_components": a0_components,
                    "relation_metrics": relations,
                    "temporal_audit": temporal,
                    "joint_form_cv": joint,
                    "decisions": decisions,
                    "claim_scope": manifest["claim_scope"],
                    "limitations": [
                        "all prompt tokens share one type; evidence/question/system are not separated",
                        "raw attention omits values, output projection, residual stream, and FFN",
                        "teacher-forced cached rows are routing proxies, not autoregressive interventions",
                        "A10 requires new base-model runs and is not inferred from cached attention",
                    ],
                },
            )
