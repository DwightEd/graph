"""Label-free reference fitting and structure-score freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from attention_lifecycle import loaded_attention
from experiments.attention_phenomenology.reference import token_buckets
from experiments.attention_phenomenology.routing import (
    build_routing_state,
    collect_routing_edges,
)
from research_dataset import open_research_dataset

from .artifacts import save_npz, write_json
from .config import AuditConfig
from .features import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    LINEAGE_FEATURE_NAMES,
    LINEAGE_RELATION_NAMES,
    RELATION_NAMES,
    build_layer_features,
    relation_scores,
)
from .lineage import LineageOperator
from .nulls import constrained_endpoint_swap
from .reference import (
    ReferenceAccumulator,
    load_reference,
    save_reference,
    standardize,
)


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def _selected_ids(dataset, limit: int | None) -> list[str]:
    sample_ids = [str(sample_id) for sample_id in dataset.sample_ids]
    return sample_ids if limit is None else sample_ids[:limit]


def _real_analysis(sample, config: AuditConfig):
    edges = collect_routing_edges(sample, config=config)
    routing = build_routing_state(edges)
    operator = LineageOperator(routing)
    features = build_layer_features(
        routing,
        operator.run(),
        recent_tokens=config.recent_tokens,
    )
    return edges, routing, operator, features


def _fit_sample(sample, config: AuditConfig) -> tuple[str, np.ndarray]:
    """Return only compact CPU features so graph tensors die at this boundary."""

    with loaded_attention(sample):
        task = str(sample.task_type or "unknown")
        analysis = _real_analysis(sample, config)
        values = analysis[-1].cpu().numpy().astype(np.float32)
    return task, values


def _null_relations(
    operator,
    routing,
    source,
    *,
    task,
    buckets,
    reference,
    config,
) -> np.ndarray:
    features = build_layer_features(
        routing,
        operator.run(source=source),
        recent_tokens=config.recent_tokens,
    ).cpu().numpy()
    standardized = standardize(
        features,
        task=task,
        buckets=buckets,
        reference=reference,
        maximum=config.maximum_standardized_value,
    )
    return relation_scores(standardized)


class StructureAudit:
    """Fit and freeze non-neural structure coordinates through two methods."""

    def __init__(self, config: AuditConfig | None = None):
        self.config = AuditConfig() if config is None else config

    def fit(self, *, train_split, output, device="cpu", limit=None) -> None:
        config = self.config
        dataset = open_research_dataset(train_split, device=device)
        settings_json = json.dumps(config.reference_settings(), sort_keys=True)
        accumulator = ReferenceAccumulator(
            capacity=config.reference_capacity,
            seed=config.random_seed,
            minimum_scale=config.reference_minimum_scale,
            settings_json=settings_json,
        )

        sample_ids = _selected_ids(dataset, limit)
        for sample_id in tqdm(
            sample_ids, desc="fit structure reference", disable=not config.show_progress
        ):
            task, values = _fit_sample(dataset[sample_id], config)
            buckets = token_buckets(len(values), config.causal_position_bins)
            for token, bucket in enumerate(buckets):
                accumulator.add(task, int(bucket), values[token])

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_reference(output, accumulator.finish())

    def score(
        self,
        *,
        split_root,
        reference_path,
        output_dir,
        device="cpu",
        limit=None,
    ) -> None:
        config = self.config
        reference = load_reference(reference_path)
        expected = json.dumps(config.reference_settings(), sort_keys=True)
        if reference.settings_json != expected:
            raise ValueError("score settings differ from the fitted reference")

        dataset = open_research_dataset(split_root, device=device)
        output_dir = Path(output_dir)
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        sample_ids = _selected_ids(dataset, limit)

        for sample_id in tqdm(
            sample_ids, desc="freeze structure scores", disable=not config.show_progress
        ):
            arrays, row = self._score_sample(dataset[sample_id], reference)
            score_path = sample_dir / f"{sample_id}.npz"
            save_npz(score_path, **arrays)
            row["score_path"] = str(score_path.relative_to(output_dir))
            manifest_rows.append(row)

        write_json(
            output_dir / "manifest.json",
            {
                "schema": "non-neural-structure-manifest-v1",
                "labels_read": False,
                "trace_alignment": "post_token_query_at_same_position",
                "evaluation_alignment": "query_t_to_response_token_t_plus_1",
                "teacher_forced_trace": True,
                "claim_scope": (
                    "prompt-connected versus response-base attention-routing proxy; "
                    "not evidence grounding or model computation ancestry"
                ),
                "split_root": str(Path(split_root).resolve()),
                "reference_path": str(Path(reference_path).resolve()),
                "config": config.to_dict(),
                "feature_names": list(FEATURE_NAMES),
                "relation_names": list(RELATION_NAMES),
                "response_endpoint_null_relations": list(LINEAGE_RELATION_NAMES),
                "samples": manifest_rows,
            },
        )

    def _score_sample(self, sample, reference):
        config = self.config
        with loaded_attention(sample) as attention:
            task = str(sample.task_type or "unknown")
            source_id = str(sample.source_id)
            response_tokens = attention.token_ids[attention.response_idx :].cpu().numpy()
            edges, routing, operator, feature_tensor = _real_analysis(sample, config)
            features = feature_tensor.cpu().numpy().astype(np.float32)
            buckets = token_buckets(len(features), config.causal_position_bins)
            standardized = standardize(
                features,
                task=task,
                buckets=buckets,
                reference=reference,
                maximum=config.maximum_standardized_value,
            )
            real_relations = relation_scores(standardized)

            null_scores = []
            changed_fraction = []
            null_audits = []
            base_seed = _sample_seed(sample.sample_id, config.random_seed)
            for replicate in range(config.null_replicates):
                null = constrained_endpoint_swap(
                    edges,
                    seed=base_seed + replicate,
                    attempts_per_edge=config.swap_attempts_per_edge,
                    lag_bins=config.response_lag_bins,
                )
                null_scores.append(
                    _null_relations(
                        operator,
                        routing,
                        null.edges.source,
                        task=task,
                        buckets=buckets,
                        reference=reference,
                        config=config,
                    )
                )
                changed_fraction.append(null.changed_fraction)
                null_audits.append(null.audit)

            final_relations = relation_scores(standardized[:, -1:, :])
            rng = np.random.default_rng(base_seed)
            shuffle_orders = []
            shuffle_scores = []
            for _ in range(config.layer_shuffle_replicates):
                order = tuple(rng.permutation(edges.num_layers).tolist())
                shuffle_orders.append(order)
                shuffled_features = build_layer_features(
                    routing,
                    operator.run(layer_order=order),
                    recent_tokens=config.recent_tokens,
                ).cpu().numpy()
                final_control = features.copy()
                for name in LINEAGE_FEATURE_NAMES:
                    index = FEATURE_INDEX[name]
                    final_control[:, -1, index] = shuffled_features[:, -1, index]
                shuffled_standardized = standardize(
                    final_control,
                    task=task,
                    buckets=buckets,
                    reference=reference,
                    maximum=config.maximum_standardized_value,
                )
                shuffle_scores.append(
                    relation_scores(shuffled_standardized[:, -1:, :])
                )

        arrays = {
            "schema": np.asarray("non-neural-structure-score-v1"),
            "sample_id": np.asarray(str(sample.sample_id)),
            "source_id": np.asarray(source_id),
            "task_type": np.asarray(task),
            "response_token_ids": response_tokens.astype(np.int32),
            "token_index": np.arange(len(features), dtype=np.int32),
            "causal_position_bucket": buckets,
            "feature_names": np.asarray(FEATURE_NAMES, dtype=str),
            "relation_names": np.asarray(RELATION_NAMES, dtype=str),
            "layer_features": features,
            "standardized_features": standardized,
            "relation_scores": real_relations,
            "final_relation_scores": final_relations,
            "response_endpoint_null_relation_scores": np.stack(null_scores).astype(
                np.float32
            ),
            "response_endpoint_null_changed_fraction": np.asarray(
                changed_fraction, dtype=np.float32
            ),
            "layer_shuffle_order": np.asarray(shuffle_orders, dtype=np.int16),
            "layer_shuffle_relation_scores": np.stack(shuffle_scores).astype(
                np.float32
            ),
        }
        row = {
            "sample_id": str(sample.sample_id),
            "source_id": source_id,
            "task_type": task,
            "response_length": len(features),
            "null_audit": {
                "causal_violations": max(
                    audit["causal_violations"] for audit in null_audits
                ),
                "duplicate_edges": max(audit["duplicate_edges"] for audit in null_audits),
                "changed_fraction_mean": float(np.mean(changed_fraction)),
            },
        }
        return arrays, row
