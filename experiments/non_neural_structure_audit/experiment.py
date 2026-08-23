"""Label-free reference fitting and structure-score freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from attention_lifecycle import loaded_attention
from experiment_protocol import (
    FrozenFile,
    HeldOutSourceAudit,
    canonical_source_group,
    file_sha256,
)
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
    LAYER_ORDER_FEATURE_NAMES,
    LAYER_ORDER_RELATION_NAMES,
    LINEAGE_RELATION_NAMES,
    RELATION_NAMES,
    build_layer_features,
    relation_scores,
)
from .lineage import LineageOperator
from .nulls import EndpointSwapPlan
from .protocol import method_sha256
from .reference import (
    ReferenceAccumulator,
    load_reference,
    save_reference,
    standardize,
)


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def _selected_ids(dataset, limit: int | None, task_type: str) -> list[str]:
    sample_ids = [
        str(sample_id)
        for sample_id in dataset.sample_ids
        if task_type.casefold() == "all"
        or str(dataset[sample_id].task_type).casefold() == task_type.casefold()
    ]
    return sample_ids if limit is None else sample_ids[:limit]


def _non_identity_permutation(rng: np.random.Generator, size: int) -> tuple[int, ...]:
    identity = np.arange(size)
    order = rng.permutation(size)
    while np.array_equal(order, identity):
        order = rng.permutation(size)
    return tuple(order.tolist())


def _analyze_edges(edges, config: AuditConfig):
    routing = build_routing_state(edges)
    operator = LineageOperator(routing)
    features = build_layer_features(
        routing,
        operator.run(),
        recent_tokens=config.recent_tokens,
    )
    return routing, operator, features


def _fit_sample(sample, config: AuditConfig) -> tuple[str, str, np.ndarray]:
    """Return only compact CPU features so graph tensors die at this boundary."""

    with loaded_attention(sample):
        task = str(sample.task_type or "unknown")
        source_id = canonical_source_group(sample)
        edges = collect_routing_edges(sample, config=config)
    feature_tensor = _analyze_edges(edges, config)[-1]
    values = feature_tensor.cpu().numpy().astype(np.float32)
    return task, source_id, values


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
    features = (
        build_layer_features(
            routing,
            operator.run(source=source),
            recent_tokens=config.recent_tokens,
        )
        .cpu()
        .numpy()
    )
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

    def fit(
        self, *, train_split, output, device="cpu", limit=None, task_type="QA"
    ) -> None:
        config = self.config
        dataset = open_research_dataset(train_split, device=device)
        train_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")
        settings_json = json.dumps(
            {**config.reference_settings(), "task_type": task_type}, sort_keys=True
        )
        accumulator = ReferenceAccumulator(
            capacity=config.reference_capacity,
            seed=config.random_seed,
            minimum_scale=config.reference_minimum_scale,
            settings_json=settings_json,
        )

        sample_ids = _selected_ids(dataset, limit, task_type)
        source_ids = []
        for sample_id in tqdm(
            sample_ids, desc="fit structure reference", disable=not config.show_progress
        ):
            task, source_id, values = _fit_sample(dataset[sample_id], config)
            source_ids.append(source_id)
            buckets = token_buckets(len(values), config.causal_position_bins)
            for token, bucket in enumerate(buckets):
                accumulator.add(task, int(bucket), values[token])

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        train_manifest.verify(Path(dataset.root) / "manifest.json")
        save_reference(
            output,
            accumulator.finish(
                train_dataset_manifest_sha256=train_manifest.sha256,
                source_ids=source_ids,
                sample_ids=sample_ids,
                audit_scope="complete_split"
                if limit is None and task_type.casefold() == "all"
                else "selected_samples",
            ),
        )

    def score(
        self,
        *,
        split_root,
        reference_path,
        output_dir,
        device="cpu",
        limit=None,
        task_type="QA",
    ) -> None:
        config = self.config
        reference_file = FrozenFile.capture(reference_path)
        reference = load_reference(reference_file.path)
        if tuple(reference.feature_names.tolist()) != FEATURE_NAMES:
            raise ValueError("reference features differ from the current method")
        expected = json.dumps(
            {**config.reference_settings(), "task_type": task_type}, sort_keys=True
        )
        if reference.settings_json != expected:
            raise ValueError("score settings differ from the fitted reference")

        dataset = open_research_dataset(split_root, device=device)
        output_dir = Path(output_dir)
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        sample_ids = _selected_ids(dataset, limit, task_type)
        source_audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=sample_ids,
            reserved_source_ids=reference.source_ids,
            require_complete_split=limit is None and task_type.casefold() == "all",
        )
        dataset_manifest = FrozenFile.capture(Path(dataset.root) / "manifest.json")

        for sample_id in tqdm(
            sample_ids, desc="freeze structure scores", disable=not config.show_progress
        ):
            sample = dataset[sample_id]
            source_audit.observe(sample)
            score_path = sample_dir / f"{sample_id}.npz"
            manifest_rows.append(
                self._score_to_file(sample, reference, score_path, output_dir)
            )

        audit = source_audit.finish()
        dataset_manifest.verify(Path(dataset.root) / "manifest.json")
        reference_file.verify(reference_file.path)

        write_json(
            output_dir / "manifest.json",
            {
                "schema": "non-neural-structure-manifest-v2",
                "labels_read": False,
                "trace_alignment": "post_token_query_at_same_position",
                "evaluation_alignment": "query_t_to_response_token_t_plus_1",
                "teacher_forced_trace": True,
                "claim_scope": (
                    "prompt-connected versus response-base attention-routing proxy; "
                    "not evidence grounding or model computation ancestry"
                ),
                "split_root": str(Path(split_root).resolve()),
                "dataset_manifest_sha256": dataset_manifest.sha256,
                "reference_path": str(reference_file.path),
                "reference_sha256": reference_file.sha256,
                "method_sha256": method_sha256(),
                "reference_source_ids": reference.source_ids.tolist(),
                "test_source_ids": list(audit.test_source_ids),
                "test_sample_ids": list(audit.test_sample_ids),
                "audit_scope": audit.test_scope,
                "task_type": task_type,
                "config": config.to_dict(),
                "feature_names": list(FEATURE_NAMES),
                "relation_names": list(RELATION_NAMES),
                "response_endpoint_null_relations": list(LINEAGE_RELATION_NAMES),
                "layer_order_null_relations": list(LAYER_ORDER_RELATION_NAMES),
                "samples": manifest_rows,
            },
        )

    def _score_to_file(self, sample, reference, score_path: Path, output_dir: Path):
        arrays, row = self._score_sample(sample, reference)
        save_npz(score_path, **arrays)
        row["score_path"] = str(score_path.relative_to(output_dir))
        row["score_sha256"] = file_sha256(score_path)
        return row

    def _score_sample(self, sample, reference):
        config = self.config
        with loaded_attention(sample) as attention:
            task = str(sample.task_type or "unknown")
            source_id = canonical_source_group(sample)
            response_tokens = (
                attention.token_ids[attention.response_idx :].cpu().numpy().copy()
            )
            edges = collect_routing_edges(sample, config=config)
        del attention
        routing, operator, feature_tensor = _analyze_edges(edges, config)
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

        null_scores = np.empty(
            (config.null_replicates, *real_relations.shape), dtype=np.float32
        )
        changed_fraction = np.empty(config.null_replicates, dtype=np.float32)
        null_audits = []
        base_seed = _sample_seed(sample.sample_id, config.random_seed)
        endpoint_plan = EndpointSwapPlan(edges, lag_bins=config.response_lag_bins)
        for replicate in tqdm(
            range(config.null_replicates),
            desc=f"{sample.sample_id} endpoint null",
            leave=False,
            disable=not config.show_progress,
        ):
            null = endpoint_plan.sample(
                seed=base_seed + replicate,
                rounds=config.swap_rounds,
            )
            null_scores[replicate] = _null_relations(
                operator,
                routing,
                null.edges.source,
                task=task,
                buckets=buckets,
                reference=reference,
                config=config,
            )
            changed_fraction[replicate] = null.changed_fraction
            null_audits.append(null.audit)
            del null

        final_relations = relation_scores(standardized[:, -1:, :])
        if edges.num_layers < 2:
            raise ValueError("layer-order null requires at least two layers")
        rng = np.random.default_rng(base_seed)
        shuffle_orders = np.empty(
            (config.layer_shuffle_replicates, edges.num_layers), dtype=np.int16
        )
        shuffle_scores = np.empty(
            (config.layer_shuffle_replicates, *final_relations.shape),
            dtype=np.float32,
        )
        for replicate in tqdm(
            range(config.layer_shuffle_replicates),
            desc=f"{sample.sample_id} layer shuffle",
            leave=False,
            disable=not config.show_progress,
        ):
            order = _non_identity_permutation(rng, edges.num_layers)
            shuffle_orders[replicate] = order
            shuffled_features = (
                build_layer_features(
                    routing,
                    operator.run(layer_order=order),
                    recent_tokens=config.recent_tokens,
                )
                .cpu()
                .numpy()
            )
            final_control = features.copy()
            for name in LAYER_ORDER_FEATURE_NAMES:
                index = FEATURE_INDEX[name]
                final_control[:, -1, index] = shuffled_features[:, -1, index]
            shuffled_standardized = standardize(
                final_control,
                task=task,
                buckets=buckets,
                reference=reference,
                maximum=config.maximum_standardized_value,
            )
            shuffle_scores[replicate] = relation_scores(
                shuffled_standardized[:, -1:, :]
            )
            del shuffled_features, final_control, shuffled_standardized

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
            "response_endpoint_null_relation_scores": null_scores,
            "response_endpoint_null_changed_fraction": changed_fraction,
            "layer_shuffle_order": shuffle_orders,
            "layer_shuffle_relation_scores": shuffle_scores,
        }
        row = {
            "sample_id": str(sample.sample_id),
            "source_id": source_id,
            "task_type": task,
            "response_length": len(features),
            "null_audit": {
                "row_mass_max_error": max(
                    audit["row_mass_max_error"] for audit in null_audits
                ),
                "role_mass_max_error": max(
                    audit["role_mass_max_error"] for audit in null_audits
                ),
                "source_count_degree_max_error": max(
                    audit["source_count_degree_max_error"] for audit in null_audits
                ),
                "causal_violations": max(
                    audit["causal_violations"] for audit in null_audits
                ),
                "duplicate_edges": max(
                    audit["duplicate_edges"] for audit in null_audits
                ),
                "eligible_response_edges": null_audits[0]["eligible_response_edges"],
                "changed_fraction_min": float(np.min(changed_fraction)),
                "changed_fraction_mean": float(np.mean(changed_fraction)),
            },
        }
        return arrays, row
