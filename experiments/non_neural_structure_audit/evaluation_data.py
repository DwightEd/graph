"""Open labels and frozen compact scores at the evaluation boundary."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from attention_lifecycle import loaded_attention

from .artifacts import load_npz, npz_shapes
from .bounded_ensemble import DiskBackedAUPRC, EnsembleAUPRC
from .bounded_samples import DiskBackedSamples
from .features import RELATION_NAMES
from .token_classes import content_token_mask

SCORE_FIELDS = (
    "response_token_ids",
    "relation_scores",
    "final_relation_scores",
    "response_endpoint_null_changed_fraction",
)

VALIDATION_FIELDS = (
    "schema",
    "sample_id",
    "source_id",
    "response_token_ids",
    "relation_names",
)

SCORE_MATRIX_FIELDS = (
    "relation_scores",
    "final_relation_scores",
    "response_endpoint_null_relation_scores",
    "response_endpoint_null_changed_fraction",
    "layer_shuffle_relation_scores",
)


@dataclass(frozen=True)
class FrozenSample:
    sample_id: str
    source_id: str
    labels: np.ndarray
    eligible: np.ndarray
    relation: np.ndarray
    final_relation: np.ndarray
    endpoint_null: np.ndarray
    layer_shuffle: np.ndarray
    endpoint_changed_fraction_mean: float
    endpoint_changed_fraction_min: float


@dataclass
class EvaluationBundle:
    samples: list[FrozenSample]
    endpoint_auprc: EnsembleAUPRC
    layer_auprc: EnsembleAUPRC
    _sample_store: DiskBackedSamples
    _temporary: TemporaryDirectory

    def close(self) -> None:
        self._sample_store.close()
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


def aligned_matrix(
    values: np.ndarray, labels: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(eligible, dtype=bool)[1:]
    return np.asarray(labels, dtype=np.int8)[1:][selected], values[:-1][selected]


def aligned_relation(
    values: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    relation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one relation before masking so no full matrix copy is created."""

    selected = np.asarray(eligible, dtype=bool)[1:]
    return (
        np.asarray(labels, dtype=np.int8)[1:][selected],
        np.asarray(values)[:-1, relation][selected],
    )


def grouped_relation(
    samples: list[FrozenSample], getter, relation: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    labels_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    scores_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in samples:
        labels, values = aligned_relation(
            getter(sample), sample.labels, sample.eligible, relation
        )
        labels_by_source[sample.source_id].append(labels)
        scores_by_source[sample.source_id].append(values)
    source_ids = sorted(labels_by_source)
    return (
        [np.concatenate(labels_by_source[source]) for source in source_ids],
        [np.concatenate(scores_by_source[source]) for source in source_ids],
    )


def validate_frozen_scores(
    *, dataset, score_dir: Path, rows: list[dict], config
) -> None:
    """Validate every selected score artifact before opening any labels."""

    expected_relations = np.asarray(RELATION_NAMES, dtype=str)
    relation_count = len(RELATION_NAMES)
    endpoint_replicates = int(config["null_replicates"])
    layer_replicates = int(config["layer_shuffle_replicates"])
    for row in rows:
        score_path = score_dir / row["score_path"]
        arrays = load_npz(score_path, VALIDATION_FIELDS)
        shapes = npz_shapes(score_path, SCORE_MATRIX_FIELDS)
        sample = dataset[str(row["sample_id"])]
        with loaded_attention(sample) as attention:
            canonical_tokens = (
                attention.token_ids[attention.response_idx :].cpu().numpy().copy()
            )
        del attention
        token_ids = arrays["response_token_ids"]
        token_count = len(canonical_tokens)
        valid = (
            str(arrays["schema"].item()) == "non-neural-structure-score-v1"
            and str(arrays["sample_id"].item()) == str(row["sample_id"])
            and str(arrays["source_id"].item()) == str(row["source_id"])
            and token_ids.ndim == 1
            and np.array_equal(token_ids, canonical_tokens)
            and int(row["response_length"]) == token_count
            and np.array_equal(arrays["relation_names"].astype(str), expected_relations)
            and shapes["relation_scores"] == (token_count, relation_count)
            and shapes["final_relation_scores"] == (token_count, relation_count)
            and shapes["response_endpoint_null_relation_scores"]
            == (endpoint_replicates, token_count, relation_count)
            and shapes["response_endpoint_null_changed_fraction"]
            == (endpoint_replicates,)
            and shapes["layer_shuffle_relation_scores"]
            == (layer_replicates, token_count, relation_count)
        )
        if not valid:
            raise ValueError("frozen score schema, identity, tokens, or shapes differ")


def _load_sample(
    *,
    label_store,
    score_dir: Path,
    row: dict,
    tokenizer,
    sample_store: DiskBackedSamples,
    endpoint_accumulator: DiskBackedAUPRC,
    layer_accumulator: DiskBackedAUPRC,
) -> FrozenSample:
    arrays = load_npz(score_dir / row["score_path"], SCORE_FIELDS)
    token_ids = arrays["response_token_ids"]
    labels = np.zeros(len(token_ids), dtype=np.int8)
    for start, end in label_store.positive_runs(
        row["sample_id"], response_count=len(token_ids)
    ):
        labels[start:end] = 1

    eligible = (
        np.ones(len(token_ids), dtype=bool)
        if tokenizer is None
        else content_token_mask(token_ids, tokenizer)
    )
    selected = eligible[1:]
    relation = arrays["relation_scores"].astype(np.float32, copy=False)
    final_relation = arrays["final_relation_scores"].astype(np.float32, copy=False)
    endpoint_null = load_npz(
        score_dir / row["score_path"],
        ("response_endpoint_null_relation_scores",),
    )["response_endpoint_null_relation_scores"].astype(np.float32, copy=False)
    endpoint_accumulator.add_masked(
        labels[1:],
        relation[:-1],
        endpoint_null[:, :-1],
        selected,
    )
    endpoint_mean = endpoint_null.mean(axis=0, dtype=np.float32)
    del endpoint_null
    layer_shuffle = load_npz(
        score_dir / row["score_path"],
        ("layer_shuffle_relation_scores",),
    )["layer_shuffle_relation_scores"].astype(np.float32, copy=False)
    layer_accumulator.add_masked(
        labels[1:],
        final_relation[:-1],
        layer_shuffle[:, :-1],
        selected,
    )
    layer_mean = layer_shuffle.mean(axis=0, dtype=np.float32)
    del layer_shuffle
    changed = arrays["response_endpoint_null_changed_fraction"]
    stored = sample_store.add(
        labels=labels,
        eligible=eligible,
        relation=relation,
        final_relation=final_relation,
        endpoint_null=endpoint_mean,
        layer_shuffle=layer_mean,
    )
    return FrozenSample(
        sample_id=str(row["sample_id"]),
        source_id=str(row["source_id"]),
        labels=stored[0],
        eligible=stored[1],
        relation=stored[2],
        final_relation=stored[3],
        endpoint_null=stored[4],
        layer_shuffle=stored[5],
        endpoint_changed_fraction_mean=float(changed.mean()),
        endpoint_changed_fraction_min=float(changed.min()),
    )


def load_frozen_samples(
    *,
    score_dir: Path,
    manifest: dict,
    scratch_dir,
    selected_sample_ids,
    dataset,
    tokenizer_path=None,
) -> EvaluationBundle:
    tokenizer = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    selected = set(selected_sample_ids)
    rows = [row for row in manifest["samples"] if str(row["sample_id"]) in selected]
    sample_capacity = sum(int(row["response_length"]) for row in rows)
    metric_capacity = sum(max(int(row["response_length"]) - 1, 0) for row in rows)
    config = manifest["config"]
    validate_frozen_scores(
        dataset=dataset,
        score_dir=score_dir,
        rows=rows,
        config=config,
    )
    label_store = dataset.prepare_evaluation_labels()
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    temporary_owner = TemporaryDirectory(prefix="structure-audit-", dir=scratch_dir)
    temporary = Path(temporary_owner.name)
    sample_store = DiskBackedSamples(
        temporary / "samples",
        capacity=sample_capacity,
        relations=len(RELATION_NAMES),
    )
    samples = []
    try:
        with (
            DiskBackedAUPRC(
                temporary / "endpoint",
                capacity=metric_capacity,
                replicates=int(config["null_replicates"]),
                relations=len(RELATION_NAMES),
            ) as endpoint,
            DiskBackedAUPRC(
                temporary / "layer",
                capacity=metric_capacity,
                replicates=int(config["layer_shuffle_replicates"]),
                relations=len(RELATION_NAMES),
            ) as layer,
        ):
            for row in rows:
                samples.append(
                    _load_sample(
                        label_store=label_store,
                        score_dir=score_dir,
                        row=row,
                        tokenizer=tokenizer,
                        sample_store=sample_store,
                        endpoint_accumulator=endpoint,
                        layer_accumulator=layer,
                    )
                )
            endpoint_auprc = endpoint.finish()
            layer_auprc = layer.finish()
        return EvaluationBundle(
            samples=samples,
            endpoint_auprc=endpoint_auprc,
            layer_auprc=layer_auprc,
            _sample_store=sample_store,
            _temporary=temporary_owner,
        )
    except Exception:
        sample_store.close()
        temporary_owner.cleanup()
        raise
