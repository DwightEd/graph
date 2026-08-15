"""End-to-end label-free experiment for causal attention topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .aligned_reservoir import AlignedReservoir
from .causal_topology import CausalTopologyConfig, CausalTopologyEncoder
from .one_class import OneClassConfig
from .token_representation import (
    _append_metadata,
    _cluster_bootstrap_difference,
    _metadata_arrays,
    _metadata_template,
    _ranking,
    _read_dataset_labels,
)
from .topology_one_class import TopologyOneClassModel, atomic_blocks


SCHEMA = "causal-topology-experiment-v1"
CHECKPOINT_SCHEMA = "causal-topology-checkpoint-v1"


@dataclass(frozen=True)
class TopologyExperimentConfig:
    """Runtime choices for the fixed label-free topology experiment."""

    reference_size: int = 12_000
    checkpoint_interval: int = 50
    bootstrap_replicates: int = 200
    seed: int = 42
    topology: CausalTopologyConfig = field(default_factory=CausalTopologyConfig)
    one_class: OneClassConfig = field(default_factory=OneClassConfig)

    def validate(self) -> None:
        if int(self.reference_size) < 4:
            raise ValueError("reference_size must be at least four")
        if int(self.checkpoint_interval) < 1:
            raise ValueError("checkpoint_interval must be positive")
        if int(self.bootstrap_replicates) < 1:
            raise ValueError("bootstrap_replicates must be positive")
        self.one_class.validate()


class TopologyExperiment:
    """Run topology encoding, blockwise one-class scoring, then evaluation."""

    def __init__(
        self,
        train_dataset,
        test_dataset,
        evaluation_dataset,
        *,
        output_dir: str | Path,
        config: TopologyExperimentConfig,
        encoder=None,
    ):
        config.validate()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.evaluation_dataset = evaluation_dataset
        self.output = Path(output_dir)
        self.config = config
        self.encoder = encoder or CausalTopologyEncoder(config.topology)

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        train_fingerprint = self._dataset_fingerprint(self.train_dataset)
        test_fingerprint = self._dataset_fingerprint(self.test_dataset)
        evaluation_fingerprint = self._dataset_fingerprint(self.evaluation_dataset)
        if evaluation_fingerprint != test_fingerprint:
            raise ValueError("evaluation dataset fingerprint does not match test scores")
        reservoir = self._build_train_reference()
        fit_bins = reservoir.bins("fit")
        calibration_bins = reservoir.bins("cal")
        model = TopologyOneClassModel(self.config.one_class).fit_loaders(
            reservoir.block_names,
            fit_bins,
            calibration_bins,
            lambda name: reservoir.block("fit", name),
            lambda name: reservoir.block("cal", name),
        )
        signature = self._signature(reservoir)
        self._save_model(model, signature)

        scores, metadata = self._score_test(model)
        artifact_path = self._save_label_free_artifact(
            scores, metadata, train_fingerprint, test_fingerprint
        )
        label_free_report = {
            "schema": SCHEMA,
            "labels_used": False,
            "primary_score": "full_signal",
            "score_names": list(scores),
            "test_nodes": int(len(next(iter(scores.values())))),
            "score_coordinates": [
                "attention_marginals", "causal_topology_exact",
            ],
            "dataset_fingerprints": {
                "train": train_fingerprint,
                "test": test_fingerprint,
            },
            "reference_split": {
                "method": "stable_sha256_of_source_or_sample_group",
                "mutually_exclusive": True,
                "total_budget": int(self.config.reference_size),
                "per_group_budget": int(math.ceil(self.config.reference_size / 2)),
                "fit_rows": int(len(fit_bins)),
                "calibration_rows": int(len(calibration_bins)),
            },
            "model_file": "topology_one_class_model.npz",
            "artifact_file": artifact_path.name,
            "saved_test_values": "fixed_scores_metadata_and_two_score_coordinates_only",
        }
        label_free_path = self.output / "topology_label_free_report.json"
        label_free_path.write_text(
            json.dumps(label_free_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        labels = _read_dataset_labels(
            self.evaluation_dataset, "open sealed topology evaluation labels"
        )
        if len(labels) != len(metadata["sample_id"]):
            raise ValueError("evaluation labels do not align with frozen topology scores")
        evaluation = {
            name: _ranking(labels, values) for name, values in scores.items()
        }
        comparisons = {
            "full_signal_vs_attention_marginals": (
                "full_signal", "attention_marginals",
            ),
            "causal_topology_exact_vs_attention_marginals": (
                "causal_topology_exact", "attention_marginals",
            ),
            "rr_multihop_exact_vs_lag_rewired": (
                "rr_multihop_exact", "rr_multihop_lag_rewired",
            ),
            "rr_multihop_exact_vs_one_hop_exact": (
                "rr_multihop_exact", "rr_one_hop_exact",
            ),
        }
        bootstrap = {
            name: _cluster_bootstrap_difference(
                labels, scores[first], scores[second], metadata["sample_id"],
                seed=self.config.seed,
                replicates=self.config.bootstrap_replicates,
                description=name,
            )
            for name, (first, second) in comparisons.items()
        }
        result = {
            **label_free_report,
            "labels_used": "evaluation_only_after_label_free_freeze",
            "score_evaluation": evaluation,
            "paired_bootstrap": bootstrap,
        }
        (self.output / "topology_experiment_report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._checkpoint_path.unlink(missing_ok=True)
        return result

    def _build_train_reference(self) -> AlignedReservoir:
        if self._checkpoint_path.is_file():
            reservoir, start = self._restore_checkpoint()
        else:
            reservoir, start = self._new_reservoir(), 0
        sample_ids = list(self.train_dataset.sample_ids)
        for index, sample_id in enumerate(tqdm(
            sample_ids[start:], desc="train topology reference", unit="sample",
            initial=start, total=len(sample_ids),
        ), start=start):
            sample = self.train_dataset[sample_id]
            attention = sample.attention()
            encoding = self.encoder.encode(attention)
            position = self._position(attention.num_response_tokens)
            reservoir.add(
                self._reference_group(sample), atomic_blocks(encoding), position
            )
            sample.release_attention()
            if (index + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(reservoir, index + 1)
        self._save_checkpoint(reservoir, len(sample_ids))
        return reservoir

    def _score_test(self, model: TopologyOneClassModel):
        score_parts: dict[str, list[np.ndarray]] = {}
        metadata = _metadata_template()
        for sample_id in tqdm(
            self.test_dataset.sample_ids, desc="score topology test", unit="sample"
        ):
            sample = self.test_dataset[sample_id]
            attention = sample.attention()
            encoding = self.encoder.encode(attention)
            result = model.transform(
                encoding, self._position(attention.num_response_tokens)
            )
            if not score_parts:
                score_parts = {name: [] for name in result.scores}
            if tuple(result.scores) != tuple(score_parts):
                raise ValueError("topology score contract changed between samples")
            for name, values in result.scores.items():
                score_parts[name].append(np.asarray(values, dtype=np.float32))
            _append_metadata(metadata, sample, attention)
            sample.release_attention()
        if not score_parts:
            raise ValueError("test dataset has no samples")
        scores = {
            name: np.concatenate(parts).astype(np.float32)
            for name, parts in score_parts.items()
        }
        return scores, _metadata_arrays(metadata)

    def _save_model(self, model: TopologyOneClassModel, signature: str) -> Path:
        path = self.output / "topology_one_class_model.npz"
        np.savez_compressed(
            path,
            schema=np.asarray(SCHEMA),
            labels_included=np.asarray(False),
            signature=np.asarray(signature),
            primary_score=np.asarray("full_signal"),
            **model.state(),
        )
        return path

    def _save_label_free_artifact(
        self,
        scores: dict[str, np.ndarray],
        metadata: dict[str, np.ndarray],
        train_fingerprint: str,
        test_fingerprint: str,
    ) -> Path:
        path = self.output / "topology_label_free.npz"
        payload = {
            "schema": np.asarray(SCHEMA),
            "labels_included": np.asarray(False),
            "primary_score": np.asarray("full_signal"),
            "train_dataset_fingerprint": np.asarray(train_fingerprint),
            "test_dataset_fingerprint": np.asarray(test_fingerprint),
            "score_names": np.asarray(tuple(scores)),
            "score_coordinates": np.column_stack((
                scores["attention_marginals"], scores["causal_topology_exact"],
            )).astype(np.float32),
            **metadata,
        }
        payload.update({f"{name}_score": values for name, values in scores.items()})
        np.savez_compressed(path, **payload)
        return path

    def _reference_group(self, sample) -> str:
        key = str(getattr(sample, "source_id", None) or sample.sample_id)
        digest = hashlib.sha256(
            f"topology-reference-v1\0{self.config.seed}\0{key}".encode("utf-8")
        ).digest()
        return "cal" if digest[0] % 2 else "fit"

    def _signature(self, reservoir: AlignedReservoir) -> str:
        snapshot = reservoir.snapshot()
        contract = {
            name: int(snapshot["widths"][name]) for name in reservoir.block_names
        }
        encoder_config = getattr(self.encoder, "config", None)
        if is_dataclass(encoder_config):
            encoder_config = asdict(encoder_config)
        value = {
            "manifest": self.train_dataset.manifest,
            "sample_ids": list(self.train_dataset.sample_ids),
            "dataset_fingerprint": self._dataset_fingerprint(self.train_dataset),
            "config": asdict(self.config),
            "encoder": {
                "type": (
                    f"{type(self.encoder).__module__}.{type(self.encoder).__qualname__}"
                ),
                "config": encoder_config,
            },
            "atomic_blocks": contract,
        }
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _dataset_fingerprint(dataset) -> str:
        rows = getattr(dataset, "rows", None)
        if isinstance(rows, dict):
            inventory = [rows[str(sample_id)] for sample_id in dataset.sample_ids]
        else:
            inventory = list(dataset.sample_ids)
        value = {"manifest": dataset.manifest, "sample_inventory": inventory}
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def _checkpoint_path(self) -> Path:
        return self.output / "train_reference_checkpoint.npz"

    def _new_reservoir(self) -> AlignedReservoir:
        return AlignedReservoir(
            position_bins=self.config.one_class.position_bins,
            size=math.ceil(self.config.reference_size / 2),
            seed=self.config.seed,
        )

    def _save_checkpoint(
        self, reservoir: AlignedReservoir, next_sample_index: int
    ) -> None:
        snapshot = reservoir.snapshot()
        names = tuple(snapshot["block_names"])
        groups = tuple(reservoir.groups)
        payload: dict[str, np.ndarray] = {
            "schema": np.asarray(CHECKPOINT_SCHEMA),
            "signature": np.asarray(self._signature(reservoir)),
            "next_sample_index": np.asarray(next_sample_index, dtype=np.int32),
            "block_names": np.asarray(names),
            "block_widths": np.asarray(
                [snapshot["widths"][name] for name in names], dtype=np.int32
            ),
            "groups": np.asarray(groups),
        }
        for group in groups:
            filled = np.asarray(snapshot["filled"][group], dtype=np.int64)
            payload[f"filled/{group}"] = filled
            payload[f"priorities/{group}"] = snapshot["priorities"][group]
            payload[f"rng_state/{group}"] = np.asarray(json.dumps(
                snapshot["rng_state"][group], sort_keys=True,
                default=lambda value: int(value),
            ))
            for index, name in enumerate(names):
                values = snapshot["values"][group][name].copy()
                for bin_id, count in enumerate(filled):
                    values[bin_id, int(count):] = 0
                payload[f"values/{group}/{index}"] = values
        temporary = self._checkpoint_path.with_name(
            f"{self._checkpoint_path.stem}.tmp.npz"
        )
        try:
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, self._checkpoint_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _restore_checkpoint(self) -> tuple[AlignedReservoir, int]:
        reservoir = self._new_reservoir()
        with np.load(self._checkpoint_path, allow_pickle=False) as saved:
            if str(saved["schema"]) != CHECKPOINT_SCHEMA:
                raise ValueError("unsupported topology checkpoint schema")
            names = tuple(saved["block_names"].astype(str).tolist())
            widths = {
                name: int(width) for name, width in zip(
                    names, saved["block_widths"].tolist()
                )
            }
            groups = tuple(saved["groups"].astype(str).tolist())
            if groups != reservoir.groups:
                raise ValueError("topology checkpoint reservoir groups changed")
            values = {
                group: {
                    name: np.asarray(saved[f"values/{group}/{index}"], dtype=np.float16)
                    for index, name in enumerate(names)
                }
                for group in groups
            }
            state = {
                "block_names": names,
                "widths": widths,
                "values": values,
                "priorities": {
                    group: np.asarray(saved[f"priorities/{group}"], dtype=np.float64)
                    for group in groups
                },
                "filled": {
                    group: np.asarray(saved[f"filled/{group}"], dtype=np.int64)
                    for group in groups
                },
                "rng_state": {
                    group: json.loads(str(saved[f"rng_state/{group}"]))
                    for group in groups
                },
            }
            next_sample_index = int(saved["next_sample_index"])
            signature = str(saved["signature"])
        reservoir.restore(state)
        if signature != self._signature(reservoir):
            raise ValueError("topology checkpoint signature does not match this run")
        if not 0 <= next_sample_index <= len(self.train_dataset.sample_ids):
            raise ValueError("topology checkpoint sample index is outside train inventory")
        return reservoir, next_sample_index

    @staticmethod
    def _position(count: int) -> np.ndarray:
        return np.arange(count, dtype=np.float32) / max(count - 1, 1)
