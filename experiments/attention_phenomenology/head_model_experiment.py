"""End-to-end head-resolved validation and supervised token experiment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from research_dataset import open_research_dataset

from .artifacts import write_json
from .causal_head_model import (
    CausalLayerTemporalDetector,
    HeadSequence,
    SequencePrediction,
    TrainingConfig,
)
from .head_effects import HeadLayerEffectMap
from .head_resolved import HeadResolvedFeatureExtractor


@dataclass(frozen=True)
class HeadModelExperimentConfig:
    validation_fraction: float = 0.2
    reuse_top_k: int = 5
    recent_response_tokens: int = 4
    block_rows: int = 8192
    train_limit: int | None = None
    test_limit: int | None = None
    seed: int = 20260820


def source_disjoint_train_validation_split(
    sequences: Sequence[HeadSequence],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[HeadSequence], list[HeadSequence]]:
    """Split whole source groups, stratified by task and hallucination presence."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    by_source: dict[str, list[HeadSequence]] = defaultdict(list)
    for sequence in sequences:
        by_source[sequence.source_id].append(sequence.validate())
    if len(by_source) < 2:
        raise ValueError("source-disjoint validation requires at least two sources")

    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for source_id, members in by_source.items():
        tasks = sorted({member.task_type for member in members})
        task = tasks[0] if len(tasks) == 1 else "mixed"
        has_positive = any(bool(member.labels.sum().item()) for member in members)
        strata[(task, has_positive)].append(source_id)

    rng = np.random.default_rng(seed)
    validation_sources: set[str] = set()
    for source_ids in strata.values():
        source_ids = sorted(source_ids)
        rng.shuffle(source_ids)
        if len(source_ids) == 1:
            continue
        count = int(round(len(source_ids) * validation_fraction))
        count = min(max(count, 1), len(source_ids) - 1)
        validation_sources.update(source_ids[:count])

    if not validation_sources:
        candidates = sorted(by_source)
        validation_sources.add(candidates[int(rng.integers(len(candidates)))])
    if len(validation_sources) == len(by_source):
        validation_sources.remove(sorted(validation_sources)[-1])

    train = []
    validation = []
    for sequence in sequences:
        target = validation if sequence.source_id in validation_sources else train
        target.append(sequence)
    if not train or not validation:
        raise RuntimeError("source split produced an empty partition")
    return train, validation


class HeadResolvedExperiment:
    """Extract, split, train, map validation effects, and evaluate once."""

    def __init__(
        self,
        *,
        experiment_config: HeadModelExperimentConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> None:
        self.experiment_config = (
            HeadModelExperimentConfig()
            if experiment_config is None
            else experiment_config
        )
        self.training_config = TrainingConfig() if training_config is None else training_config

    def run(
        self,
        *,
        train_split,
        test_split,
        output_dir,
        device: str = "cpu",
    ) -> dict[str, object]:
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("output_dir must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)

        extractor = HeadResolvedFeatureExtractor(
            reuse_top_k=self.experiment_config.reuse_top_k,
            recent_response_tokens=self.experiment_config.recent_response_tokens,
            block_rows=self.experiment_config.block_rows,
        )
        all_train = self._extract_split(
            train_split,
            extractor=extractor,
            device=device,
            limit=self.experiment_config.train_limit,
            description="head features train+validation",
        )
        train, validation = source_disjoint_train_validation_split(
            all_train,
            validation_fraction=self.experiment_config.validation_fraction,
            seed=self.experiment_config.seed,
        )
        self._require_both_classes(train, "train")
        self._require_both_classes(validation, "validation")

        test = self._extract_split(
            test_split,
            extractor=extractor,
            device=device,
            limit=self.experiment_config.test_limit,
            description="head features test",
        )
        reserved_sources = {sequence.source_id for sequence in all_train}
        test_sources = {sequence.source_id for sequence in test}
        overlap = reserved_sources & test_sources
        if overlap:
            raise ValueError(f"train/test source overlap: {sorted(overlap)[:3]}")
        self._require_both_classes(test, "test")

        detector = CausalLayerTemporalDetector(self.training_config, device=device)
        history = detector.fit(train, validation)
        validation_metrics = detector.evaluate(validation)
        test_metrics = detector.evaluate(test)
        detector.save(output_dir / "model.pt")

        if detector.normalizer is None:
            raise RuntimeError("fitted detector has no normalizer")
        validation_effects = HeadLayerEffectMap().compute(
            (
                (
                    detector.normalizer.transform(sequence.values).cpu().numpy(),
                    sequence.labels.cpu().numpy(),
                )
                for sequence in validation
            ),
            feature_names=extractor.feature_names,
        )
        effect_paths = validation_effects.save(output_dir, prefix="validation")

        predictions = detector.predict(test)
        self._save_predictions(output_dir / "test_scores.npz", predictions)
        write_json(output_dir / "training_history.json", {"epochs": history})
        split_audit = {
            "train_sample_ids": [sequence.sample_id for sequence in train],
            "validation_sample_ids": [sequence.sample_id for sequence in validation],
            "test_sample_ids": [sequence.sample_id for sequence in test],
            "train_source_ids": sorted({sequence.source_id for sequence in train}),
            "validation_source_ids": sorted(
                {sequence.source_id for sequence in validation}
            ),
            "test_source_ids": sorted(test_sources),
        }
        write_json(output_dir / "split_audit.json", split_audit)

        result = {
            "schema": "head-resolved-layer-temporal-evaluation-v1",
            "labels_read": True,
            "protocol": (
                "train and validation are source-disjoint partitions of train_split; "
                "validation selects the epoch and owns head/layer effect maps; "
                "test_split is evaluated only after model selection"
            ),
            "train_split": str(Path(train_split).resolve()),
            "test_split": str(Path(test_split).resolve()),
            "experiment_config": asdict(self.experiment_config),
            "training_config": asdict(self.training_config),
            "feature_names": list(extractor.feature_names),
            "geometry": {
                "layers": int(train[0].values.shape[1]),
                "heads": int(train[0].values.shape[2]),
                "features": int(train[0].values.shape[3]),
            },
            "samples": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "tokens": {
                "train": sum(len(sequence.labels) for sequence in train),
                "validation": sum(len(sequence.labels) for sequence in validation),
                "test": sum(len(sequence.labels) for sequence in test),
            },
            "best_epoch": detector.best_epoch,
            "validation": validation_metrics,
            "test": test_metrics,
            "outputs": {
                "model": "model.pt",
                "test_scores": "test_scores.npz",
                "training_history": "training_history.json",
                "split_audit": "split_audit.json",
                "validation_effect_csv": effect_paths["csv"].name,
                "validation_effect_tensor": effect_paths["tensor"].name,
                "validation_effect_figure": effect_paths["figure"].name,
            },
        }
        write_json(output_dir / "evaluation.json", result)
        return result

    @staticmethod
    def _extract_split(
        split_root,
        *,
        extractor: HeadResolvedFeatureExtractor,
        device: str,
        limit: int | None,
        description: str,
    ) -> list[HeadSequence]:
        dataset = open_research_dataset(split_root, device=device)
        labels = dataset.prepare_evaluation_labels()
        sample_ids = dataset.sample_ids if limit is None else dataset.sample_ids[:limit]
        sequences = []
        for sample_id in tqdm(sample_ids, desc=description, unit="sample"):
            sample = dataset[sample_id]
            try:
                # Feature extraction cannot access the label store. Labels are
                # aligned only after the complete causal tensor exists.
                features = extractor.extract(sample)
                token_labels = labels.response_labels(sample).float().cpu()
                sequences.append(
                    HeadSequence(
                        sample_id=str(sample.sample_id),
                        source_id=str(sample.source_id),
                        task_type=str(sample.task_type or "unknown"),
                        values=features.values.detach().cpu().to(torch.float16),
                        labels=token_labels,
                    ).validate()
                )
            finally:
                sample.release_attention()
        return sequences

    @staticmethod
    def _require_both_classes(sequences: Sequence[HeadSequence], name: str) -> None:
        labels = torch.cat([sequence.labels for sequence in sequences])
        if not bool((labels == 0).any() and (labels == 1).any()):
            raise ValueError(f"{name} partition needs both token classes; increase limit")

    @staticmethod
    def _save_predictions(path: Path, predictions: Sequence[SequencePrediction]) -> None:
        sample_id = []
        source_id = []
        task_type = []
        token_index = []
        labels = []
        current = []
        forecast = []
        for prediction in predictions:
            count = len(prediction.labels)
            sample_id.append(np.full(count, prediction.sample_id, dtype=str))
            source_id.append(np.full(count, prediction.source_id, dtype=str))
            task_type.append(np.full(count, prediction.task_type, dtype=str))
            token_index.append(np.arange(count, dtype=np.int32))
            labels.append(prediction.labels)
            current.append(prediction.current_probability)
            forecast.append(prediction.forecast_probability)
        np.savez_compressed(
            path,
            sample_id=np.concatenate(sample_id),
            source_id=np.concatenate(source_id),
            task_type=np.concatenate(task_type),
            token_index=np.concatenate(token_index),
            token_label=np.concatenate(labels),
            current_probability=np.concatenate(current),
            forecast_probability=np.concatenate(forecast),
        )
