"""Label-free normal-state fitting and token-level routing-basin scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .routing import (
    CONTROL_NAMES,
    CausalRoutingFeatureExtractor,
    FEATURE_NAMES,
    RoutingFeatureConfig,
    RoutingSequence,
)


COMPONENT_NAMES = (
    "state_novelty",
    "transition_surprise",
    "basin_commitment",
    "smoothed_commitment",
)


@dataclass(frozen=True)
class DetectorConfig:
    """Fixed normal-reference and causal calibration choices."""

    calibration_fraction: float = 0.2
    ridge: float = 1e-2
    smoothing_decay: float = 0.9
    student_df: float = 4.0
    threshold_quantile: float = 0.95

    def validate(self) -> None:
        if not 0 < float(self.calibration_fraction) < 1:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) <= 0:
            raise ValueError("ridge must be positive and finite")
        if not 0 <= float(self.smoothing_decay) < 1:
            raise ValueError("smoothing_decay must be in [0, 1)")
        if not math.isfinite(float(self.student_df)) or float(self.student_df) <= 0:
            raise ValueError("student_df must be positive and finite")
        if not 0 < float(self.threshold_quantile) < 1:
            raise ValueError("threshold_quantile must be in (0, 1)")


@dataclass(frozen=True)
class TokenScoreTable:
    """Complete response-token rows with calibrated and explanatory scores."""

    score: np.ndarray
    component_score: dict[str, np.ndarray]
    component_raw: dict[str, np.ndarray]
    winning_component: np.ndarray
    valid: np.ndarray
    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    data_source: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray
    control_names: tuple[str, ...]
    controls: np.ndarray
    threshold: float
    online_causal_score: bool
    alignment: str


class TokenRoutingDetector:
    """Fit one normal routing reference and score each held-out token."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        feature_config: RoutingFeatureConfig | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.config.validate()
        self.feature_config = feature_config or RoutingFeatureConfig()
        self.feature_config.validate()
        self.extractor = CausalRoutingFeatureExtractor(self.feature_config)

    def fit(self, train_dataset, *, limit: int | None = None) -> "TokenRoutingDetector":
        """Fit nuisance, dynamics, and calibration references without labels."""

        self._capture_geometry(train_dataset)
        sequences = self._extract_dataset(
            train_dataset, limit=limit, description="fit routing reference"
        )
        source_ids = sorted({sequence.source_id for sequence in sequences})
        if len(source_ids) < 3:
            raise ValueError("fit requires at least three source groups")
        calibration_count = min(
            len(source_ids) - 1,
            max(2, round(len(source_ids) * self.config.calibration_fraction)),
        )
        ordered = sorted(source_ids, key=self._source_key)
        self.calibration_source_ids = tuple(ordered[:calibration_count])
        component_count = max(1, calibration_count // 2)
        self.component_calibration_source_ids = tuple(ordered[:component_count])
        self.final_calibration_source_ids = tuple(
            ordered[component_count:calibration_count]
        )
        self.fit_source_ids = tuple(ordered[calibration_count:])
        self.train_source_ids = tuple(source_ids)

        fit_sequences = [
            sequence
            for sequence in sequences
            if sequence.source_id in self.fit_source_ids
        ]
        component_calibration_sequences = [
            sequence
            for sequence in sequences
            if sequence.source_id in self.component_calibration_source_ids
        ]
        final_calibration_sequences = [
            sequence
            for sequence in sequences
            if sequence.source_id in self.final_calibration_source_ids
        ]
        self.task_types = tuple(
            sorted({sequence.task_type for sequence in fit_sequences})
        )
        fit_values, fit_controls = self._valid_rows(fit_sequences)
        if len(fit_values) < 2:
            raise ValueError("fit source groups contain too few valid token rows")
        self.nuisance_coefficients = self._ridge_fit(fit_controls, fit_values)
        fit_residual = fit_values - self._multiply(
            self._design(fit_controls), self.nuisance_coefficients
        )
        self.residual_center = np.median(fit_residual, axis=0).astype(np.float32)
        mad = 1.4826 * np.median(
            np.abs(fit_residual - self.residual_center), axis=0
        )
        standard_deviation = fit_residual.std(axis=0)
        self.residual_scale = np.where(
            mad > 1e-6,
            mad,
            np.where(standard_deviation > 1e-6, standard_deviation, 1.0),
        ).astype(np.float32)
        self.dynamics_coefficients = self._fit_dynamics(fit_sequences)

        calibration_raw = self._component_rows(component_calibration_sequences)
        self.component_reference = {}
        for name in COMPONENT_NAMES:
            reference = calibration_raw[name][calibration_raw[f"valid/{name}"]]
            if not len(reference):
                raise ValueError(
                    f"component calibration contains no valid {name} tokens"
                )
            self.component_reference[name] = np.sort(reference.astype(np.float32))

        final_raw = self._component_rows(final_calibration_sequences)
        final_score = self._calibrated_components(final_raw)
        calibration_max = self._available_maximum(final_score)
        self.final_reference = np.sort(
            calibration_max[final_raw["valid"]].astype(np.float32)
        )
        if not len(self.final_reference):
            raise ValueError("final calibration contains no valid token rows")
        self.threshold = float(
            -math.log(1.0 - float(self.config.threshold_quantile))
        )
        return self

    def score(self, dataset, *, limit: int | None = None) -> TokenScoreTable:
        """Return one prefix-causal anomaly row for every held-out token."""

        self._require_fitted()
        self._check_geometry(dataset)
        sequences = self._extract_dataset(
            dataset, limit=limit, description="score routing tokens"
        )
        if (
            sequences[0].names != self.feature_names
            or sequences[0].control_names != self.control_names
        ):
            raise ValueError("routing feature schema differs from the reference")
        overlap = sorted(
            {sequence.source_id for sequence in sequences}.intersection(
                self.train_source_ids
            )
        )
        if overlap:
            raise ValueError(
                f"score source groups overlap the fitted reference: {overlap[:3]}"
            )
        rows = self._component_rows(sequences)
        component_score = self._calibrated_components(rows)
        maximum = self._available_maximum(component_score)
        score = self._upper_tail_surprisal(self.final_reference, maximum)
        score[~rows["valid"]] = np.nan
        winning = np.asarray(COMPONENT_NAMES, dtype="U32")[
            np.argmax(
                np.column_stack(
                    [
                        np.nan_to_num(component_score[name], nan=-np.inf)
                        for name in COMPONENT_NAMES
                    ]
                ),
                axis=1,
            )
        ]
        winning[~rows["valid"]] = "invalid"
        return TokenScoreTable(
            score=score,
            component_score=component_score,
            component_raw={
                name: np.where(rows[f"valid/{name}"], rows[name], np.nan)
                for name in COMPONENT_NAMES
            },
            winning_component=winning,
            valid=rows["valid"],
            sample_id=rows["sample_id"],
            source_id=rows["source_id"],
            task_type=rows["task_type"],
            data_source=rows["data_source"],
            token_index=rows["token_index"],
            response_length=rows["response_length"],
            feature_names=sequences[0].names,
            features=np.concatenate(
                [sequence.values.float().cpu().numpy() for sequence in sequences]
            ),
            control_names=sequences[0].control_names,
            controls=np.concatenate(
                [sequence.controls.float().cpu().numpy() for sequence in sequences]
            ),
            threshold=self.threshold,
            online_causal_score=True,
            alignment=self.alignment,
        )

    def state(self) -> dict[str, np.ndarray]:
        """Return a strict array-only state suitable for an NPZ artifact."""

        self._require_fitted()
        state = {
            "schema": np.asarray("token-routing-basin-reference-v2"),
            "alignment": np.asarray(self.alignment),
            "num_layers": np.asarray(self.num_layers, dtype=np.int16),
            "num_heads": np.asarray(self.num_heads, dtype=np.int16),
            "attention_floor": np.asarray(self.attention_floor, dtype=np.float32),
            "feature_names": np.asarray(self.feature_names),
            "control_names": np.asarray(self.control_names),
            "fit_source_ids": np.asarray(self.fit_source_ids),
            "calibration_source_ids": np.asarray(self.calibration_source_ids),
            "component_calibration_source_ids": np.asarray(
                self.component_calibration_source_ids
            ),
            "final_calibration_source_ids": np.asarray(
                self.final_calibration_source_ids
            ),
            "train_source_ids": np.asarray(self.train_source_ids),
            "task_types": np.asarray(self.task_types),
            "nuisance_coefficients": self.nuisance_coefficients,
            "residual_center": self.residual_center,
            "residual_scale": self.residual_scale,
            "dynamics_coefficients": self.dynamics_coefficients,
            "final_reference": self.final_reference,
            "threshold": np.asarray(self.threshold, dtype=np.float32),
        }
        for name, value in asdict(self.config).items():
            state[f"detector_config/{name}"] = np.asarray(value)
        for name, value in asdict(self.feature_config).items():
            state[f"feature_config/{name}"] = np.asarray(value)
        for name in COMPONENT_NAMES:
            state[f"component_reference/{name}"] = self.component_reference[name]
        return state

    @classmethod
    def from_state(cls, state) -> "TokenRoutingDetector":
        """Construct a fitted detector from a validated NPZ-like mapping."""

        schema = cls._scalar_text(state, "schema")
        if schema != "token-routing-basin-reference-v2":
            raise ValueError("unsupported token routing reference schema")
        detector_fields = DetectorConfig.__dataclass_fields__
        feature_fields = RoutingFeatureConfig.__dataclass_fields__
        detector = cls(
            DetectorConfig(
                **{
                    name: np.asarray(state[f"detector_config/{name}"]).item()
                    for name in detector_fields
                }
            ),
            feature_config=RoutingFeatureConfig(
                **{
                    name: np.asarray(state[f"feature_config/{name}"]).item()
                    for name in feature_fields
                }
            ),
        )
        detector.alignment = cls._scalar_text(state, "alignment")
        detector.num_layers = int(np.asarray(state["num_layers"]).item())
        detector.num_heads = int(np.asarray(state["num_heads"]).item())
        detector.attention_floor = float(np.asarray(state["attention_floor"]).item())
        detector.feature_names = tuple(map(str, np.asarray(state["feature_names"])))
        detector.control_names = tuple(map(str, np.asarray(state["control_names"])))
        detector.fit_source_ids = tuple(map(str, np.asarray(state["fit_source_ids"])))
        detector.calibration_source_ids = tuple(
            map(str, np.asarray(state["calibration_source_ids"]))
        )
        detector.component_calibration_source_ids = tuple(
            map(str, np.asarray(state["component_calibration_source_ids"]))
        )
        detector.final_calibration_source_ids = tuple(
            map(str, np.asarray(state["final_calibration_source_ids"]))
        )
        detector.train_source_ids = tuple(
            map(str, np.asarray(state["train_source_ids"]))
        )
        detector.task_types = tuple(map(str, np.asarray(state["task_types"])))
        for name in (
            "nuisance_coefficients",
            "residual_center",
            "residual_scale",
            "dynamics_coefficients",
            "final_reference",
        ):
            setattr(detector, name, np.asarray(state[name], dtype=np.float32))
        detector.component_reference = {
            name: np.asarray(state[f"component_reference/{name}"], dtype=np.float32)
            for name in COMPONENT_NAMES
        }
        detector.threshold = float(np.asarray(state["threshold"]).item())
        detector._validate_loaded_state()
        return detector

    def _capture_geometry(self, dataset) -> None:
        manifest = dataset.manifest
        self.alignment = str(manifest.get("alignment", ""))
        if self.alignment != "post_token_query_at_same_position":
            raise ValueError("dataset has an unsupported token/attention alignment")
        self.num_layers = int(manifest["num_layers"])
        self.num_heads = int(manifest["num_heads"])
        self.attention_floor = float(manifest["attention_floor"])
        self.feature_names = FEATURE_NAMES
        self.control_names = CONTROL_NAMES

    def _check_geometry(self, dataset) -> None:
        manifest = dataset.manifest
        actual = (
            str(manifest.get("alignment", "")),
            int(manifest["num_layers"]),
            int(manifest["num_heads"]),
            float(manifest["attention_floor"]),
        )
        expected = (
            self.alignment,
            self.num_layers,
            self.num_heads,
            self.attention_floor,
        )
        if actual[:3] != expected[:3] or not math.isclose(
            actual[3], expected[3], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("dataset attention geometry differs from the reference")

    def _extract_dataset(self, dataset, *, limit, description):
        if limit is not None and int(limit) < 1:
            raise ValueError("limit must be positive")
        sequences = []
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        total = (
            min(dataset_size, int(limit))
            if dataset_size is not None and limit is not None
            else dataset_size
        )
        for index, sample in enumerate(
            tqdm(dataset, total=total, desc=description, unit="sample")
        ):
            if limit is not None and index >= int(limit):
                break
            sequence = self.extractor.extract(sample)
            if sequences and (
                sequence.names != sequences[0].names
                or sequence.control_names != sequences[0].control_names
            ):
                raise ValueError("routing feature schemas changed within a dataset")
            sequences.append(sequence)
        if not sequences:
            raise ValueError("dataset selection contains no samples")
        return sequences

    @staticmethod
    def _source_key(source_id):
        return hashlib.sha256(str(source_id).encode("utf-8")).digest()

    @staticmethod
    def _design(controls):
        controls = np.asarray(controls, dtype=np.float32)
        return np.column_stack(
            (np.ones(len(controls), dtype=np.float32), controls)
        )

    def _ridge_fit(self, controls, targets):
        return self._ridge_solution(self._design(controls), targets)

    def _nuisance_controls(self, sequence):
        controls = sequence.controls.float().cpu().numpy()
        task = np.asarray(
            [float(sequence.task_type == name) for name in self.task_types],
            dtype=np.float32,
        )
        if not len(task):
            return controls
        return np.column_stack(
            (controls, np.repeat(task[None, :], len(controls), axis=0))
        )

    def _valid_rows(self, sequences):
        values = np.concatenate(
            [sequence.values.float().cpu().numpy() for sequence in sequences]
        )
        controls = np.concatenate(
            [self._nuisance_controls(sequence) for sequence in sequences]
        )
        valid = np.concatenate(
            [sequence.valid.cpu().numpy() for sequence in sequences]
        ).astype(bool)
        return values[valid], controls[valid]

    def _standardize(self, sequence: RoutingSequence):
        values = sequence.values.float().cpu().numpy()
        controls = self._nuisance_controls(sequence)
        residual = values - self._multiply(
            self._design(controls), self.nuisance_coefficients
        )
        return (residual - self.residual_center) / self.residual_scale

    def _fit_dynamics(self, sequences):
        previous, current = [], []
        for sequence in sequences:
            z = self._standardize(sequence)
            valid = sequence.valid.cpu().numpy().astype(bool)
            pair = valid[1:] & valid[:-1]
            if pair.any():
                previous.append(z[:-1][pair])
                current.append(z[1:][pair])
        if not previous:
            raise ValueError("fit source groups contain no valid token transitions")
        previous = np.concatenate(previous)
        current = np.concatenate(current)
        return self._ridge_solution(self._design(previous), current)

    def _ridge_solution(self, design, targets):
        """Solve a small ridge system through torch for stable local execution."""

        design_tensor = torch.as_tensor(design, dtype=torch.float64)
        target_tensor = torch.as_tensor(targets, dtype=torch.float64)
        penalty = torch.eye(design_tensor.shape[1], dtype=torch.float64)
        penalty *= float(self.config.ridge)
        penalty[0, 0] = 0
        coefficients = torch.linalg.solve(
            design_tensor.T @ design_tensor + penalty,
            design_tensor.T @ target_tensor,
        )
        return coefficients.float().cpu().numpy()

    @staticmethod
    def _multiply(left, right):
        return (
            torch.as_tensor(left, dtype=torch.float32)
            .matmul(torch.as_tensor(right, dtype=torch.float32))
            .cpu()
            .numpy()
        )

    def _sequence_components(self, sequence):
        z = self._standardize(sequence).astype(np.float32)
        valid = sequence.valid.cpu().numpy().astype(bool)
        state = np.mean(
            np.log1p((z * z) / self.config.student_df), axis=1
        ).astype(np.float32)
        transition = np.zeros(len(z), dtype=np.float32)
        if len(z) > 1:
            prediction = self._multiply(
                self._design(z[:-1]), self.dynamics_coefficients
            )
            error = z[1:] - prediction
            transition[1:] = np.mean(
                np.log1p((error * error) / self.config.student_df), axis=1
            )
            transition[1:][~(valid[1:] & valid[:-1])] = 0

        directions = {
            "prompt_top1_share": 1,
            "response_effective_source_fraction": -1,
            "response_top1_share": 1,
            "recent_response_share": 1,
            "prompt_anchor_run_fraction": 1,
            "multiplex_route_effective_rank_fraction": -1,
            "multiplex_route_dominant_mode_share": 1,
            "relative_route_velocity": -1,
        }
        oriented = np.column_stack(
            [
                z[:, sequence.names.index(name)] * direction
                for name, direction in directions.items()
            ]
        )
        commitment = np.maximum(oriented, 0).mean(axis=1).astype(np.float32)
        smoothed = np.zeros_like(commitment)
        previous_smoothed = 0.0
        for token, value in enumerate(commitment):
            if valid[token]:
                previous_smoothed = (
                    self.config.smoothing_decay * previous_smoothed
                    + (1 - self.config.smoothing_decay) * float(value)
                )
            else:
                previous_smoothed = 0.0
            smoothed[token] = previous_smoothed
        transition_valid = np.zeros_like(valid)
        if len(valid) > 1:
            transition_valid[1:] = valid[1:] & valid[:-1]
        return {
            "state_novelty": state,
            "transition_surprise": transition,
            "basin_commitment": commitment,
            "smoothed_commitment": smoothed,
            "valid": valid,
            "valid/state_novelty": valid,
            "valid/transition_surprise": transition_valid,
            "valid/basin_commitment": valid,
            "valid/smoothed_commitment": valid,
        }

    def _component_rows(self, sequences):
        components = [self._sequence_components(sequence) for sequence in sequences]
        rows = {
            name: np.concatenate([component[name] for component in components])
            for name in COMPONENT_NAMES
        }
        rows["valid"] = np.concatenate(
            [component["valid"] for component in components]
        )
        for name in COMPONENT_NAMES:
            rows[f"valid/{name}"] = np.concatenate(
                [component[f"valid/{name}"] for component in components]
            )
        rows["sample_id"] = np.concatenate(
            [
                np.repeat(np.asarray(sequence.sample_id), len(sequence.values))
                for sequence in sequences
            ]
        )
        rows["source_id"] = np.concatenate(
            [
                np.repeat(np.asarray(sequence.source_id), len(sequence.values))
                for sequence in sequences
            ]
        )
        rows["task_type"] = np.concatenate(
            [
                np.repeat(np.asarray(sequence.task_type), len(sequence.values))
                for sequence in sequences
            ]
        )
        rows["data_source"] = np.concatenate(
            [
                np.repeat(np.asarray(sequence.data_source), len(sequence.values))
                for sequence in sequences
            ]
        )
        rows["token_index"] = np.concatenate(
            [np.arange(len(sequence.values), dtype=np.int32) for sequence in sequences]
        )
        rows["response_length"] = np.concatenate(
            [
                np.full(len(sequence.values), len(sequence.values), dtype=np.int32)
                for sequence in sequences
            ]
        )
        return rows

    def _calibrated_components(self, rows):
        scores = {}
        for name in COMPONENT_NAMES:
            values = self._upper_tail_surprisal(
                self.component_reference[name], rows[name]
            )
            values[~rows[f"valid/{name}"]] = np.nan
            scores[name] = values
        return scores

    @staticmethod
    def _available_maximum(component_score):
        stacked = np.column_stack(
            [component_score[name] for name in COMPONENT_NAMES]
        )
        return np.max(np.nan_to_num(stacked, nan=-np.inf), axis=1)

    @staticmethod
    def _upper_tail_surprisal(reference, values):
        reference = np.asarray(reference, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        first_greater_equal = np.searchsorted(reference, values, side="left")
        probability = (len(reference) - first_greater_equal + 1) / (
            len(reference) + 1.0
        )
        return -np.log(probability).astype(np.float32)

    def _require_fitted(self):
        if not hasattr(self, "final_reference"):
            raise RuntimeError("fit must be called before score or state")

    @staticmethod
    def _scalar_text(state, name):
        value = np.asarray(state[name])
        if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
            raise ValueError(f"reference field {name!r} must be scalar text")
        item = value.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)

    def _validate_loaded_state(self):
        if self.alignment != "post_token_query_at_same_position":
            raise ValueError("reference has unsupported attention alignment")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("reference attention geometry must be positive")
        if not math.isfinite(self.attention_floor) or self.attention_floor < 0:
            raise ValueError("reference attention floor must be finite and non-negative")
        if self.feature_names != FEATURE_NAMES or self.control_names != CONTROL_NAMES:
            raise ValueError("reference feature schema differs from this implementation")
        fit = set(self.fit_source_ids)
        component_calibration = set(self.component_calibration_source_ids)
        final_calibration = set(self.final_calibration_source_ids)
        calibration = set(self.calibration_source_ids)
        train = set(self.train_source_ids)
        if (
            not fit
            or not component_calibration
            or not final_calibration
            or fit & calibration
            or component_calibration & final_calibration
            or calibration != component_calibration | final_calibration
            or train != fit | calibration
        ):
            raise ValueError("reference source-group partitions are inconsistent")
        expected_features = len(self.feature_names)
        expected_controls = len(self.control_names)
        if self.nuisance_coefficients.shape != (
            expected_controls + len(self.task_types) + 1,
            expected_features,
        ):
            raise ValueError("reference nuisance coefficients have invalid shape")
        if self.dynamics_coefficients.shape != (
            expected_features + 1,
            expected_features,
        ):
            raise ValueError("reference dynamics coefficients have invalid shape")
        if self.residual_center.shape != (expected_features,) or (
            self.residual_scale.shape != (expected_features,)
        ):
            raise ValueError("reference feature statistics have invalid shape")
        numeric_state = (
            self.nuisance_coefficients,
            self.dynamics_coefficients,
            self.residual_center,
            self.residual_scale,
        )
        if any(not np.isfinite(values).all() for values in numeric_state):
            raise ValueError("reference fitted arrays must be finite")
        if np.any(self.residual_scale <= 0):
            raise ValueError("reference feature scales must be positive")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("reference threshold must be positive and finite")
        for reference in (*self.component_reference.values(), self.final_reference):
            if reference.ndim != 1 or not len(reference):
                raise ValueError("calibration references must be non-empty vectors")
            if not np.isfinite(reference).all() or np.any(reference[1:] < reference[:-1]):
                raise ValueError("calibration references must be finite and sorted")
