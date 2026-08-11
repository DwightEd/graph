"""Paired onset-aligned validation for structural attention features."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from behavior import validate_positive_runs
from cache import sha256
from onset_validation import causal_rewire, merge_positive_runs
from research_dataset import (
    STRUCTURAL_FEATURE_NAMES,
    ResearchDataset,
    relations_from_graph,
    structural_features_from_relations,
)

PRIMARY_FEATURES = (
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "history_edge_share",
)
STRATUM_FIELDS = ("task_type", "data_source", "generator_model", "temperature")
_BLOCK_SIZE = 4096
METHOD_VERSION = "onset-validation-v1"


@dataclass(frozen=True)
class ValidationConfig:
    canonical_split: Path | str
    output_dir: Path | str
    effect_width: int = 3
    bootstraps: int = 10_000
    permutations: int = 10_000
    rewires: int = 100
    rewire_burn_in_sweeps: int = 10
    rewire_thinning_sweeps: int = 2
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("effect_width", "bootstraps", "permutations", "rewires"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rewire_burn_in_sweeps < 0:
            raise ValueError("rewire_burn_in_sweeps must be non-negative")
        if self.rewire_thinning_sweeps < 1:
            raise ValueError("rewire_thinning_sweeps must be positive")


def onset_delta(
    features: np.ndarray, *, start: int, end: int, effect_width: int
) -> np.ndarray:
    """Return onset-window mean minus the equally wide adjacent pre-window."""
    matrix = np.asarray(features, dtype=np.float64)
    width = min(int(effect_width), int(end) - int(start))
    if width < 1 or start < width or start + width > len(matrix):
        raise ValueError("onset window and its pre-window must fit the response")
    return matrix[start : start + width].mean(0) - matrix[start - width : start].mean(0)


def map_pseudo_onset(*, start: int, error_tokens: int, control_tokens: int) -> int:
    """Map a response onset by normalized position, including both endpoints."""
    if error_tokens < 1 or control_tokens < 1:
        raise ValueError("response token counts must be positive")
    if error_tokens == 1:
        return 0
    position = round(start * (control_tokens - 1) / (error_tokens - 1))
    return min(max(position, 0), control_tokens - 1)


def _bootstrap_means(values: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    means = np.empty(draws, dtype=np.float64)
    for offset in range(0, draws, _BLOCK_SIZE):
        size = min(_BLOCK_SIZE, draws - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(1)
    return means


def paired_statistics(
    effects: np.ndarray, *, bootstraps: int, permutations: int, seed: int
) -> dict[str, float | int | None]:
    """Bootstrap confidence intervals and a two-sided paired sign-flip test."""
    values = np.asarray(effects, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("paired statistics require at least one effect")
    rng = np.random.default_rng(seed)
    means = _bootstrap_means(values, int(bootstraps), rng)
    observed = float(values.mean())
    extreme = 0
    for offset in range(0, int(permutations), _BLOCK_SIZE):
        size = min(_BLOCK_SIZE, int(permutations) - offset)
        signs = rng.integers(0, 2, size=(size, values.size)) * 2 - 1
        extreme += int(np.count_nonzero(np.abs((signs * values).mean(1)) >= abs(observed)))
    scale = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "n_pairs": int(values.size),
        "mean_effect": observed,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "sign_flip_p": float((extreme + 1) / (int(permutations) + 1)),
        "dz": float(observed / scale) if scale > 0 else None,
    }


def holm_adjust(p_values: np.ndarray) -> list[float]:
    """Return standard Holm-adjusted p-values in the original feature order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(values)))
    ).clip(max=1.0)
    return adjusted.tolist()


class OnsetValidation:
    """Match exact-stratum controls and estimate paired onset effects."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.dataset = ResearchDataset(
            config.canonical_split, device=config.device, verify_hashes=True
        )
        if self.dataset.manifest["alignment"] != "post_token_query_at_same_position":
            raise ValueError("onset validation requires post_token_query_at_same_position")
        for sample_id, row in self.dataset.rows.items():
            for field in STRATUM_FIELDS:
                if field not in row:
                    raise ValueError(f"index row {sample_id} is missing {field}")
        self.labels = self.dataset.labels()
        self.feature_indices = [STRUCTURAL_FEATURE_NAMES.index(name) for name in PRIMARY_FEATURES]
        self.excluded_initial_events = 0
        self.excluded_initial_event_sample_ids: list[str] = []
        self.unmatched_error_samples: list[str] = []

    def run(self) -> dict:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        matches, events = self._match_samples()
        if not matches:
            raise ValueError("no exact-stratum matched pairs are available")
        pair_rows, event_rows = self._paired_effects(matches)
        primary_rows = self._primary_effects(pair_rows)
        null_rows, topology_test, rewire_metadata = self._rewire_null(matches, pair_rows)

        self._write_csv(output_dir / "matches.csv", matches)
        self._write_csv(output_dir / "pair_effects.csv", pair_rows)
        self._write_csv(output_dir / "event_study.csv", event_rows)
        self._write_csv(output_dir / "primary_effects.csv", primary_rows)
        self._write_csv(output_dir / "rewire_null.csv", null_rows)
        self._plot_event_study(output_dir / "event_study.png", event_rows)

        metadata = {
            "events": events,
            "pairs": len(matches),
            "excluded_initial_events": self.excluded_initial_events,
            "excluded_initial_event_sample_ids": self.excluded_initial_event_sample_ids,
            "unmatched_error_samples": self.unmatched_error_samples,
            "primary_features": list(PRIMARY_FEATURES),
            "effect_width": self.config.effect_width,
            "bootstraps": self.config.bootstraps,
            "permutations": self.config.permutations,
            "rewires": self.config.rewires,
            "seed": self.config.seed,
            "device": self.config.device,
            "alignment": {
                "coordinate_system": "response_relative",
                "effect_width": self.config.effect_width,
                "pseudo_onset": "normalized_start_position",
            },
            "matching": {
                "stratum_fields": list(STRATUM_FIELDS),
                "without_replacement": True,
            },
            "length_cost": (
                "abs(log1p(error_prompt_tokens) - log1p(control_prompt_tokens)) + "
                "abs(log1p(error_response_tokens) - log1p(control_response_tokens))"
            ),
            "rewire": rewire_metadata,
            "topology_test": topology_test,
            "input_provenance": self._input_provenance(),
            "method_version": METHOD_VERSION,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "events": events,
            "pairs": len(matches),
            "matches": matches,
            "primary_effects": primary_rows,
            "topology_test": topology_test,
        }

    def _match_samples(self) -> tuple[list[dict], int]:
        errors_by_stratum: dict[tuple, list[dict]] = {}
        controls_by_stratum: dict[tuple, list[dict]] = {}
        for sample_id in tqdm(self.dataset.sample_ids, desc="scan onset labels", leave=True):
            sample = self.dataset[sample_id]
            attention = sample.attention()
            raw_runs = validate_positive_runs(
                attention.num_response_tokens, self.labels.positive_runs(sample_id)
            )
            runs = merge_positive_runs(raw_runs)
            valid_runs = []
            for start, end in runs:
                width = min(self.config.effect_width, end - start)
                if start < width:
                    self.excluded_initial_events += 1
                    self.excluded_initial_event_sample_ids.append(str(sample_id))
                else:
                    valid_runs.append((start, end))
            summary = {
                "sample_id": str(sample_id),
                "source_id": sample.source_id,
                "prompt_tokens": attention.response_idx,
                "response_tokens": attention.num_response_tokens,
                "runs": valid_runs,
                **{field: getattr(sample, field) for field in STRATUM_FIELDS},
            }
            bucket = errors_by_stratum if raw_runs else controls_by_stratum
            if raw_runs and not valid_runs:
                continue
            bucket.setdefault(self._stratum(sample), []).append(summary)

        matches = []
        for stratum, errors in errors_by_stratum.items():
            controls = controls_by_stratum.get(stratum, [])
            errors.sort(key=lambda row: row["sample_id"])
            controls.sort(key=lambda row: row["sample_id"])
            cost = np.full((len(errors), len(controls) + len(errors)), 1e12, dtype=np.float64)
            for error_index, error in enumerate(errors):
                for control_index, control in enumerate(controls):
                    if self._supports_runs(error, control):
                        cost[error_index, control_index] = self._length_cost(error, control)
                cost[error_index, len(controls) + error_index] = 1e9
            error_indices, control_indices = linear_sum_assignment(cost)
            for error_index, control_index in zip(error_indices, control_indices):
                if control_index >= len(controls):
                    self.unmatched_error_samples.append(errors[error_index]["sample_id"])
                    continue
                if cost[error_index, control_index] >= 1e12:
                    self.unmatched_error_samples.append(errors[error_index]["sample_id"])
                    continue
                error, control = errors[error_index], controls[control_index]
                first_start, first_end = error["runs"][0]
                matches.append(
                    {
                        "error_sample_id": error["sample_id"],
                        "control_sample_id": control["sample_id"],
                        "error_source_id": error["source_id"],
                        "control_source_id": control["source_id"],
                        "run_start": first_start,
                        "run_end": first_end,
                        "merged_runs": json.dumps(error["runs"]),
                        "match_stratum": "exact_metadata_nearest_lengths",
                        "error_prompt_tokens": error["prompt_tokens"],
                        "error_response_tokens": error["response_tokens"],
                        "control_prompt_tokens": control["prompt_tokens"],
                        "control_response_tokens": control["response_tokens"],
                        "length_cost": self._length_cost(error, control),
                        **{field: error[field] for field in STRATUM_FIELDS},
                    }
                )
        matches.sort(key=lambda row: row["error_sample_id"])
        self.excluded_initial_event_sample_ids = sorted(
            set(self.excluded_initial_event_sample_ids)
        )
        self.unmatched_error_samples.sort()
        return matches, sum(len(json.loads(match["merged_runs"])) for match in matches)

    def _paired_effects(self, matches: list[dict]) -> tuple[list[dict], list[dict]]:
        pair_rows = []
        pair_event_rows = []
        for match in tqdm(matches, desc="estimate paired effects", leave=True):
            error_features = self._graph_features(self.dataset[match["error_sample_id"]])
            control_features = self._graph_features(self.dataset[match["control_sample_id"]])
            runs = json.loads(match["merged_runs"])
            error_deltas, control_deltas = [], []
            event_values: dict[tuple[int, str], list[tuple[float, float]]] = {}
            for start, end in runs:
                pseudo_start = map_pseudo_onset(
                    start=start,
                    error_tokens=len(error_features),
                    control_tokens=len(control_features),
                )
                width = min(self.config.effect_width, end - start)
                error_deltas.append(onset_delta(error_features, start=start, end=end, effect_width=width))
                control_deltas.append(
                    onset_delta(
                        control_features,
                        start=pseudo_start,
                        end=pseudo_start + width,
                        effect_width=width,
                    )
                )
                self._collect_event_values(
                    event_values, error_features, control_features, start, pseudo_start
                )

            error_delta = np.mean(error_deltas, axis=0)
            control_delta = np.mean(control_deltas, axis=0)
            for feature, error_value, control_value in zip(
                STRUCTURAL_FEATURE_NAMES, error_delta, control_delta
            ):
                pair_rows.append(
                    {
                        "error_sample_id": match["error_sample_id"],
                        "control_sample_id": match["control_sample_id"],
                        "feature": feature,
                        "error_delta": float(error_value),
                        "control_delta": float(control_value),
                        "effect": float(error_value - control_value),
                        "event_count": len(runs),
                    }
                )
            for (relative_time, feature), values in event_values.items():
                error_values, control_values = np.asarray(values).T
                pair_event_rows.append(
                    {
                        "pair_id": match["error_sample_id"],
                        "relative_time": relative_time,
                        "feature": feature,
                        "error_value": float(error_values.mean()),
                        "control_value": float(control_values.mean()),
                    }
                )
        return pair_rows, self._aggregate_event_values(pair_event_rows)

    def _primary_effects(self, pair_rows: list[dict]) -> list[dict]:
        rows, p_values = [], []
        for feature_index, feature in enumerate(PRIMARY_FEATURES):
            effects = np.asarray([row["effect"] for row in pair_rows if row["feature"] == feature])
            stats = paired_statistics(
                effects,
                bootstraps=self.config.bootstraps,
                permutations=self.config.permutations,
                seed=self.config.seed + feature_index,
            )
            rows.append({"feature": feature, **stats})
            p_values.append(stats["sign_flip_p"])
        for row, adjusted in zip(rows, holm_adjust(np.asarray(p_values))):
            row["holm_p"] = adjusted
        return rows

    def _rewire_null(
        self, matches: list[dict], pair_rows: list[dict]
    ) -> tuple[list[dict], dict, dict]:
        history_index = STRUCTURAL_FEATURE_NAMES.index("history_lag")
        actual = float(np.mean([
            row["effect"] for row in pair_rows if row["feature"] == "history_lag"
        ]))
        effects_by_draw = [[] for _ in range(self.config.rewires)]
        accepted_by_draw = np.zeros(self.config.rewires, dtype=int)
        changed_by_draw = np.zeros(self.config.rewires, dtype=int)
        burn_in_accepted = 0

        with tqdm(
            total=len(matches) * self.config.rewires,
            desc="causal rewire pair-draws",
            leave=True,
        ) as progress:
            for pair_index, match in enumerate(matches):
                error = self.dataset[match["error_sample_id"]]
                control = self.dataset[match["control_sample_id"]]
                error_attention, control_attention = error.attention(), control.attention()
                error_state, control_state = error.original_graph(), control.original_graph()
                burn_seed = self.config.seed + self.config.rewires * len(matches) + pair_index
                error_state, accepted_error = causal_rewire(
                    error_state,
                    seed=2 * burn_seed,
                    sweeps=self.config.rewire_burn_in_sweeps,
                )
                control_state, accepted_control = causal_rewire(
                    control_state,
                    seed=2 * burn_seed + 1,
                    sweeps=self.config.rewire_burn_in_sweeps,
                )
                burn_in_accepted += accepted_error + accepted_control
                runs = json.loads(match["merged_runs"])
                for draw in range(self.config.rewires):
                    previous_error_source = error_state.edge_index[0]
                    previous_control_source = control_state.edge_index[0]
                    draw_seed = self.config.seed + draw * len(matches) + pair_index
                    error_state, accepted_error = causal_rewire(
                        error_state,
                        seed=2 * draw_seed,
                        sweeps=self.config.rewire_thinning_sweeps,
                    )
                    control_state, accepted_control = causal_rewire(
                        control_state,
                        seed=2 * draw_seed + 1,
                        sweeps=self.config.rewire_thinning_sweeps,
                    )
                    error_features = self._features_from_graph(error_attention, error_state)
                    control_features = self._features_from_graph(control_attention, control_state)
                    effects = []
                    for start, end in runs:
                        width = min(self.config.effect_width, end - start)
                        pseudo_start = map_pseudo_onset(
                            start=start,
                            error_tokens=len(error_features),
                            control_tokens=len(control_features),
                        )
                        effects.append(
                            onset_delta(
                                error_features, start=start, end=end, effect_width=width
                            )[history_index]
                            - onset_delta(
                                control_features,
                                start=pseudo_start,
                                end=pseudo_start + width,
                                effect_width=width,
                            )[history_index]
                        )
                    effects_by_draw[draw].append(float(np.mean(effects)))
                    accepted_by_draw[draw] += accepted_error + accepted_control
                    changed = (
                        not torch.equal(error_state.edge_index[0], previous_error_source)
                        or not torch.equal(
                            control_state.edge_index[0], previous_control_source
                        )
                    )
                    changed_by_draw[draw] += int(changed)
                    progress.update()

        rows = [
            {
                "draw": draw,
                "feature": "history_lag",
                "pair_mean_null": float(np.mean(effects_by_draw[draw])),
                "accepted_swaps": int(accepted_by_draw[draw]),
                "changed_pairs": int(changed_by_draw[draw]),
            }
            for draw in range(self.config.rewires)
        ]
        null_values = np.asarray([row["pair_mean_null"] for row in rows])
        changed_draws = int(np.count_nonzero(changed_by_draw))
        rewire_metadata = {
            "draws": self.config.rewires,
            "chain": "continuous",
            "method": "approximate lazy constrained rewire MCMC",
            "burn_in": self.config.rewire_burn_in_sweeps,
            "thinning": self.config.rewire_thinning_sweeps,
            "burn_in_accepted_swaps": int(burn_in_accepted),
            "accepted_swaps": int(burn_in_accepted + accepted_by_draw.sum()),
            "draw_accepted_swaps": int(accepted_by_draw.sum()),
            "changed_draws": changed_draws,
            "changed_pairs": int(changed_by_draw.sum()),
        }
        if changed_draws == 0:
            return rows, {
                "status": "not_estimable",
                "actual": actual,
                "null_center": None,
                "null_q025": None,
                "null_q975": None,
                "excess": None,
                "approximate_randomization_p": None,
            }, rewire_metadata
        center = float(null_values.mean())
        excess = actual - center
        p_value = (np.count_nonzero(np.abs(null_values - center) >= abs(excess)) + 1) / (
            len(null_values) + 1
        )
        return rows, {
            "status": "estimable",
            "actual": actual,
            "null_center": center,
            "null_q025": float(np.quantile(null_values, 0.025)),
            "null_q975": float(np.quantile(null_values, 0.975)),
            "excess": float(excess),
            "approximate_randomization_p": float(p_value),
        }, rewire_metadata

    def _graph_features(self, sample) -> np.ndarray:
        attention = sample.attention()
        return self._features_from_graph(attention, sample.original_graph())

    @staticmethod
    def _features_from_graph(attention, graph) -> np.ndarray:
        features = structural_features_from_relations(attention, relations_from_graph(attention, graph))
        return features.detach().cpu().numpy().astype(np.float64)

    def _supports_runs(self, error: dict, control: dict) -> bool:
        for start, end in error["runs"]:
            width = min(self.config.effect_width, end - start)
            pseudo_start = map_pseudo_onset(
                start=start,
                error_tokens=error["response_tokens"],
                control_tokens=control["response_tokens"],
            )
            if pseudo_start < width or pseudo_start + width > control["response_tokens"]:
                return False
        return True

    @staticmethod
    def _length_cost(error: dict, control: dict) -> float:
        return float(
            abs(np.log1p(error["prompt_tokens"]) - np.log1p(control["prompt_tokens"]))
            + abs(
                np.log1p(error["response_tokens"])
                - np.log1p(control["response_tokens"])
            )
        )

    @staticmethod
    def _stratum(sample) -> tuple:
        return tuple(getattr(sample, field) for field in STRATUM_FIELDS)

    def _input_provenance(self) -> dict:
        root = Path(self.config.canonical_split).resolve()
        manifest = self.dataset.manifest
        provenance = {
            "canonical_split": str(root),
            "manifest_sha256": sha256(root / "manifest.json"),
            "index_sha256": sha256(root / "index.jsonl"),
            "labels_sha256": sha256(root / "labels.jsonl"),
            "attention_floor": manifest["attention_floor"],
            "num_layers": manifest["num_layers"],
            "num_heads": manifest["num_heads"],
        }
        if "observer_model" in manifest:
            provenance["observer_model"] = manifest["observer_model"]
        return provenance

    def _collect_event_values(
        self,
        event_values: dict[tuple[int, str], list[tuple[float, float]]],
        error_features: np.ndarray,
        control_features: np.ndarray,
        start: int,
        pseudo_start: int,
    ) -> None:
        for relative_time in range(-self.config.effect_width, self.config.effect_width):
            error_position = start + relative_time
            control_position = pseudo_start + relative_time
            if error_position < 0 or control_position < 0:
                continue
            if error_position >= len(error_features) or control_position >= len(control_features):
                continue
            for feature, index in zip(PRIMARY_FEATURES, self.feature_indices):
                event_values.setdefault((relative_time, feature), []).append(
                    (error_features[error_position, index], control_features[control_position, index])
                )

    def _aggregate_event_values(self, values: list[dict]) -> list[dict]:
        grouped: dict[tuple[int, str], list[dict]] = {}
        for value in values:
            grouped.setdefault((value["relative_time"], value["feature"]), []).append(value)
        rows = []
        for row_index, ((relative_time, feature), members) in enumerate(sorted(grouped.items())):
            error_values = np.asarray([member["error_value"] for member in members])
            control_values = np.asarray([member["control_value"] for member in members])
            effects = error_values - control_values
            bootstrap = _bootstrap_means(
                effects, self.config.bootstraps, np.random.default_rng(self.config.seed + 100 + row_index)
            )
            rows.append(
                {
                    "relative_time": relative_time,
                    "feature": feature,
                    "pairs": len(members),
                    "mean_error": float(error_values.mean()),
                    "mean_control": float(control_values.mean()),
                    "mean_effect": float(effects.mean()),
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
        return rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _plot_event_study(path: Path, rows: list[dict]) -> None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            len(PRIMARY_FEATURES),
            1,
            figsize=(9, 12),
            sharex=True,
            constrained_layout=True,
        )
        for axis, feature in zip(axes, PRIMARY_FEATURES):
            selected = [row for row in rows if row["feature"] == feature]
            time = np.asarray([row["relative_time"] for row in selected])
            effect = np.asarray([row["mean_effect"] for row in selected])
            low = np.asarray([row["ci_low"] for row in selected])
            high = np.asarray([row["ci_high"] for row in selected])
            axis.plot(time, effect, marker="o")
            axis.fill_between(time, low, high, alpha=0.2)
            axis.axvline(0, color="black", linewidth=1, linestyle="--")
            axis.axhline(0, color="black", linewidth=1)
            axis.set(ylabel=feature)
        axes[-1].set_xlabel("Tokens relative to onset")
        figure.suptitle("Error minus matched-control event study (95% pair bootstrap CI)")
        figure.savefig(path, dpi=180)
        plt.close(figure)
