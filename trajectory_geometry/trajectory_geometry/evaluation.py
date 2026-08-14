"""Frozen Gate-A, label-free route-feature evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


VIEW_NAMES = (
    "nuisance_only",
    "prompt_mass_low",
    "mass_only",
    "dynamics_only",
    "route_embedding",
    "summary_only",
    "full",
)


@dataclass(frozen=True)
class GateAConfig:
    position_bins: int = 5
    length_bins: int = 4
    positions_per_sample: int = 64
    references_per_group: int = 2048
    pca_components: int = 64
    neighbors: int = 20
    seed: int = 20260814
    bootstrap: int = 2000


@dataclass
class FeatureRecord:
    sample_id: str
    source_id: str
    task_type: str
    data_source: str
    source: Path
    feature: Path
    response_idx: int
    length: int


@dataclass
class Detector:
    median: torch.Tensor
    scale: torch.Tensor
    center: torch.Tensor
    components: torch.Tensor
    reference: torch.Tensor
    calibration: torch.Tensor


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _safe_source_metadata(path: Path) -> dict[str, Any]:
    """Read source identifiers only. ``y_token`` is deliberately untouched."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {key: _scalar(payload[key]) for key in ("response_id", "source_id", "task_type", "data_source") if key in payload} | {
                "response_idx": int(np.asarray(payload["response_idx"]).item())
            }
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "response_id": str(payload["response_id"]),
        "source_id": str(payload["source_id"]),
        "task_type": str(payload.get("task_type", "unknown")),
        "data_source": str(payload.get("data_source", "unknown")),
        "response_idx": int(torch.as_tensor(payload["response_idx"]).item()),
    }


def _records(feature_dir: str | Path) -> list[FeatureRecord]:
    feature_dir = Path(feature_dir).resolve()
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["state"] != "complete" or manifest.get("schema") != "trajectory-geometry-route-dynamics-v1":
        raise ValueError("feature manifest is incomplete")
    records: list[FeatureRecord] = []
    for row in tqdm(manifest["records"], desc=f"load {feature_dir.name} metadata", unit="sample"):
        feature = feature_dir / Path(str(row["output"])).name
        source = Path(str(row["source"])).resolve()
        with np.load(feature, allow_pickle=False) as data:
            sample_id = _scalar(data["sample_id"])
            response_idx = int(data["response_idx"])
            length = int(data["route_embedding"].shape[0])
            if int(data["token_count"]) - response_idx != length:
                raise ValueError("feature response count does not align")
        metadata = _safe_source_metadata(source)
        if sample_id != str(metadata["response_id"]) or response_idx != metadata["response_idx"]:
            raise ValueError("feature and source identifiers do not align")
        records.append(FeatureRecord(sample_id, str(metadata["source_id"]), str(metadata["task_type"]), str(metadata["data_source"]), source, feature, response_idx, length))
    return records


def _manifest_signature(feature_dir: str | Path) -> tuple[Any, ...]:
    manifest = json.loads((Path(feature_dir) / "manifest.json").read_text(encoding="utf-8"))
    return tuple(manifest.get(key) for key in ("schema", "embedding_dim", "projection_seed", "prompt_bins", "history_lag_edges"))


def _feature_arrays(record: FeatureRecord) -> dict[str, np.ndarray]:
    with np.load(record.feature, allow_pickle=False) as data:
        mass = np.column_stack([data[name].astype(np.float32) for name in ("prompt_mass", "history_mass", "self_mass", "unresolved_mass")])
        dynamics = np.column_stack([data[name].astype(np.float32) for name in ("temporal_js", "depth_js", "head_js", "route_acceleration")])
        return {"mass": mass, "dynamics": dynamics, "route": data["route_embedding"].astype(np.float32)}


def _sample_positions(length: int, maximum: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=np.int64))


def _length_edges(records: list[FeatureRecord], config: GateAConfig) -> dict[str, tuple[float, float, np.ndarray]]:
    values: dict[str, list[int]] = {}
    for record in records:
        values.setdefault(record.task_type, []).append(record.length)
    return {
        task: (float(np.min(lengths)), float(np.max(lengths)), np.quantile(lengths, np.arange(1, config.length_bins) / config.length_bins))
        for task, lengths in values.items()
    }


def _groups(records: list[FeatureRecord], edges: dict[str, tuple[float, float, np.ndarray]], config: GateAConfig):
    for record in records:
        lower, upper, cuts = edges[record.task_type]
        length_bin = int(np.searchsorted(cuts, np.clip(record.length, lower, upper), side="right"))
        positions = _sample_positions(record.length, config.positions_per_sample)
        position_bin = np.minimum(config.position_bins - 1, positions * config.position_bins // record.length)
        yield record, positions, [(record.task_type, int(pos), length_bin) for pos in position_bin]


def _reservoir_add(store: dict[Any, list[np.ndarray]], seen: dict[Any, int], generators: dict[Any, np.random.Generator], key: Any, value: np.ndarray, *, limit: int, seed: int) -> None:
    count = seen.get(key, 0)
    seen[key] = count + 1
    bucket = store.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append(value)
        return
    generator = generators.setdefault(key, np.random.default_rng(seed + int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")))
    replacement = generator.integers(count + 1)
    if replacement < limit:
        bucket[int(replacement)] = value


def _robust_pca(reference: np.ndarray, config: GateAConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(reference, dtype=torch.float32, device=device)
    median = value.median(dim=0).values
    q1, q3 = torch.quantile(value, torch.tensor([0.25, 0.75], device=device), dim=0)
    scale = (q3 - q1).clamp_min(1e-6)
    normalized = (value - median) / scale
    center = normalized.mean(dim=0)
    normalized = normalized - center
    components = torch.empty((0, normalized.shape[1]), device=device)
    if normalized.shape[1] > config.pca_components:
        covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
        _, vectors = torch.linalg.eigh(covariance)
        components = vectors[:, -config.pca_components :].T
        normalized = normalized @ components.T
    return median, scale, center, components, normalized


def _transform(value: np.ndarray, detector: Detector) -> torch.Tensor:
    result = (torch.as_tensor(value, dtype=torch.float32, device=detector.reference.device) - detector.median) / detector.scale - detector.center
    return result if detector.components.numel() == 0 else result @ detector.components.T


def _knn(query: torch.Tensor, reference: torch.Tensor, neighbors: int, *, leave_one_out: bool = False) -> torch.Tensor:
    distances = torch.cdist(query, reference)
    if leave_one_out:
        distances.fill_diagonal_(float("inf"))
    return distances.topk(neighbors, largest=False).values.mean(dim=1)


def onset_selection_mask(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Keep normal and first-error tokens; drop later tokens in error spans."""
    selected = np.ones(labels.size, dtype=bool)
    for start, stop in zip(offsets, np.append(offsets[1:], labels.size)):
        errors = np.flatnonzero(labels[start:stop])
        if errors.size:
            consecutive = errors[1:][np.diff(errors) == 1]
            selected[start + consecutive] = False
    return selected


def continuation_selection_mask(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Keep all normal tokens and only continuations of positive spans."""
    selected = labels == 0
    for start, stop in zip(offsets, np.append(offsets[1:], labels.size)):
        selected[start + 1 : stop] |= (labels[start + 1 : stop] == 1) & (labels[start : stop - 1] == 1)
    return selected


def token_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": auprc,
        "lift": auprc / prevalence if prevalence else float("nan"),
        "prevalence": prevalence,
    }


def nearest_position_concordance(labels: np.ndarray, scores: np.ndarray, offsets: np.ndarray) -> float:
    values = []
    for start, stop in zip(offsets, np.append(offsets[1:], labels.size)):
        positives = np.flatnonzero(labels[start:stop] == 1)
        negatives = np.flatnonzero(labels[start:stop] == 0)
        if not positives.size or not negatives.size:
            continue
        for position in positives:
            distance = np.abs(negatives - position)
            nearest = negatives[distance == distance.min()]
            values.append(np.mean(scores[start + position] > scores[start + nearest]) + 0.5 * np.mean(scores[start + position] == scores[start + nearest]))
    return float(np.mean(values)) if values else float("nan")


def _prepared_ranking(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    return order, labels[order], np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]


def _batched_weighted_metrics(prepared: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order, labels, starts = prepared
    value = weights[:, order]
    group_pos = np.add.reduceat(value * labels, starts, axis=1)
    group_neg = np.add.reduceat(value * (1 - labels), starts, axis=1)
    positive, negative = group_pos.sum(axis=1), group_neg.sum(axis=1)
    tp_before = np.concatenate([np.zeros((value.shape[0], 1)), np.cumsum(group_pos, axis=1)[:, :-1]], axis=1)
    fp_before = np.concatenate([np.zeros((value.shape[0], 1)), np.cumsum(group_neg, axis=1)[:, :-1]], axis=1)
    auroc = np.sum(group_neg * (tp_before + 0.5 * group_pos), axis=1) / (positive * negative)
    denominator = tp_before + fp_before + group_pos + group_neg
    precision = np.divide(tp_before + group_pos, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return auroc, np.sum(group_pos * precision, axis=1) / positive


def bootstrap_contrasts(labels: np.ndarray, score_views: dict[str, np.ndarray], source_ids: np.ndarray, *, baselines: tuple[str, ...], draws: np.ndarray, batch_size: int = 32, progress: bool = False) -> dict[str, dict[str, dict[str, float | int]]]:
    unique, inverse = np.unique(source_ids, return_inverse=True)
    prepared = {name: _prepared_ranking(labels, score) for name, score in score_views.items()}
    sampled = {name: [[], []] for name in score_views}
    starts = range(0, draws.shape[0], batch_size)
    for start in tqdm(starts, disable=not progress, desc="cluster bootstrap", unit="batch"):
        batch = draws[start : start + batch_size]
        counts = np.zeros((batch.shape[0], unique.size), dtype=np.float64)
        np.add.at(counts, (np.arange(batch.shape[0])[:, None], batch), 1)
        weights = counts[:, inverse]
        for name in score_views:
            auroc, auprc = _batched_weighted_metrics(prepared[name], weights)
            sampled[name][0].append(auroc)
            sampled[name][1].append(auprc)
    metrics = {name: (np.concatenate(values[0]), np.concatenate(values[1])) for name, values in sampled.items()}
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    point = {name: token_metrics(labels, score) for name, score in score_views.items()}
    for baseline in baselines:
        result[baseline] = {}
        for index, metric in enumerate(("auroc", "auprc")):
            delta = metrics["full"][index] - metrics[baseline][index]
            valid = np.isfinite(delta)
            result[baseline][metric] = {"point": point["full"][metric] - point[baseline][metric], "ci_low": float(np.quantile(delta[valid], 0.025)), "ci_high": float(np.quantile(delta[valid], 0.975)), "valid_replicates": int(valid.sum())}
    return result


def bootstrap_delta(labels: np.ndarray, full: np.ndarray, baseline: np.ndarray, source_ids: np.ndarray, *, n_boot: int | None = None, seed: int | None = None, draws: np.ndarray | None = None) -> dict[str, dict[str, float | int]]:
    if draws is None:
        if n_boot is None or seed is None:
            raise ValueError("n_boot and seed are required when draws are absent")
        draws = np.random.default_rng(seed).integers(0, np.unique(source_ids).size, size=(n_boot, np.unique(source_ids).size))
    return bootstrap_contrasts(labels, {"full": full, "baseline": baseline}, source_ids, baselines=("baseline",), draws=draws)["baseline"]


class GateAEvaluator:
    def __init__(self, config: GateAConfig = GateAConfig(), device: str = "cuda") -> None:
        self.config = config
        self.device = torch.device(device)

    def low_prompt_score(self, reference: np.ndarray, value: np.ndarray) -> np.ndarray:
        sorted_reference = np.sort(reference[:, 0])
        return 1.0 - np.searchsorted(sorted_reference, value[:, 0], side="right") / sorted_reference.size

    def _fit_detectors(self, train: list[FeatureRecord], edges: dict[str, tuple[float, float, np.ndarray]]) -> tuple[dict[str, dict[tuple[str, int, int] | str, Detector]], dict[str, dict[tuple[str, int, int], np.ndarray]]]:
        collected: dict[str, dict[Any, list[np.ndarray]]] = {name: {} for name in ("nuisance_only", "mass_only", "dynamics_only", "route_embedding")}
        seen: dict[str, dict[Any, int]] = {name: {} for name in collected}
        generators: dict[str, dict[Any, np.random.Generator]] = {name: {} for name in collected}
        prompt_reference: dict[tuple[str, int, int], list[np.ndarray]] = {}
        prompt_seen: dict[tuple[str, int, int], int] = {}
        prompt_generators: dict[tuple[str, int, int], np.random.Generator] = {}
        for record, positions, groups in tqdm(_groups(train, edges, self.config), total=len(train), desc="fit Gate-A references", unit="sample"):
            arrays = _feature_arrays(record)
            nuisance = np.column_stack([positions / max(1, record.length - 1), np.full(positions.size, np.log1p(record.length))])
            values = {"nuisance_only": nuisance, "mass_only": arrays["mass"][positions], "dynamics_only": arrays["dynamics"][positions], "route_embedding": arrays["route"][positions]}
            for name, value in values.items():
                keys = [record.task_type] * positions.size if name == "nuisance_only" else groups
                for key, row in zip(keys, value):
                    _reservoir_add(collected[name], seen[name], generators[name], key, row, limit=self.config.references_per_group, seed=self.config.seed)
            for group, value in zip(groups, arrays["mass"][positions, :1]):
                _reservoir_add(prompt_reference, prompt_seen, prompt_generators, group, value, limit=self.config.references_per_group, seed=self.config.seed)
        detectors: dict[str, dict[Any, Detector]] = {name: {} for name in collected}
        for name, by_group in collected.items():
            for group, rows in by_group.items():
                value = np.asarray(rows, dtype=np.float32)
                if value.shape[0] <= self.config.neighbors:
                    raise ValueError(f"group {group} has too few train references")
                median, scale, center, components, reference = _robust_pca(value, self.config, self.device)
                loo = _knn(reference, reference, self.config.neighbors, leave_one_out=True)
                detectors[name][group] = Detector(median, scale, center, components, reference, torch.sort(loo).values)
        return detectors, {group: np.sort(np.asarray(value, dtype=np.float32).reshape(-1)) for group, value in prompt_reference.items()}

    def _score_record(self, record: FeatureRecord, edges: dict[str, tuple[float, float, np.ndarray]], detectors: dict[str, dict[Any, Detector]], prompt_reference: dict[tuple[str, int, int], np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        arrays = _feature_arrays(record)
        positions = np.arange(record.length)
        lower, upper, cuts = edges[record.task_type]
        length_bin = int(np.searchsorted(cuts, np.clip(record.length, lower, upper), side="right"))
        groups = [(record.task_type, int(min(self.config.position_bins - 1, position * self.config.position_bins // record.length)), length_bin) for position in positions]
        nuisance = np.column_stack([positions / max(1, record.length - 1), np.full(record.length, np.log1p(record.length))])
        values = {"nuisance_only": nuisance, "mass_only": arrays["mass"], "dynamics_only": arrays["dynamics"], "route_embedding": arrays["route"]}
        result: dict[str, np.ndarray] = {}
        for name, value in values.items():
            output = np.empty(record.length, dtype=np.float32)
            keys = [record.task_type] * record.length if name == "nuisance_only" else groups
            for key in set(keys):
                rows = np.asarray([index for index, group in enumerate(keys) if group == key])
                detector = detectors[name][key]
                distance = _knn(_transform(value[rows], detector), detector.reference, self.config.neighbors)
                calibrated = torch.searchsorted(detector.calibration, distance, right=True)
                output[rows] = (calibrated.float() / detector.calibration.numel()).cpu().numpy()
            result[name] = output
        result["prompt_mass_low"] = np.asarray([1.0 - np.searchsorted(prompt_reference[group], arrays["mass"][index, 0], side="right") / prompt_reference[group].size for index, group in enumerate(groups)], dtype=np.float32)
        result["summary_only"] = (result["mass_only"] + result["dynamics_only"]) / 2
        result["full"] = (result["mass_only"] + result["dynamics_only"] + result["route_embedding"]) / 3
        metadata = {"position": positions, "length": np.full(record.length, record.length)}
        return result, metadata

    def score(self, train_dir: str | Path, test_dir: str | Path, output_dir: str | Path) -> tuple[list[FeatureRecord], list[FeatureRecord], Path]:
        if _manifest_signature(train_dir) != _manifest_signature(test_dir):
            raise ValueError("train and test feature manifests disagree")
        train, test = _records(train_dir), _records(test_dir)
        if {record.source_id for record in train} & {record.source_id for record in test}:
            raise ValueError("train and test source_id sets overlap")
        edges = _length_edges(train, self.config)
        detectors, prompt_reference = self._fit_detectors(train, edges)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path = output_dir / "detector_state.pt"
        temporary_state = state_path.with_suffix(".tmp.pt")
        cpu_detectors = {name: {group: Detector(*(getattr(detector, field).cpu() for field in ("median", "scale", "center", "components", "reference", "calibration"))) for group, detector in values.items()} for name, values in detectors.items()}
        torch.save({"config": self.config.__dict__, "length_edges": edges, "detectors": cpu_detectors, "prompt_reference": prompt_reference}, temporary_state)
        temporary_state.replace(state_path)
        payload: dict[str, list[np.ndarray]] = {key: [] for key in ("sample_id", "source_id", "task_type", "data_source", "position", "length")}
        payload.update({f"score_{name}": [] for name in VIEW_NAMES})
        for record in tqdm(test, desc="score Gate-A", unit="sample"):
            scores, metadata = self._score_record(record, edges, detectors, prompt_reference)
            payload["sample_id"].append(np.full(record.length, record.sample_id))
            payload["source_id"].append(np.full(record.length, record.source_id))
            payload["task_type"].append(np.full(record.length, record.task_type))
            payload["data_source"].append(np.full(record.length, record.data_source))
            payload["position"].append(metadata["position"])
            payload["length"].append(metadata["length"])
            for name in VIEW_NAMES:
                payload[f"score_{name}"].append(scores[name])
        target = output_dir / "scores_label_free.npz"
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **{key: np.concatenate(value) for key, value in payload.items()})
        temporary.replace(target)
        return train, test, target

    def _labels(self, test: list[FeatureRecord], score_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(score_path, allow_pickle=False) as scores:
            expected_ids = scores["sample_id"]
        labels, offsets, ids = [], [0], []
        for record in tqdm(test, desc="read evaluation labels", unit="sample"):
            payload = torch.load(record.source, map_location="cpu", weights_only=True)
            if str(payload["response_id"]) != record.sample_id or str(payload["source_id"]) != record.source_id or str(payload.get("task_type", "unknown")) != record.task_type or str(payload.get("data_source", "unknown")) != record.data_source or int(torch.as_tensor(payload["response_idx"]).item()) != record.response_idx:
                raise ValueError("test source metadata changed after label-free scoring")
            value = torch.as_tensor(payload["y_token"]).flatten().cpu().numpy()
            if value.size != record.response_idx + record.length:
                raise ValueError("response labels do not align with feature rows")
            if np.any((value != 0) & (value != 1)) or np.any(value[: record.response_idx] != 0):
                raise ValueError("y_token must be binary with a normal prompt prefix")
            labels.append(value[record.response_idx:].astype(np.int8))
            ids.append(np.full(record.length, record.sample_id))
            offsets.append(offsets[-1] + record.length)
        if not np.array_equal(np.concatenate(ids), expected_ids):
            raise ValueError("label rows do not align with score rows")
        return np.concatenate(labels), np.asarray(offsets[:-1])

    def evaluate(self, train_dir: str | Path, test_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        train, test, score_path = self.score(train_dir, test_dir, output_dir)
        labels, offsets = self._labels(test, score_path)
        with np.load(score_path, allow_pickle=False) as data:
            sources, tasks = data["source_id"], data["task_type"]
            onset_rows = onset_selection_mask(labels, offsets)
            continuation_rows = continuation_selection_mask(labels, offsets)
            report: dict[str, Any] = {"schema": "trajectory-geometry-gate-a-v1", "scores": str(score_path), "views": {}, "topology": "not_tested_missing_rewired_features"}
            for name in VIEW_NAMES:
                scores = data[f"score_{name}"]
                view = {
                    "overall": token_metrics(labels, scores),
                    "onset": token_metrics(labels[onset_rows], scores[onset_rows]),
                    "continuation": token_metrics(labels[continuation_rows], scores[continuation_rows]) if len(np.unique(labels[continuation_rows])) == 2 else None,
                    "nearest_position_matched_concordance": nearest_position_concordance(labels, scores, offsets),
                }
                per_sample = [token_metrics(labels[start:stop], scores[start:stop])["auroc"] for start, stop in zip(offsets, np.append(offsets[1:], labels.size)) if len(np.unique(labels[start:stop])) == 2]
                view["within_response_macro_auroc"] = float(np.mean(per_sample)) if per_sample else float("nan")
                view["by_task"] = {
                    str(task): {
                        "overall": token_metrics(labels[tasks == task], scores[tasks == task]),
                        "onset": token_metrics(
                            labels[(tasks == task) & onset_rows],
                            scores[(tasks == task) & onset_rows],
                        ),
                    }
                    for task in np.unique(tasks)
                    if len(np.unique(labels[tasks == task])) == 2
                    and len(np.unique(labels[(tasks == task) & onset_rows])) == 2
                }
                report["views"][name] = view
            full = data["score_full"]
            draws = np.random.default_rng(self.config.seed).integers(0, np.unique(sources).size, size=(self.config.bootstrap, np.unique(sources).size))
            report["contrasts"] = bootstrap_contrasts(
                labels,
                {
                    "full": full,
                    "prompt_mass_low": data["score_prompt_mass_low"],
                    "nuisance_only": data["score_nuisance_only"],
                    "summary_only": data["score_summary_only"],
                },
                sources,
                baselines=("prompt_mass_low", "nuisance_only", "summary_only"),
                draws=draws,
                progress=True,
            )
        full_metrics = report["views"]["full"]["overall"]
        def positive(contrast: str) -> bool:
            return all(item["point"] > 0 and item["ci_low"] > 0 for item in report["contrasts"][contrast].values())
        report["claims"] = {"C1": bool(full_metrics["auroc"] > 0.5 and full_metrics["auprc"] > full_metrics["prevalence"] and report["views"]["full"]["within_response_macro_auroc"] > 0.5 and positive("prompt_mass_low") and positive("nuisance_only")), "C2": positive("summary_only")}
        output_dir = Path(output_dir)
        temporary_results = output_dir / "results.json.tmp"
        temporary_results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary_results.replace(output_dir / "results.json")
        full_view = report["views"]["full"]
        overall = full_view["overall"]
        onset = full_view["onset"]
        continuation = full_view["continuation"]
        continuation_text = "n/a" if continuation is None else f"AUROC={continuation['auroc']:.4f} AUPRC={continuation['auprc']:.4f}"
        lines = [
            f"overall tokens={overall['n']} positives={overall['positives']} prevalence={overall['prevalence']:.4f}",
            f"topology={report['topology']}",
            "view                  AUROC   AUPRC   lift",
            *[
                f"{name:20} {report['views'][name]['overall']['auroc']:.4f}  "
                f"{report['views'][name]['overall']['auprc']:.4f}  "
                f"{report['views'][name]['overall']['lift']:.2f}x"
                for name in VIEW_NAMES
            ],
            f"full onset AUROC={onset['auroc']:.4f} AUPRC={onset['auprc']:.4f}; continuation {continuation_text}",
            f"full within-response AUROC={full_view['within_response_macro_auroc']:.4f}; matched concordance={full_view['nearest_position_matched_concordance']:.4f}",
            *[
                f"full-{baseline} dAUROC={contrast['auroc']['point']:.4f} "
                f"CI=[{contrast['auroc']['ci_low']:.4f},{contrast['auroc']['ci_high']:.4f}] "
                f"dAUPRC={contrast['auprc']['point']:.4f} "
                f"CI=[{contrast['auprc']['ci_low']:.4f},{contrast['auprc']['ci_high']:.4f}]"
                for baseline, contrast in report["contrasts"].items()
            ],
            f"C1={report['claims']['C1']}: full beats chance/prevalence and both prompt/nuisance baselines, with within-response AUROC > .5.",
            f"C2={report['claims']['C2']}: full adds signal beyond calibrated mass+dynamics summary.",
            f"scores: {score_path}",
            f"results: {output_dir / 'results.json'}",
        ]
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report
