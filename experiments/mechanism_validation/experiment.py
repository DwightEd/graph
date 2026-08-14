"""Runnable, post-hoc validation workflows for saved attention mechanisms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from experiments.mechanism_validation.evaluation import (
    cluster_bootstrap,
    paired_cluster_delta,
    ranking_metrics,
    same_response_effect,
)
from experiments.mechanism_validation.graph_ablation import (
    apply_trace_variant,
    extract_traces,
    fixed_graph_descriptors,
)
from research_dataset import open_research_dataset


@dataclass
class FeatureSplit:
    values: np.ndarray
    valid: np.ndarray
    sample_ids: np.ndarray
    source_ids: np.ndarray
    positions: np.ndarray
    prompt_lengths: np.ndarray
    response_lengths: np.ndarray
    task_types: np.ndarray
    data_sources: np.ndarray
    names: list[str]
    family_slices: dict[str, slice]
    inventory: dict[str, tuple[str, int]]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empty_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output_dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _family_slices(metadata: dict) -> dict[str, slice]:
    return {name: slice(*bounds) for name, bounds in metadata["family_slices"].items()}


def load_feature_split(directory: str | Path, *, max_tokens: int | None = None, seed: int = 0) -> FeatureSplit:
    """Load compact per-response tensors without consulting a dataset or labels."""
    directory = Path(directory)
    metadata = _read_json(directory / "metadata.json")
    if metadata.get("schema") != "mechanism_features.v3" or metadata.get("labels_included") is not False:
        raise ValueError("feature metadata must exclude labels")
    files = sorted(path for path in directory.glob("*.pt") if path.name != "metadata.pt")
    if not files:
        raise ValueError("feature directory contains no sample artifacts")
    index_path = directory / "index.json"
    if not index_path.exists():
        raise ValueError("feature index is required")
    index = _read_json(index_path)["samples"]
    inventory = {str(row["sample_id"]): (str(row["source_id"]), int(row["tokens"])) for row in index}
    if len(inventory) != len(index) or set(inventory) != {path.stem for path in files}:
        raise ValueError("feature index and files must match exactly")
    total = sum(tokens for _, tokens in inventory.values())
    selected = None
    if max_tokens is not None and total > max_tokens:
        selected = np.sort(np.random.default_rng(seed).choice(total, size=max_tokens, replace=False))
    values = []
    valid = []
    sample_ids = []
    source_ids = []
    positions = []
    prompt_lengths = []
    response_lengths = []
    task_types = []
    data_sources = []
    expected_width = len(metadata["feature_names"])
    offset = 0
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=True)
        source_id, indexed_count = inventory[path.stem]
        if str(item["sample_id"]) != path.stem or str(item["source_id"]) != source_id or len(item["values"]) != indexed_count:
            raise ValueError("feature payload and index disagree")
        if "valid" not in item:
            raise ValueError(f"feature artifact lacks valid mask: {path}")
        tensor, mask = item["values"], item["valid"]
        if tensor.ndim != 2 or tensor.shape != mask.shape or tensor.shape[1] != expected_width:
            raise ValueError(f"invalid feature artifact: {path}")
        original_count = len(tensor)
        if selected is not None:
            local = selected[(selected >= offset) & (selected < offset + original_count)] - offset
            tensor, mask = tensor[local], mask[local]
            positions_for_item = local.tolist()
        else:
            positions_for_item = list(range(original_count))
        offset += original_count
        count = len(tensor)
        values.append(tensor.float().numpy())
        valid.append(mask.bool().numpy())
        sample_ids.extend([str(item["sample_id"])] * count)
        source_ids.extend([str(item["source_id"])] * count)
        positions.extend(positions_for_item)
        prompt_lengths.extend([int(item.get("prompt_length", 0))] * count)
        response_lengths.extend([original_count] * count)
        task_types.extend([str(item.get("task_type"))] * count)
        data_sources.extend([str(item.get("data_source"))] * count)
    return FeatureSplit(
        np.concatenate(values), np.concatenate(valid), np.asarray(sample_ids), np.asarray(source_ids),
        np.asarray(positions), np.asarray(prompt_lengths), np.asarray(response_lengths),
        np.asarray(task_types), np.asarray(data_sources),
        list(metadata["feature_names"]), _family_slices(metadata), inventory,
    )


def _load_labels(split_root: str | Path, features: FeatureSplit) -> np.ndarray:
    """This is deliberately called only after all label-free artifacts are loaded."""
    dataset = open_research_dataset(split_root, device="cpu", retain_embedded_labels=True)
    samples = {}
    # Formal caches seal embedded labels until every sample has been visited.
    # Release each sparse attention payload immediately; only label vectors stay cached.
    for sample in dataset:
        attention = sample.attention()
        samples[sample.sample_id] = (str(sample.source_id), attention.num_response_tokens)
        sample.release_attention()
    if set(samples) != set(features.inventory):
        raise ValueError("dataset and artifact sample inventories differ")
    for sample_id, (source_id, _) in features.inventory.items():
        if samples[sample_id][0] != source_id:
            raise ValueError("artifact source_id does not match dataset")
    labels = dataset.labels()
    label_map = {}
    feature_sample_ids = set(features.inventory)
    for sample_id in feature_sample_ids:
        sample = dataset[sample_id]
        response_labels = labels.response_labels(sample).detach().cpu().numpy()
        if len(response_labels) != samples[sample_id][1] or len(response_labels) != features.inventory[sample_id][1]:
            raise ValueError("label response length does not match feature artifact")
        label_map[sample_id] = response_labels
        sample.release_attention()
    if not set(features.sample_ids).issubset(label_map):
        raise ValueError("feature sample is absent from label split")
    return np.asarray(
        [label_map[str(sample_id)][int(position)] for sample_id, position in zip(features.sample_ids, features.positions)],
        dtype=np.int64,
    )


def _numeric_nuisance(features: FeatureSplit) -> list[np.ndarray]:
    q = features.positions / np.maximum(features.response_lengths - 1, 1)
    return [q, q ** 2, q ** 3, np.log1p(features.positions), np.log1p(features.prompt_lengths), np.log1p(features.response_lengths)]


def _nuisance_pair(train: FeatureSplit, test: FeatureSplit) -> tuple[np.ndarray, np.ndarray]:
    train_columns = _numeric_nuisance(train)
    test_columns = _numeric_nuisance(test)
    for train_values, test_values in (
        (train.task_types, test.task_types),
        (train.data_sources, test.data_sources),
    ):
        for category in np.unique(train_values)[1:]:
            train_columns.append(train_values == category)
            test_columns.append(test_values == category)
    return (
        np.column_stack(train_columns).astype(np.float32),
        np.column_stack(test_columns).astype(np.float32),
    )


def _standardized_nuisance(train: FeatureSplit, test: FeatureSplit) -> tuple[np.ndarray, np.ndarray]:
    train_values, test_values = _nuisance_pair(train, test)
    mean, scale = train_values.mean(0), train_values.std(0)
    return (train_values - mean) / np.where(scale > 0, scale, 1.0), (test_values - mean) / np.where(scale > 0, scale, 1.0)


def _fit_preprocessor(train_values: np.ndarray, train_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(train_valid, train_values, np.nan)
    medians = np.nanmedian(masked, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    complete = np.where(train_valid, train_values, medians)
    scale = complete.std(axis=0)
    return medians, np.where(scale > 0, scale, 1.0)


def _transform(values: np.ndarray, valid: np.ndarray, medians: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.where(valid, values, medians) - medians) / scale


def _sample_train(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, size=maximum, replace=False))


def _overlap_audit(train: FeatureSplit, test: FeatureSplit) -> dict:
    train_samples, test_samples = set(train.inventory), set(test.inventory)
    train_sources = {source for source, _ in train.inventory.values()}
    test_sources = {source for source, _ in test.inventory.values()}
    return {
        "train_samples": len(train_samples), "test_samples": len(test_samples),
        "overlapping_samples": len(train_samples & test_samples),
        "train_sources": len(train_sources), "test_sources": len(test_sources),
        "overlapping_sources": len(train_sources & test_sources),
    }


def _probe(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y[0]) if len(train_y) else 0.0)
    return LogisticRegression(class_weight="balanced", max_iter=1000).fit(train_x, train_y).predict_proba(test_x)[:, 1]


def _metric_with_control(labels, scores, split: FeatureSplit, bootstrap: int, seed: int) -> dict:
    result = ranking_metrics(labels, scores)
    result["same_response"] = same_response_effect(labels, scores, split.sample_ids, split.positions)
    if bootstrap:
        sampled = cluster_bootstrap(labels, scores, split.source_ids, n_resamples=bootstrap, seed=seed)
        result["bootstrap_95"] = {
            "auroc": np.nanquantile(sampled["auroc"], [.025, .975]).tolist(),
            "auprc": np.nanquantile(sampled["auprc"], [.025, .975]).tolist(),
            "valid_replicates": sampled["valid_replicates"],
        }
    return result


def _diagnostic_results(train: FeatureSplit, train_labels: np.ndarray, test: FeatureSplit, test_labels: np.ndarray,
                        *, seed: int, maximum: int) -> tuple[dict, dict[str, np.ndarray]]:
    medians, scale = _fit_preprocessor(train.values, train.valid)
    train_mechanisms = _transform(train.values, train.valid, medians, scale)
    test_mechanisms = _transform(test.values, test.valid, medians, scale)
    nuisance_train, nuisance_test = _standardized_nuisance(train, test)
    train_index = _sample_train(np.arange(len(train_labels)), maximum, seed)
    sets = {"nuisance_only": np.empty(0, dtype=np.int64), "full": np.arange(train_mechanisms.shape[1])}
    if set(train.family_slices) == {"node", "graph"}:
        sets["node_only"] = np.arange(train.family_slices["node"].start, train.family_slices["node"].stop)
        sets["graph_only"] = np.arange(train.family_slices["graph"].start, train.family_slices["graph"].stop)
    else:
        for family, selection in train.family_slices.items():
            own = np.arange(selection.start, selection.stop)
            sets[f"single:{family}"] = own
            sets[f"leave_out:{family}"] = np.setdiff1d(sets["full"], own)
    metrics, scores = {}, {}
    for name, columns in tqdm(sets.items(), desc="mechanism probes", unit="probe"):
        train_x = np.column_stack((nuisance_train, train_mechanisms[:, columns]))
        test_x = np.column_stack((nuisance_test, test_mechanisms[:, columns]))
        score = _probe(train_x[train_index], train_labels[train_index], test_x)
        scores[name] = score
        metrics[name] = _metric_with_control(test_labels, score, test, 0, seed)
    return metrics, scores


def _fit_graph_decoder(train: FeatureSplit, train_labels: np.ndarray, columns: np.ndarray):
    medians, scale = _fit_preprocessor(train.values, train.valid)
    train_mechanisms = _transform(train.values, train.valid, medians, scale)
    nuisance_train, _ = _nuisance_pair(train, train)
    nuisance_mean, nuisance_scale = nuisance_train.mean(0), nuisance_train.std(0)
    nuisance_scale = np.where(nuisance_scale > 0, nuisance_scale, 1.0)
    train_x = np.column_stack(((nuisance_train - nuisance_mean) / nuisance_scale, train_mechanisms[:, columns]))
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(train_x, train_labels)
    return model, medians, scale, nuisance_mean, nuisance_scale, columns


def _apply_graph_decoder(decoder, train: FeatureSplit, test: FeatureSplit) -> np.ndarray:
    model, medians, scale, nuisance_mean, nuisance_scale, columns = decoder
    _, nuisance_test = _nuisance_pair(train, test)
    mechanisms = _transform(test.values, test.valid, medians, scale)
    test_x = np.column_stack(((nuisance_test - nuisance_mean) / nuisance_scale, mechanisms[:, columns]))
    return model.predict_proba(test_x)[:, 1]


def evaluate_mechanisms(train_split, train_features, test_split, test_features, output_dir, *, bootstrap=200, seed=0,
                        max_train_tokens=100000) -> dict:
    output = _empty_output(output_dir)
    train_metadata, test_metadata = _read_json(Path(train_features) / "metadata.json"), _read_json(Path(test_features) / "metadata.json")
    for key in ("schema", "ema_decay", "attention_floor", "feature_names", "family_slices"):
        if train_metadata.get(key) != test_metadata.get(key):
            raise ValueError(f"train/test feature metadata differ: {key}")
    train = load_feature_split(train_features, max_tokens=max_train_tokens, seed=seed)
    test = load_feature_split(test_features)
    train_labels = _load_labels(train_split, train)
    test_labels = _load_labels(test_split, test)
    metrics, scores = _diagnostic_results(train, train_labels, test, test_labels, seed=seed,
                                          maximum=max_train_tokens)
    nuisance_metrics = metrics["nuisance_only"]
    adjusted_global_mean = {}
    nuisance_train, nuisance_test = _standardized_nuisance(train, test)
    train_index = _sample_train(np.arange(len(train_labels)), max_train_tokens, seed)
    for column, name in tqdm(list(enumerate(train.names)), desc="adjusted global means", unit="feature"):
        if not name.endswith(":global_mean"):
            continue
        medians, scale = _fit_preprocessor(train.values[:, [column]], train.valid[:, [column]])
        train_x = np.column_stack((nuisance_train, _transform(train.values[:, [column]], train.valid[:, [column]], medians, scale)))
        test_x = np.column_stack((nuisance_test, _transform(test.values[:, [column]], test.valid[:, [column]], medians, scale)))
        metric = _metric_with_control(test_labels, _probe(train_x[train_index], train_labels[train_index], test_x), test, 0, seed)
        adjusted_global_mean[name] = {"held_out": metric, "point_delta": {key: metric[key] - nuisance_metrics[key] for key in ("auroc", "auprc")}}
    univariate = {}
    medians, scale = _fit_preprocessor(train.values, train.valid)
    train_values = _transform(train.values, train.valid, medians, scale)
    test_values = _transform(test.values, test.valid, medians, scale)
    for column, name in tqdm(list(enumerate(train.names)), desc="univariate mechanisms", unit="feature"):
        train_valid = train.valid[:, column]
        test_valid = test.valid[:, column]
        positive = train_values[(train_labels == 1) & train_valid, column]
        negative = train_values[(train_labels == 0) & train_valid, column]
        direction = float(positive.mean() - negative.mean()) if len(positive) and len(negative) else 1.0
        selected = np.flatnonzero(test_valid)
        subset = FeatureSplit(
            test.values[selected], test.valid[selected], test.sample_ids[selected], test.source_ids[selected],
            test.positions[selected], test.prompt_lengths[selected], test.response_lengths[selected],
            test.task_types[selected], test.data_sources[selected], test.names, test.family_slices, test.inventory,
        )
        raw_scores = test_values[selected, column]
        oriented_scores = raw_scores * np.sign(direction or 1)
        univariate[name] = {
            "train_direction": "higher" if direction >= 0 else "lower",
            "raw_test": _metric_with_control(test_labels[selected], raw_scores, subset, 0, seed),
            "train_oriented_test": _metric_with_control(
                test_labels[selected], oriented_scores, subset,
                bootstrap if name.endswith(":global_mean") else 0, seed,
            ),
        }
    np.savez_compressed(output / "predictions.npz", labels=test_labels, sample_ids=test.sample_ids, positions=test.positions,
                        **{f"probe_{name}": score for name, score in scores.items()})
    result = {"analysis_status": "post_hoc_exploratory", "probe_uses_labels": True,
              "train_tokens": len(train_labels), "test_tokens": len(test_labels),
              "overlap_audit": _overlap_audit(train, test),
              "cache_bound_audit": {"train": {key: train_metadata[key] for key in ("cache_bound_invalid_rows", "cache_bound_total_rows", "cache_bound_invalid_fraction")}, "test": {key: test_metadata[key] for key in ("cache_bound_invalid_rows", "cache_bound_total_rows", "cache_bound_invalid_fraction")}},
              "univariate": univariate, "supervised_diagnostic": metrics, "adjusted_global_mean": adjusted_global_mean}
    _write_json(output / "results.json", result)
    return result


def evaluate_lookback(train_split, train_features, test_split, test_features,
                      output_dir, *, bootstrap=200, seed=0,
                      max_train_tokens=100000) -> dict:
    """Evaluate only the corrected scalar Lookback ratio as a post-hoc diagnostic."""
    output = _empty_output(output_dir)
    train_metadata = _read_json(Path(train_features) / "metadata.json")
    test_metadata = _read_json(Path(test_features) / "metadata.json")
    for key in ("schema", "ema_decay", "attention_floor", "feature_names", "family_slices"):
        if train_metadata.get(key) != test_metadata.get(key):
            raise ValueError(f"train/test feature metadata differ: {key}")
    train = load_feature_split(
        train_features, max_tokens=max_train_tokens, seed=seed
    )
    test = load_feature_split(test_features)
    feature = "retained_length_normalized_lookback:global_mean"
    column = train.names.index(feature)
    train_labels = _load_labels(train_split, train)
    test_labels = _load_labels(test_split, test)

    medians, scale = _fit_preprocessor(
        train.values[:, [column]], train.valid[:, [column]]
    )
    train_lookback = _transform(
        train.values[:, [column]], train.valid[:, [column]], medians, scale
    )[:, 0]
    test_lookback = _transform(
        test.values[:, [column]], test.valid[:, [column]], medians, scale
    )[:, 0]
    train_valid = train.valid[:, column]
    test_valid = test.valid[:, column]
    positive = train_lookback[(train_labels == 1) & train_valid]
    negative = train_lookback[(train_labels == 0) & train_valid]
    direction = float(positive.mean() - negative.mean())
    selected = np.flatnonzero(test_valid)
    test_subset = FeatureSplit(
        test.values[selected], test.valid[selected], test.sample_ids[selected],
        test.source_ids[selected], test.positions[selected],
        test.prompt_lengths[selected], test.response_lengths[selected],
        test.task_types[selected], test.data_sources[selected], test.names,
        test.family_slices, test.inventory,
    )
    raw = _metric_with_control(
        test_labels[selected], test_lookback[selected], test_subset, 0, seed
    )
    oriented = _metric_with_control(
        test_labels[selected], test_lookback[selected] * np.sign(direction or 1),
        test_subset, bootstrap, seed,
    )

    nuisance_train, nuisance_test = _standardized_nuisance(train, test)
    train_index = _sample_train(
        np.arange(len(train_labels)), max_train_tokens, seed
    )
    nuisance_score = _probe(
        nuisance_train[train_index], train_labels[train_index], nuisance_test
    )
    adjusted_score = _probe(
        np.column_stack((nuisance_train, train_lookback))[train_index],
        train_labels[train_index],
        np.column_stack((nuisance_test, test_lookback)),
    )
    nuisance = _metric_with_control(
        test_labels, nuisance_score, test, 0, seed
    )
    adjusted = _metric_with_control(
        test_labels, adjusted_score, test, bootstrap, seed
    )
    result = {
        "schema": "lookback-ratio-diagnostic-v1",
        "analysis_status": "post_hoc_supervised_scalar_diagnostic",
        "probe_uses_labels": True,
        "feature": feature,
        "undefined_ratio_fill": float(train_metadata["attention_floor"]),
        "train_direction": "higher" if direction >= 0 else "lower",
        "univariate": {"raw_test": raw, "train_oriented_test": oriented},
        "nuisance_only": nuisance,
        "lookback_plus_nuisance": {
            "held_out": adjusted,
            "point_delta": {
                metric: adjusted[metric] - nuisance[metric]
                for metric in ("auroc", "auprc")
            },
        },
        "train_tokens": len(train_labels),
        "test_tokens": len(test_labels),
        "overlap_audit": _overlap_audit(train, test),
    }
    np.savez_compressed(
        output / "predictions.npz", labels=test_labels,
        sample_ids=test.sample_ids, positions=test.positions,
        lookback=test_lookback, nuisance=nuisance_score,
        lookback_plus_nuisance=adjusted_score,
    )
    _write_json(output / "results.json", result)
    return result


def _mechanism_node_features(directory: Path, sample_id: str, response_idx: int, tokens: int, metadata: dict) -> tuple[torch.Tensor, torch.Tensor]:
    item = torch.load(directory / f"{sample_id}.pt", map_location="cpu", weights_only=True)
    values, valid = item["values"].float(), item["valid"].bool()
    if len(values) != tokens - response_idx:
        raise ValueError("mechanism artifact response length does not match attention sample")
    indices = [index for index, name in enumerate(metadata["feature_names"]) if name.endswith(":global_mean")]
    if not indices:
        raise ValueError("mechanism artifact lacks global_mean features")
    response_nodes = torch.where(valid[:, indices], values[:, indices], torch.zeros_like(values[:, indices]))
    nodes = torch.zeros((tokens, len(indices)), dtype=torch.float32)
    nodes[response_idx:] = response_nodes
    node_valid = torch.zeros((tokens, len(indices)), dtype=torch.bool)
    node_valid[response_idx:] = valid[:, indices]
    return nodes, node_valid


def build_graphs(split_root, mechanism_features, output_dir, *, device="cuda", variants=None, seed=0) -> dict:
    output = _empty_output(output_dir)
    variants = variants or ["exact", "no_edges", "unit_mass", "uniform_on_support", "weight_shuffle", "source_rewire", "rp_only", "rr_only", "source_free"]
    feature_directory = Path(mechanism_features)
    mechanism_metadata = _read_json(feature_directory / "metadata.json")
    if mechanism_metadata.get("labels_included") is not False:
        raise ValueError("mechanism metadata must exclude labels")
    dataset = open_research_dataset(split_root, device=device)
    index = {variant: [] for variant in variants}
    feature_names = None
    source_aware = None
    samples = []
    node_feature_names = [name for name in mechanism_metadata["feature_names"] if name.endswith(":global_mean")]
    total = len(dataset) if hasattr(dataset, "__len__") else None
    for sample in tqdm(dataset, total=total, desc="build graph", unit="sample"):
        attention = sample.attention()
        nodes, node_valid = _mechanism_node_features(feature_directory, sample.sample_id, attention.response_idx, attention.num_tokens, mechanism_metadata)
        nodes, node_valid = nodes.to(device), node_valid.to(device)
        base_dir = output / "base"
        base_dir.mkdir(exist_ok=True)
        torch.save({"sample_id": sample.sample_id, "source_id": sample.source_id,
                    "prompt_length": attention.response_idx,
                    "task_type": sample.task_type, "data_source": sample.data_source,
                    "values": nodes[attention.response_idx:].detach().cpu().to(torch.float16),
                    "valid": node_valid[attention.response_idx:].detach().cpu()},
                   base_dir / f"{sample.sample_id}.pt")
        trace = extract_traces(attention)
        sample_seed = (seed + int.from_bytes(hashlib.sha256(str(sample.sample_id).encode()).digest()[:8], "little")) % (2**63 - 1)
        for variant in variants:
            transformed, audit = (trace, {"variant": "source_free", "edges_before": int(trace.value.numel()), "edges_after": int(trace.value.numel()), "changed_fraction": 0.0}) if variant == "source_free" else apply_trace_variant(trace, variant, response_idx=attention.response_idx, seed=sample_seed)
            descriptors = fixed_graph_descriptors(transformed, nodes, attention.response_idx, attention.num_layers, attention.num_heads, source_free=variant == "source_free")
            variant_dir = output / variant
            variant_dir.mkdir(exist_ok=True)
            torch.save({"sample_id": sample.sample_id, "source_id": sample.source_id,
                        "values": descriptors.features.detach().cpu().to(torch.float16)},
                       variant_dir / f"{sample.sample_id}.pt")
            index[variant].append({"sample_id": sample.sample_id, "source_id": sample.source_id, **audit})
            feature_names = descriptors.feature_names
            source_aware = descriptors.source_aware.detach().cpu().tolist()
        samples.append({"sample_id": sample.sample_id, "source_id": sample.source_id,
                        "tokens": attention.num_response_tokens})
        sample.release_attention()
    _write_json(output / "metadata.json", {"schema": "graph_features.v2", "labels_included": False, "seed": seed,
                                               "randomization_repeats": 1, "analysis_status": "exploratory",
                                               "variants": variants,
                                               "feature_names": feature_names, "source_aware": source_aware,
                                               "node_feature_names": node_feature_names,
                                               "mechanism_fingerprint": {key: mechanism_metadata[key] for key in ("schema", "ema_decay", "attention_floor", "feature_names", "family_slices")}})
    _write_json(output / "index.json", {"labels_included": False, "samples": samples, "variants": index})
    return {"responses": sum(len(rows) for rows in index.values()) // len(variants), "variants": len(variants)}


def _graph_split(directory: Path, variant: str, *, max_tokens: int | None = None, seed: int = 0) -> FeatureSplit:
    metadata = _read_json(directory / "metadata.json")
    if metadata.get("schema") != "graph_features.v2" or metadata.get("labels_included") is not False:
        raise ValueError("graph metadata must exclude labels")
    variant_dir = directory / variant
    files = sorted(variant_dir.glob("*.pt"))
    index_path = directory / "index.json"
    if not index_path.exists():
        raise ValueError("graph index is required")
    index = _read_json(index_path)["samples"]
    inventory = {str(row["sample_id"]): (str(row["source_id"]), int(row["tokens"])) for row in index}
    if len(inventory) != len(index) or set(inventory) != {path.stem for path in files} or set(inventory) != {path.stem for path in (directory / "base").glob("*.pt")}:
        raise ValueError("graph index, base, and variant files must match exactly")
    counts = {sample_id: tokens for sample_id, (_, tokens) in inventory.items()}
    total = sum(counts.get(path.stem, 0) for path in files)
    selected = None
    if max_tokens is not None and total > max_tokens:
        selected = np.sort(np.random.default_rng(seed).choice(total, size=max_tokens, replace=False))
    values = []
    sample_ids = []
    source_ids = []
    positions = []
    prompt_lengths = []
    response_lengths = []
    task_types = []
    data_sources = []
    valid = []
    offset = 0
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=True)
        base = torch.load(directory / "base" / path.name, map_location="cpu", weights_only=True)
        source_id, indexed_count = inventory[path.stem]
        if str(item["sample_id"]) != path.stem or str(base["sample_id"]) != path.stem or str(item["source_id"]) != source_id or str(base["source_id"]) != source_id or len(item["values"]) != indexed_count:
            raise ValueError("graph payload and index disagree")
        if "valid" not in base:
            raise ValueError("graph base artifact lacks valid mask")
        original_count = len(item["values"])
        if len(base["values"]) != original_count:
            raise ValueError("graph descriptor and fixed node feature lengths differ")
        if selected is not None:
            local = selected[(selected >= offset) & (selected < offset + original_count)] - offset
            base_values, base_valid, graph_values = base["values"][local], base["valid"][local], item["values"][local]
            position = local.tolist()
        else:
            base_values, base_valid, graph_values = base["values"], base["valid"], item["values"]
            position = list(range(original_count))
        offset += original_count
        count = len(graph_values)
        values.append(torch.cat((base_values, graph_values), dim=1).float().numpy())
        valid.append(torch.cat((base_valid.bool(), torch.isfinite(graph_values)), dim=1).numpy())
        sample_ids.extend([str(item["sample_id"])] * count)
        source_ids.extend([str(item["source_id"])] * count)
        positions.extend(position); response_lengths.extend([original_count] * count)
        prompt_lengths.extend([int(base["prompt_length"])] * count)
        task_types.extend([str(base["task_type"])] * count)
        data_sources.extend([str(base["data_source"])] * count)
    all_values = np.concatenate(values)
    node_count = len(metadata["node_feature_names"])
    names = [f"node:{name}" for name in metadata["node_feature_names"]] + metadata["feature_names"]
    return FeatureSplit(all_values, np.concatenate(valid), np.asarray(sample_ids), np.asarray(source_ids), np.asarray(positions),
                        np.asarray(prompt_lengths), np.asarray(response_lengths),
                        np.asarray(task_types), np.asarray(data_sources), names,
                        {"node": slice(0, node_count), "graph": slice(node_count, all_values.shape[1])}, inventory)


def evaluate_graphs(train_split, train_graphs, test_split, test_graphs, output_dir, *, seed=0, max_train_tokens=100000,
                    bootstrap=200) -> dict:
    output = _empty_output(output_dir)
    train_root, test_root = Path(train_graphs), Path(test_graphs)
    train_metadata, test_metadata = _read_json(train_root / "metadata.json"), _read_json(test_root / "metadata.json")
    for key in ("schema", "labels_included", "seed", "variants", "feature_names", "node_feature_names", "source_aware", "mechanism_fingerprint"):
        if train_metadata.get(key) != test_metadata.get(key):
            raise ValueError(f"train/test graph metadata differ: {key}")
    if train_metadata.get("labels_included") is not False:
        raise ValueError("graph metadata must exclude labels")
    variants = train_metadata["variants"]
    if not variants or variants[0] != "exact":
        raise ValueError("graph evaluation requires exact as the first variant")
    first = "exact"
    first_train = _graph_split(train_root, first, max_tokens=max_train_tokens, seed=seed)
    first_test = _graph_split(test_root, first)
    train_labels = _load_labels(train_split, first_train)
    test_labels = _load_labels(test_split, first_test)
    result = {"analysis_status": "post_hoc_exploratory", "probe_uses_labels": True,
              "overlap_audit": _overlap_audit(first_train, first_test),
              "representation_sufficiency": {}, "decoder_sensitivity": {},
              "representation_point_deltas": {}, "decoder_point_deltas": {}, "paired_cluster_intervals": {}}
    saved = {"labels": test_labels, "sample_ids": first_test.sample_ids, "positions": first_test.positions}
    exact_train = exact_test = None
    representation_scores, decoder_scores = {}, {}
    for variant in tqdm(variants, desc="graph variants", unit="variant"):
        if variant == first:
            train_data, test_data = first_train, first_test
        else:
            train_data = _graph_split(train_root, variant, max_tokens=max_train_tokens, seed=seed)
            test_data = _graph_split(test_root, variant)
        metrics, scores = _diagnostic_results(train_data, train_labels, test_data, test_labels,
                                              seed=seed, maximum=max_train_tokens)
        result["representation_sufficiency"][variant] = metrics
        representation_scores[variant] = scores
        if train_data.inventory != first_train.inventory or test_data.inventory != first_test.inventory or not (np.array_equal(train_data.sample_ids, first_train.sample_ids) and np.array_equal(train_data.source_ids, first_train.source_ids) and np.array_equal(train_data.positions, first_train.positions) and np.array_equal(test_data.sample_ids, first_test.sample_ids) and np.array_equal(test_data.source_ids, first_test.source_ids) and np.array_equal(test_data.positions, first_test.positions)):
            raise ValueError("graph variants are not aligned")
        for name, score in scores.items():
            saved[f"{variant}__{name}"] = score
        if variant == "exact":
            exact_train, exact_test = train_data, test_data
        elif variant != first:
            del train_data, test_data
    if exact_train is not None:
        decoders = {name: _fit_graph_decoder(exact_train, train_labels,
                    np.arange((exact_train.family_slices["graph"] if name == "graph_only" else slice(0, len(exact_train.names))).start,
                              (exact_train.family_slices["graph"] if name == "graph_only" else slice(0, len(exact_train.names))).stop))
                    for name in ("full", "graph_only")}
        for variant in variants:
            test_data = exact_test if variant == "exact" else _graph_split(test_root, variant)
            result["decoder_sensitivity"][variant] = {}
            for name in ("full", "graph_only"):
                scores = _apply_graph_decoder(decoders[name], exact_train, test_data)
                result["decoder_sensitivity"][variant][name] = _metric_with_control(test_labels, scores, test_data, 0, seed)
                decoder_scores.setdefault(variant, {})[name] = scores
                saved[f"decoder_exact__{variant}__{name}"] = scores
            if variant != "exact":
                del test_data
        exact_metrics = result["representation_sufficiency"]["exact"]
        for variant, metrics in result["representation_sufficiency"].items():
            if variant != "exact":
                result["representation_point_deltas"][f"exact_minus_{variant}"] = {
                    name: {metric: exact_metrics[name][metric] - metrics[name][metric] for metric in ("auroc", "auprc")}
                    for name in ("full", "graph_only")
                    if exact_metrics[name]["auroc"] is not None and metrics[name]["auroc"] is not None
                }
                result["decoder_point_deltas"][f"exact_minus_{variant}"] = {
                    name: {metric: result["decoder_sensitivity"]["exact"][name][metric] - result["decoder_sensitivity"][variant][name][metric] for metric in ("auroc", "auprc")}
                    for name in ("full", "graph_only") if result["decoder_sensitivity"]["exact"][name]["auroc"] is not None and result["decoder_sensitivity"][variant][name]["auroc"] is not None}
        for key, left, right, view, scores in (
            ("representation_exact_full_vs_no_edges_full", "exact", "no_edges", "full", representation_scores),
            ("decoder_exact_full_vs_no_edges_full", "exact", "no_edges", "full", decoder_scores),
            ("representation_rp_only_graph_only_vs_rr_only_graph_only", "rp_only", "rr_only", "graph_only", representation_scores),
        ):
            if left in scores and right in scores:
                result["paired_cluster_intervals"][key] = paired_cluster_delta(test_labels, scores[left][view], scores[right][view], first_test.source_ids, n_resamples=bootstrap, seed=seed)
    np.savez_compressed(output / "predictions.npz", **saved)
    _write_json(output / "results.json", result)
    return result
