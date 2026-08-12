"""Strict label-blind validation of attention-graph construction choices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from cache import sha256
from .graph import GraphBuildConfig, build_attention_graph
from .graph_variants import VARIANTS, rewire_moved_fractions, transform_graph
from .patterns import provenance_curves

DEFAULT_VARIANTS = ("full", "no_edges", "marginals", "source_rewire", "binary", "shuffle_layers")
VARIANT_ROLES = {
    "full": "reference", "no_edges": "diagonal_only_control", "marginals": "source_identity_control",
    "source_rewire": "rr_incidence_control", "binary": "support_only_control", "shuffle_layers": "layer_order_control",
    "collapse_relations": "expected_invariance_control", "mean_heads": "expected_invariance_control",
}


@dataclass(frozen=True)
class GraphValidationConfig:
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    checkpoints: int = 8
    neighbors: int = 16
    reference_size: int = 100_000
    span_width: int = 8
    seed: int = 42

    def validate(self):
        if "full" not in self.variants or len(set(self.variants)) != len(self.variants) or any(name not in VARIANTS for name in self.variants):
            raise ValueError("variants must be unique, include full, and be supported")
        if self.checkpoints < 2 or min(self.neighbors, self.reference_size, self.span_width) < 1:
            raise ValueError("checkpoints >= 2 and remaining limits must be positive")


def _fingerprint(dataset):
    manifest = dataset.manifest
    rows = getattr(dataset, "rows", {})
    items = []
    for sample_id in dataset.sample_ids:
        row = rows[sample_id]
        items.append((str(sample_id), str(row.get("sha256", ""))))
    payload = {"manifest": manifest, "samples": items}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _geometry(dataset):
    manifest = dataset.manifest
    return {name: manifest.get(name) for name in ("schema", "num_layers", "num_heads", "alignment", "attention_floor", "observer_model")}


def _stable_seed(seed, sample_id):
    digest = hashlib.sha256(str(sample_id).encode()).digest()
    return int((seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


def _scale(values):
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    std = values.std(axis=0)
    return center.astype(np.float32), np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0)).astype(np.float32)


def _score(train, test, config):
    center, scale = _scale(train)
    train, test = (train - center) / scale, (test - center) / scale
    rng = np.random.default_rng(config.seed)
    reference = train[np.sort(rng.choice(len(train), min(config.reference_size, len(train)), replace=False))]
    model = NearestNeighbors(n_neighbors=min(config.neighbors, len(reference))).fit(reference)
    return test.astype(np.float32), model.kneighbors(test, return_distance=True)[0].mean(1).astype(np.float32), center, scale


def _metadata(rows):
    return {name: np.asarray(rows[name], dtype=dtype) for name, dtype in (
        ("sample_id", str), ("source_id", str), ("token_index", np.int32), ("task_type", str),
        ("data_source", str), ("generator_model", str),
    )}


def _spans(values, metadata, width):
    output, rows = [], {name: [] for name in (*metadata, "span_start", "span_end")}
    skipped, start = 0, 0
    sample_ids = metadata["sample_id"]
    while start < len(sample_ids):
        end = start + 1
        while end < len(sample_ids) and sample_ids[end] == sample_ids[start]:
            end += 1
        if end - start < width:
            skipped += 1
        else:
            for token_start in range(start, end - width + 1):
                token_end, window = token_start + width, values[token_start:token_start + width]
                output.append(np.concatenate((window.mean(0), window[-1] - window[0])))
                for name, items in metadata.items():
                    rows[name].append(items[token_start])
                rows["span_start"].append(token_start - start)
                rows["span_end"].append(token_end - start)
        start = end
    dimension = values.shape[1] * 2
    matrix = np.asarray(output, dtype=np.float32).reshape(len(output), dimension)
    return matrix, {name: np.asarray(value) for name, value in rows.items()}, skipped


def _valid_output(output):
    return not output.exists() or not any(output.iterdir())


class GraphValidator:
    """Build every candidate graph once per sample and fit train-only kNN."""

    def __init__(self, config=None):
        self.config = GraphValidationConfig() if config is None else config
        self.config.validate()

    def _extract(self, dataset, graph_config, split_name):
        matrices = {variant: [] for variant in self.config.variants}
        moved = []
        rows = {name: [] for name in ("sample_id", "source_id", "token_index", "task_type", "data_source", "generator_model")}
        for sample_id in tqdm(dataset.sample_ids, desc=f"{split_name} graphs", unit="graph"):
            sample = dataset[sample_id]
            graph = build_attention_graph(sample.attention(), graph_config)
            count = graph.num_nodes - graph.response_idx
            for variant in self.config.variants:
                transformed = transform_graph(graph, variant, seed=_stable_seed(self.config.seed, sample.sample_id))
                signature, _ = provenance_curves(transformed, checkpoints=self.config.checkpoints, signature_view="prompt_absorption")
                matrices[variant].append(signature.detach().cpu().numpy())
                if variant == "source_rewire":
                    moved.append(rewire_moved_fractions(graph, transformed))
            rows["sample_id"].extend([sample.sample_id] * count)
            rows["source_id"].extend([sample.source_id] * count)
            rows["token_index"].extend(range(count))
            for field in ("task_type", "data_source", "generator_model"):
                rows[field].extend([str(getattr(sample, field))] * count)
            sample.release_attention()
        metadata = _metadata(rows)
        return {name: np.concatenate(value).astype(np.float32) for name, value in matrices.items()}, metadata, moved, set(metadata["source_id"].tolist())

    def run(self, train_dataset, test_dataset, output_dir, *, graph_config=None):
        graph_config = GraphBuildConfig() if graph_config is None else graph_config
        if _geometry(train_dataset) != _geometry(test_dataset):
            raise ValueError("train and test attention geometry differ")
        output = Path(output_dir)
        if not _valid_output(output):
            raise ValueError("graph validation output directory must be empty")
        output.mkdir(parents=True, exist_ok=True)
        train, train_metadata, train_moved, train_sources = self._extract(train_dataset, graph_config, "train")
        test, metadata, test_moved, test_sources = self._extract(test_dataset, graph_config, "test")
        if train_sources & test_sources:
            raise ValueError("train and test source groups overlap")
        artifacts = []
        for variant in self.config.variants:
            token_embedding, token_score, center, scale = _score(train[variant], test[variant], self.config)
            train_span, _, train_skipped = _spans(train[variant], train_metadata, self.config.span_width)
            test_span, span_metadata, test_skipped = _spans(test[variant], metadata, self.config.span_width)
            if not len(train_span) or not len(test_span):
                raise ValueError("span width leaves no train or test spans")
            span_embedding, span_score, span_center, span_scale = _score(train_span, test_span, self.config)
            name = f"{variant}.npz"
            path = output / name
            np.savez_compressed(path, schema=np.asarray("attention-graph-construction-v2"), variant=np.asarray(variant),
                variant_role=np.asarray(VARIANT_ROLES[variant]), representation=np.asarray("layer_provenance_prompt_absorption"),
                token_embedding=token_embedding, token_score=token_score, span_embedding=span_embedding, span_score=span_score,
                center=center, scale=scale, span_center=span_center, span_scale=span_scale, **metadata,
                span_sample_id=span_metadata["sample_id"].astype(str), span_source_id=span_metadata["source_id"].astype(str),
                span_start=span_metadata["span_start"].astype(np.int32), span_end=span_metadata["span_end"].astype(np.int32),
                span_task_type=span_metadata["task_type"].astype(str), span_data_source=span_metadata["data_source"].astype(str),
                span_generator_model=span_metadata["generator_model"].astype(str))
            artifacts.append({"variant": variant, "variant_role": VARIANT_ROLES[variant], "path": name, "sha256": sha256(path), "train_spans_skipped": train_skipped, "test_spans_skipped": test_skipped})
        manifest = {"schema": "attention-graph-construction-v2", "labels_consumed": False, "labels_retained": False,
                    "formal_labels_may_be_embedded_but_unused": True, "representation": "layer_provenance_prompt_absorption",
                    "geometry": _geometry(train_dataset), "train_fingerprint": _fingerprint(train_dataset), "test_fingerprint": _fingerprint(test_dataset),
                    "graph": asdict(graph_config), "validation": asdict(self.config), "variants": artifacts,
                    "source_rewire": {
                        split: {key: float(np.mean([row[key] for row in values])) if values else None for key in ("overall", "rp", "rr")}
                        for split, values in (("train", train_moved), ("test", test_moved))
                    }}
        train_rr = manifest["source_rewire"]["train"]["rr"]
        test_rr = manifest["source_rewire"]["test"]["rr"]
        manifest["source_rewire"]["status"] = (
            "estimable" if train_rr is not None and test_rr is not None and train_rr > 0 and test_rr > 0 else "non_estimable"
        )
        (output / "label_free_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"output_dir": str(output), "artifacts": [row["path"] for row in artifacts], "labels_consumed": False}


def _ranking(labels, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return {"n": int(len(labels)), "positives": int(labels.sum()), "auroc": None, "auprc": None}
    return {"n": int(len(labels)), "positives": int(labels.sum()), "auroc": float(roc_auc_score(labels, score)), "auprc": float(average_precision_score(labels, score))}


def _report(labels, scores, metadata):
    result = {"overall": _ranking(labels, scores)}
    for field in ("data_source", "task_type"):
        values = metadata[field].astype(str)
        result[field] = {group: _ranking(labels[values == group], scores[values == group]) for group in sorted(np.unique(values).tolist())}
    return result


def _bootstrap(full_labels, full_score, labels, score, source_ids, *, seed, bootstraps):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not np.array_equal(full_labels, labels):
        raise ValueError("full and variant labels are not paired")
    if len(np.unique(labels)) < 2 or len(np.unique(source_ids)) < 2:
        return {metric: {"point": None, "ci_low": None, "ci_high": None, "n_boot": 0} for metric in ("auroc", "auprc")}
    metric = {"auroc": roc_auc_score, "auprc": average_precision_score}
    point = {name: float(function(labels, full_score) - function(labels, score)) for name, function in metric.items()}
    rng, groups, draws = np.random.default_rng(seed), np.unique(source_ids), {name: [] for name in metric}
    for _ in range(bootstraps):
        ids = np.concatenate([np.flatnonzero(source_ids == group) for group in rng.choice(groups, len(groups), replace=True)])
        if len(np.unique(labels[ids])) == 2:
            for name, function in metric.items():
                draws[name].append(function(labels[ids], full_score[ids]) - function(labels[ids], score[ids]))
    return {name: {"point": value, "ci_low": float(np.quantile(draws[name], .025)) if draws[name] else None,
                   "ci_high": float(np.quantile(draws[name], .975)) if draws[name] else None, "n_boot": len(draws[name]),
                   "direction": "full_minus_variant", "inference_scope": "paired_source_cluster_bootstrap"} for name, value in point.items()}


def _labels(dataset):
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        sample.attention()
        sample.release_attention()
    store, output = dataset.labels(), {}
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        output[sample_id] = store.response_labels(sample).cpu().numpy()
        sample.release_attention()
    return output


def _structured(*columns):
    return np.rec.fromarrays(columns, names=[f"f{i}" for i in range(len(columns))])


def _expected_spans(dataset, labels, width):
    sample_ids, starts, ends = [], [], []
    for sample_id in dataset.sample_ids:
        count = len(labels[sample_id])
        for start in range(max(0, count - width + 1)):
            sample_ids.append(sample_id)
            starts.append(start)
            ends.append(start + width)
    return _structured(np.asarray(sample_ids), np.asarray(starts, dtype=np.int32), np.asarray(ends, dtype=np.int32))


def _expected_metadata(dataset, labels):
    rows = {}
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        count = len(labels[sample_id])
        rows[sample_id] = {
            "source_id": str(sample.source_id), "task_type": str(sample.task_type),
            "data_source": str(sample.data_source), "generator_model": str(sample.generator_model),
            "count": count,
        }
        sample.release_attention()
    return rows


def evaluate_graph_artifacts(dataset, artifact_dir, output_path, *, bootstraps=400, seed=42):
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    directory = Path(artifact_dir)
    manifest = json.loads((directory / "label_free_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "attention-graph-construction-v2" or manifest["test_fingerprint"] != _fingerprint(dataset):
        raise ValueError("artifacts do not belong to this exact test split")
    labels, reports, rows = _labels(dataset), {}, {}
    canonical = _expected_metadata(dataset, labels)
    width = int(manifest["validation"]["span_width"])
    expected_span = _expected_spans(dataset, labels, width)
    for entry in manifest["variants"]:
        path = directory / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("artifact hash does not match manifest")
        with np.load(path, allow_pickle=False) as value:
            if value["schema"].item() != manifest["schema"] or value["variant"].item() != entry["variant"]:
                raise ValueError("artifact schema or variant differs from manifest")
            token_key = _structured(value["sample_id"], value["token_index"])
            span_key = _structured(value["span_sample_id"], value["span_start"], value["span_end"])
            if len(np.unique(token_key)) != len(token_key) or len(np.unique(span_key)) != len(span_key):
                raise ValueError("artifact contains duplicate rows")
            expected_sample = []
            expected_index = []
            for sample_id in dataset.sample_ids:
                expected_sample.extend([sample_id] * len(labels[sample_id]))
                expected_index.extend(range(len(labels[sample_id])))
            expected_token = _structured(np.asarray(expected_sample), np.asarray(expected_index, dtype=np.int32))
            if not np.array_equal(np.sort(token_key), np.sort(expected_token)):
                raise ValueError("artifact does not cover every response token exactly")
            if np.any(value["span_end"] - value["span_start"] != width) or np.any(value["span_start"] < 0):
                raise ValueError("artifact contains malformed span boundaries")
            if not np.array_equal(np.sort(span_key), np.sort(expected_span)):
                raise ValueError("artifact does not cover every fixed-width span exactly")
            for field in ("source_id", "task_type", "data_source", "generator_model"):
                expected = np.asarray([canonical[str(sample_id)][field] for sample_id in value["sample_id"]])
                if not np.array_equal(value[field].astype(str), expected):
                    raise ValueError(f"artifact token {field} differs from canonical split")
                expected_span_field = np.asarray([canonical[str(sample_id)][field] for sample_id in value["span_sample_id"]])
                if not np.array_equal(value[f"span_{field}"].astype(str), expected_span_field):
                    raise ValueError(f"artifact span {field} differs from canonical split")
            token_labels = np.asarray([labels[str(s)][int(i)] for s, i in zip(value["sample_id"], value["token_index"])])
            span_labels = np.asarray([labels[str(s)][int(a):int(b)].max() for s, a, b in zip(value["span_sample_id"], value["span_start"], value["span_end"])])
            reports[entry["variant"]] = {"role": entry["variant_role"], "token": _report(token_labels, value["token_score"], {f: value[f] for f in ("data_source", "task_type")} ),
                "span": _report(span_labels, value["span_score"], {"data_source": value["span_data_source"], "task_type": value["span_task_type"]})}
            rows[entry["variant"]] = {"token": (token_key, token_labels, value["token_score"].copy(), value["source_id"].copy()),
                "span": (span_key, span_labels, value["span_score"].copy(), value["span_source_id"].copy())}
    rewire_status = manifest["source_rewire"]["status"]
    for variant, report in reports.items():
        report["full_minus_variant"] = {}
        for unit in ("token", "span"):
            key, label, score, source = rows[variant][unit]
            full_key, full_label, full_score, _ = rows["full"][unit]
            if not np.array_equal(key, full_key):
                raise ValueError("graph variants do not cover identical ordered rows")
            if variant == "source_rewire" and rewire_status != "estimable":
                report["full_minus_variant"][unit] = {
                    metric: {"point": None, "ci_low": None, "ci_high": None, "n_boot": 0,
                             "status": "non_estimable", "direction": "full_minus_variant"}
                    for metric in ("auroc", "auprc")
                }
            else:
                report["full_minus_variant"][unit] = _bootstrap(full_label, full_score, label, score, source, seed=seed, bootstraps=bootstraps)
    result = {"schema": "attention-graph-construction-evaluation-v2", "labels_consumed": True,
              "labels_consumed_during": "evaluation_only", "source_rewire": manifest["source_rewire"], "variants": reports}
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
