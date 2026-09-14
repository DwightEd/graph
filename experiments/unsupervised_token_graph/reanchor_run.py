"""Stream frozen, label-free per-head source-flow measurements from NPZ caches."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .channels import iter_channels
from .data import CacheDataset
from .information import SourceFlow, future_influence
from .reanchor import ReanchorAnalyzer


VERSION = "source-carrier-information-v1"


def read_metadata(path):
    if path is None:
        return {}
    result = {}
    fields = ("id", "source_id", "response_sha256", "offsets", "split", "official_split", "task", "generator")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("cache", row.get("trace", "")))
            if not key or key in result:
                raise ValueError("metadata needs unique relative cache/trace paths")
            result[key] = {k: row[k] for k in fields if k in row}
    return result


def analyze_record(record, analyzer=None, root_bins=None, layers=None, heads=None,
                   controls=True, offline_influence=False, horizon_low=10, horizon_high=100):
    """All returned measurements are [channel, query]; identities stay explicit."""
    analyzer = analyzer or ReanchorAnalyzer()
    flow = SourceFlow(root_bins)
    collected, layer_ids, head_ids = {}, [], []
    for graph in iter_channels(record, layers, heads):
        values = {**asdict(analyzer.run(graph)), **flow.run(graph)}
        if controls:
            permuted = flow.run(graph, control="history_permuted")
            values["permuted_source_mismatch_bits"] = permuted["source_mismatch_bits"]
            values["permuted_prompt_reach"] = permuted["prompt_reach"]
        if offline_influence:
            influence, count = future_influence(graph, horizon_low, horizon_high)
            values["offline_fai"], values["offline_fai_count"] = influence, count
        for name, array in values.items():
            collected.setdefault(name, []).append(array)
        layer_ids.append(graph.layer); head_ids.append(graph.head)
    if not layer_ids:
        raise ValueError("no attention channels selected")
    result = {k: np.stack(v) for k, v in collected.items()}
    result.update(layer_ids=np.asarray(layer_ids), head_ids=np.asarray(head_ids),
                  query_positions=graph.queries, prediction_positions=graph.prediction_positions,
                  prompt_length=np.asarray(graph.prompt_length),
                  total_tokens=np.asarray(graph.attention.shape[1]),
                  cache_format=np.asarray("canonical_csr" if record.sparse is not None else "dense"))
    return result


def run(cache_root, output, metadata=None, window=10, event_quantile=.9, cooldown=2,
        min_history=8, root_bins=None, layers=None, heads=None, pattern="*.npz", controls=True,
        offline_influence=False, horizon_low=10, horizon_high=100, resume=False):
    dataset, output = CacheDataset(cache_root, pattern), Path(output)
    if not len(dataset):
        raise FileNotFoundError("no attention NPZ files matched the cache pattern")
    meta = read_metadata(metadata)
    config = dict(version=VERSION, cache=str(dataset.root.resolve()), pattern=pattern,
                  window=window, event_quantile=event_quantile, cooldown=cooldown,
                  min_history=min_history, root_bins=root_bins, layers=layers, heads=heads,
                  controls=controls, offline_influence=offline_influence,
                  horizon_low=horizon_low, horizon_high=horizon_high,
                  metadata=meta, labels_read=False)
    settings = output / "settings.json"
    if output.exists():
        if not resume or not settings.exists() or json.loads(settings.read_text()) != config:
            raise ValueError("use a new output, or --resume with identical method and metadata")
    else:
        output.mkdir(parents=True)
        settings.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "complete.json").unlink(missing_ok=True)
    analyzer = ReanchorAnalyzer(window, event_quantile, cooldown, min_history)
    rows = []
    basenames = [p.name for p in dataset.files]
    for path in tqdm(dataset.files, desc="reanchor/source-flow", unit="sample"):
        relative = path.relative_to(dataset.root)
        identity = meta.get(relative.as_posix(), {})
        if not identity and relative.name in meta:
            if basenames.count(relative.name) > 1:
                raise ValueError("ambiguous metadata basename; specify the relative cache path")
            identity = meta[relative.name]
        destination = output / "samples" / relative
        stamp = [path.stat().st_size, path.stat().st_mtime_ns]
        if destination.exists() and resume:
            with np.load(destination, allow_pickle=False) as saved:
                if saved["cache_stamp"].tolist() != stamp:
                    raise ValueError("cache changed since scoring: " + str(relative))
                row = json.loads(str(saved["record_json"]))
        else:
            record = dataset.cache.load(path)
            arrays = analyze_record(record, analyzer, root_bins, layers, heads, controls,
                                    offline_influence, horizon_low, horizon_high)
            cache_meta = record.metadata or {}
            def scalar(name, fallback=""):
                value = cache_meta.get(name, fallback)
                return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
            row = dict(cache=relative.as_posix(), file=(Path("samples") / relative).as_posix(),
                       id=str(identity.get("id", scalar("sample_id", record.response_id))),
                       source_id=str(identity.get("source_id", record.source_id)),
                       split=str(identity.get("official_split", identity.get("split", scalar("official_split", scalar("dataset_split"))))),
                       task=str(identity.get("task", scalar("task"))),
                       generator=str(identity.get("generator", scalar("generator"))),
                       response_sha256=str(identity.get("response_sha256", scalar("response_sha256"))),
                       queries=len(arrays["query_positions"]), channels=len(arrays["layer_ids"]),
                       events=int(arrays["event"].sum()))
            offsets = identity.get("offsets", record.offsets)
            if offsets is not None:
                arrays["offsets"] = np.asarray(offsets, dtype=np.int64)
            if record.token_ids is not None:
                arrays["token_ids"] = record.token_ids
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays, cache_stamp=np.asarray(stamp),
                                    record_json=np.asarray(json.dumps(row, ensure_ascii=False)))
            temporary.replace(destination)
            del record, arrays
        rows.append(row)
    summary = dict(version=VERSION, responses=rows, labels_read=False,
                   graph="per-head token DAG; prompt boundary roots; self and unknown terminals",
                   score_semantics="routing disagreement, not truth probability")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "complete.json").write_text(json.dumps(dict(complete=True, version=VERSION, responses=len(rows), labels_read=False)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", help="label-free JSONL: cache/trace, id, source_id, split, response_sha256, offsets")
    parser.add_argument("--pattern", default="*.npz", help="use **/*.npz for nested task/split directories")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--event-quantile", type=float, default=.9)
    parser.add_argument("--cooldown", type=int, default=2)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--root-bins", type=int, help="optional lossy prompt-address grouping; omitted = exact token roots")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--heads", type=int, nargs="+")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--offline-influence", action="store_true", help="export future-use diagnostics separately; never an online score")
    parser.add_argument("--horizon-low", type=int, default=10)
    parser.add_argument("--horizon-high", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args(argv))
    args["cache_root"] = args.pop("cache")
    args["controls"] = not args.pop("no_controls")
    rows = run(**args)
    print(json.dumps(dict(complete=True, responses=len(rows), labels_read=False)))


if __name__ == "__main__":
    main()
