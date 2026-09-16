"""Replay the EXISTING checkpoint and save additional frozen controls separately."""

import json
from pathlib import Path
import zlib

import numpy as np
from tqdm import tqdm

from ..graph import degree, prefix_graph, transform_graph
from ..model import load_checkpoint
from .common import write_csv, write_json
from .controls import (degree_preserving_rewire, model_graph, oracle_span_cut,
                       permute_head_identity, shuffle_endpoints, subset_edges)
from .representations import features, matched_relation_contrasts, relation_rows, trace_model


def read_graph(record, prepared=None):
    path = Path(record["graph"])
    if prepared is not None:
        path = Path(prepared) / "graphs" / record["split"] / (str(record["id"]) + ".npz")
    with np.load(path, allow_pickle=False) as saved:
        graph = {key: saved[key] for key in saved.files}
    identity = json.loads(str(graph["record_json"]))
    if str(identity["id"]) != str(record["id"]) or str(identity["source_id"]) != str(record["source_id"]):
        raise ValueError("Graph identity differs from saved prediction: " + str(path))
    for key in ("gold", "offsets", "spans"):
        if key in record and not np.array_equal(graph[key], record[key]):
            raise ValueError("Graph/prediction token alignment differs: " + key)
    if "response" in record and str(graph["response"]) != str(record["response"]):
        raise ValueError("Graph/prediction text differs")
    return graph, path


def frozen_score(model, graph, divisor):
    import torch

    with torch.no_grad():
        logits = model(graph, divisor=divisor)
        return torch.sigmoid(logits[int(graph["prompt_length"]):]).cpu().numpy()


def capture_controls(model, graph, sample, seed, repeats):
    """One graph at a time; do not retain a collection of altered edge tensors."""
    divisor = degree(graph, model.normalization)
    values, diagnostics = {}, {}
    empty = subset_edges(graph, np.zeros(graph["edge_index"].shape[1], bool))
    values["score_no_messages"] = frozen_score(model, empty, divisor)
    values["score_no_relay"] = trace_model(model, graph, block_relay=True)[0]
    values["score_edge_only_messages"] = trace_model(model, graph, block_sender=True)[0]
    for edge_only in (False, True):
        name = "head_identity_edge_only" if edge_only else "head_identity_joint"
        view, info = permute_head_identity(graph, seed, edge_only)
        values["score_" + name] = frozen_score(model, view, divisor)
        diagnostics[name] = info
        del view
    for repeat in range(repeats):
        current_seed = seed + repeat
        for independent in (False, True):
            name = ("independent" if independent else "coupled") + f"_{repeat}"
            view, info = shuffle_endpoints(graph, current_seed, independent)
            values["score_" + name] = frozen_score(model, view, divisor)
            diagnostics[name] = info
            del view
        view, info = degree_preserving_rewire(graph, current_seed)
        values[f"score_degree_rewire_{repeat}"] = frozen_score(model, view, divisor)
        diagnostics[f"degree_rewire_{repeat}"] = info
        del view
        oracle, random_cut, info = oracle_span_cut(graph, sample, current_seed)
        if repeat == 0:
            values["score_oracle_span_cut"] = frozen_score(model, oracle, divisor)
        values[f"score_matched_random_cut_{repeat}"] = frozen_score(model, random_cut, divisor)
        diagnostics[f"oracle_cut_{repeat}"] = info
        del oracle, random_cut
    return values, diagnostics


def missing_prefix_control(model, graph, sample, sites=8):
    """Fill a MISSING prefix audit on a disclosed diagnostic subset, not all-token evaluation."""
    if "score_prefix" in sample:
        return {}, {"prefix": "Existing saved prefix scores retained without changing coverage."}
    count = len(sample["gold"])
    chosen = np.unique(np.r_[np.linspace(0, count - 1, min(sites, count), dtype=int),
                             np.flatnonzero(sample["onset"])])
    result = np.full(count, np.nan)
    for token in chosen:
        cropped = prefix_graph(graph, int(graph["prompt_length"]) + int(token) + 1)
        result[token] = frozen_score(model, cropped, degree(cropped, model.normalization))[-1]
    return {"score_prefix": result}, {"prefix": "Uniform positions plus gold onsets; diagnostic subset; degree recomputed.",
                                      "prefix_tokens": chosen.tolist()}


def resolve_checkpoint(settings, override=None):
    saved = settings["checkpoint"]
    return Path(override if override else saved[0] if isinstance(saved, list) else saved)


def capture_samples(samples, settings, output, prepared=None, checkpoint=None,
                    device="cpu", repeats=3, max_lag=32, edge_chunk=4096, replay_atol=2e-5):
    """Exact replay is a prerequisite; mismatch stops attribution, never refits."""
    output = Path(output)
    directory = output / "captures"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(settings, checkpoint)
    variant = settings["variant"]
    normalization = "out" if variant == "charm_out" else "in"
    model, hp = load_checkpoint(checkpoint, device, normalization, edge_chunk)
    config = dict(checkpoint=str(checkpoint.resolve()), checkpoint_size=checkpoint.stat().st_size,
                  checkpoint_mtime=checkpoint.stat().st_mtime_ns, settings=settings, repeats=repeats,
                  max_lag=max_lag, replay_atol=replay_atol, software="deep-audit-v1")
    config_path = output / "capture_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Capture configuration changed; use a new --output directory")
    write_json(config_path, config)
    enriched, pairs, diagnostics = [], [], []
    for sample in tqdm(samples, desc="frozen CHARM audit", unit="answer"):
        graph, path = read_graph(sample, prepared)
        seed = zlib.crc32(sample["id"].encode()) + settings["seed"]
        view, _ = transform_graph(model_graph(graph), variant, seed)
        destination = directory / (sample["id"] + ".npz")
        stamp = [path.stat().st_size, path.stat().st_mtime_ns]
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if saved["graph_stamp"].tolist() != stamp:
                    raise ValueError("Prepared graph changed; stale additional capture")
                values = {k: saved[k] for k in saved.files if k.startswith("score_")}
                feature_values = {k[8:]: saved[k] for k in saved.files if k.startswith("feature_")}
                diagnostic = json.loads(str(saved["diagnostic_json"]))
                replay = saved["replay_score"]
        else:
            replay, states = trace_model(model, view)
            delta = float(np.max(abs(replay - sample["score"])))
            if delta > replay_atol:
                raise ValueError(f"{sample['id']}: saved-score replay mismatch {delta:.6g}; no attribution performed")
            feature_values = features(view, states)
            values, diagnostic = capture_controls(model, view, sample, seed, repeats)
            prefix, prefix_info = missing_prefix_control(model, view, sample)
            values.update(prefix)
            diagnostic.update(prefix_info)
            diagnostic["replay_max_error"] = delta
            arrays = {"feature_" + name: value for name, value in feature_values.items()}
            with destination.with_suffix(".partial").open("wb") as stream:
                np.savez_compressed(stream, **arrays, **values, replay_score=replay,
                                    graph_stamp=np.asarray(stamp), diagnostic_json=np.asarray(json.dumps(diagnostic)))
            destination.with_suffix(".partial").replace(destination)
        if not np.allclose(replay, sample["score"], rtol=0, atol=replay_atol):
            raise ValueError("Saved prediction changed since additional capture")
        enriched.append(dict(sample, **values))
        diagnostics.append(dict(id=sample["id"], source_id=sample["source_id"], **diagnostic))
        pairs.extend(relation_rows(sample, view, feature_values, graph["token_ids"], max_lag))
        del graph, view, feature_values
    write_json(output / "intervention_integrity.json", diagnostics)
    write_csv(output / "relation_pairs.csv", pairs)
    contrasts = matched_relation_contrasts(pairs)
    write_csv(output / "matched_span_relations.csv", contrasts)
    write_json(output / "representation_summary.json", relation_summary(contrasts))
    return enriched, model


def relation_summary(contrasts, bootstrap=1000):
    result = dict(matched_answer_lag_cells=len(contrasts), sources=len({r["source_id"] for r in contrasts}),
                  note="Same answer, exact lag, connected status and repeated token ID; unmatched cells are not substituted.")
    if not contrasts:
        result["source_mean_differences"] = None
        return result
    fields = [name for name in contrasts[0] if name.endswith("_same_minus_different")]
    result["source_mean_differences"] = {}
    for field in fields:
        sources = {}
        for row in contrasts:
            sources.setdefault(row["source_id"], []).append(row[field])
        means = [float(np.mean(values)) for values in sources.values()]
        rng = np.random.default_rng(42)
        draws = [float(np.mean(rng.choice(means, len(means), replace=True)))
                 for _ in range(bootstrap if len(means) > 1 else 0)]
        result["source_mean_differences"][field] = dict(
            mean=float(np.mean(means)), sources=len(means),
            source_bootstrap_ci95=np.quantile(draws, [.025, .975]).tolist() if draws else None,
            note="Exploratory interval, no multiple-comparison correction.")
    return result
