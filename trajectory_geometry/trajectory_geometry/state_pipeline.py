"""End-to-end, label-free fitting and encoding for the graph state model."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .data import load_attention_sample
from .hidden import AttentionHiddenPair, load_hidden_sample
from .state_model import (
    CONTROL_NAMES,
    SCHEMA,
    GraphStateModel,
    StateModelConfig,
    encode_graph_state,
    fit_graph_state_model,
    save_graph_state_model,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _encode_split(
    pairs: list[AttentionHiddenPair],
    output_dir: Path,
    model: GraphStateModel,
    split: str,
) -> dict[str, object]:
    directory = output_dir / split
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.jsonl"
    temporary_index = directory / "index.jsonl.tmp"
    records = []
    graph_gain = []
    rewire_gap = []
    with temporary_index.open("w", encoding="utf-8") as index_handle:
        for index, pair in enumerate(pairs, start=1):
            before = time.time()
            attention = load_attention_sample(pair.attention_path)
            hidden = load_hidden_sample(pair.hidden_path)
            states, offset = hidden.align(attention)
            encoding = encode_graph_state(attention, states, offset, model)
            output = directory / f"state_{pair.sample_id}.npz"
            temporary = output.with_suffix(".npz.tmp.npz")
            np.savez_compressed(
                temporary,
                schema=np.asarray(SCHEMA),
                labels_included=np.asarray(False),
                sample_id=np.asarray(pair.sample_id),
                response_idx=np.asarray(attention.response_idx, dtype=np.int32),
                response_token_ids=attention.token_ids[attention.response_idx :].astype(
                    np.int64, copy=False
                ),
                node_control_embedding=encoding.embeddings["node_control"].astype(np.float16),
                true_graph_embedding=encoding.embeddings["true_graph"].astype(np.float16),
                rewired_graph_embedding=encoding.embeddings["rewired_graph"].astype(np.float16),
                node_control_layer_mse=encoding.raw_residual_norm["node_control"].astype(
                    np.float16
                ),
                true_graph_layer_mse=encoding.raw_residual_norm["true_graph"].astype(
                    np.float16
                ),
                rewired_graph_layer_mse=encoding.raw_residual_norm["rewired_graph"].astype(
                    np.float16
                ),
                graph_gain=encoding.graph_gain,
                rewire_gap=encoding.rewire_gap,
                route_controls=encoding.controls.astype(np.float16),
                route_control_names=np.asarray(CONTROL_NAMES),
            )
            temporary.replace(output)
            row = {
                "sample_id": pair.sample_id,
                "attention": str(pair.attention_path),
                "hidden": str(pair.hidden_path),
                "output": str(output.resolve()),
                "response_tokens": attention.response_tokens,
            }
            records.append(row)
            index_handle.write(json.dumps(row, sort_keys=True) + "\n")
            graph_gain.append(encoding.graph_gain)
            rewire_gap.append(encoding.rewire_gap)
            print(
                f"[{split} {index}/{len(pairs)}] {pair.sample_id}: "
                f"{attention.response_tokens} tokens, {time.time() - before:.2f}s",
                flush=True,
            )
    temporary_index.replace(index_path)
    all_graph_gain = np.concatenate(graph_gain)
    all_rewire_gap = np.concatenate(rewire_gap)
    return {
        "samples": len(pairs),
        "response_tokens": int(all_graph_gain.size),
        "graph_gain_mean": float(all_graph_gain.mean()),
        "graph_gain_median": float(np.median(all_graph_gain)),
        "rewire_gap_mean": float(all_rewire_gap.mean()),
        "rewire_gap_median": float(np.median(all_rewire_gap)),
        "index": str(index_path.resolve()),
    }


def run_graph_state_pipeline(
    train_pairs: list[AttentionHiddenPair],
    test_pairs: list[AttentionHiddenPair],
    output_dir: str | Path,
    *,
    config: StateModelConfig | None = None,
    save_train_embeddings: bool = True,
) -> dict[str, object]:
    config = config or StateModelConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("[1/4] fitting one shared hidden-state projection on train only", flush=True)
    model = fit_graph_state_model(train_pairs, config)
    model_path = output_dir / "graph_state_model.npz"
    save_graph_state_model(model, model_path)
    print(
        "[2/4] frozen label-free calibration: "
        + json.dumps(model.calibration, sort_keys=True),
        flush=True,
    )
    splits: dict[str, object] = {}
    if save_train_embeddings:
        print("[3/4] encoding train node representations", flush=True)
        splits["train"] = _encode_split(train_pairs, output_dir, model, "train")
    else:
        print("[3/4] train embedding export skipped", flush=True)
    print("[4/4] encoding test node representations without reading labels", flush=True)
    splits["test"] = _encode_split(test_pairs, output_dir, model, "test")
    manifest = {
        "schema": SCHEMA,
        "state": "complete",
        "labels_read": False,
        "model": str(model_path.resolve()),
        "config": config.__dict__,
        "projection_input_dim": model.projection.input_dim,
        "projection_dim": model.projection.output_dim,
        "attention_layers": model.attention_layers,
        "modeled_transitions": model.transitions,
        "attention_layer_offset": model.attention_layer_offset,
        "heads": model.heads,
        "head_components": config.head_components,
        "node_embedding_dim": min(model.transitions, config.dct_components)
        * model.projection.output_dim,
        "calibration": model.calibration,
        "splits": splits,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest
