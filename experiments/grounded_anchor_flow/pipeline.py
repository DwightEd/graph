"""Build source-resolved anchor-flow artifacts from functional message graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .flow import analyze_flow, flow_to_dict

STATE_DIRECTORY = "grounded_anchor_flow"
CHANNELS = {
    "functional": "positive_function",
    "attention": "attention",
    "message": "residual_message_norm",
}
CONTROL_FIELDS = (
    "response_seeded_path_share",
    "response_seeded_anchor_flow",
    "direct_response_share",
    "valid",
    "anchor_valid",
)


def read_index(root: Path) -> list[dict[str, Any]]:
    path = root / "index.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Run one graph operator on functional capacity and two matched controls."""

    if graph.get("schema") != "functional-message-graph-v2":
        raise ValueError("grounded flow requires functional-message-graph-v2")
    result: dict[str, Any] = {
        "schema": "grounded-anchor-flow-v1",
        "sample_id": str(graph["sample_id"]),
        "source_id": str(graph["source_id"]),
        "task_type": str(graph["task_type"]),
        "generator_model": graph.get("generator_model"),
        "response_start": int(graph["response_start"]),
        "target_logprob": torch.as_tensor(graph["target_logprob"]).float(),
        "labels_used": False,
    }
    functional = analyze_flow(graph, channel=CHANNELS["functional"])
    result.update(
        {f"functional_{name}": value for name, value in flow_to_dict(functional).items()}
    )
    for prefix in ("attention", "message"):
        flow = flow_to_dict(analyze_flow(graph, channel=CHANNELS[prefix]))
        result.update({f"{prefix}_{name}": flow[name] for name in CONTROL_FIELDS})
    return result


def build_split(
    graph_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    graph_root = Path(graph_root)
    output_root = Path(output_root)
    sample_dir = output_root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = read_index(output_root)
    completed = {str(row["sample_id"]) for row in rows}
    graph_rows = read_index(graph_root)
    written = 0

    for row in graph_rows:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        if limit is not None and written >= limit:
            break
        graph = torch.load(
            graph_root / row["path"], map_location="cpu", weights_only=False
        )
        artifact = analyze_graph(graph)
        path = sample_dir / f"{sample_id}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(artifact, temporary)
        temporary.replace(path)
        record = {
            "sample_id": sample_id,
            "source_id": artifact["source_id"],
            "task_type": artifact["task_type"],
            "generator_model": artifact["generator_model"],
            "path": str(path.relative_to(output_root)),
            "response_tokens": len(artifact["functional_response_seeded_path_share"]),
        }
        with (output_root / "index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        rows.append(record)
        completed.add(sample_id)
        written += 1

    complete = limit is None and len(rows) == len(graph_rows)
    manifest = {
        "schema": "grounded-anchor-flow-v1",
        "graph_root": str(graph_root.resolve()),
        "samples": len(rows),
        "complete": complete,
        "labels_used": False,
        "channels": CHANNELS,
        "primary": "functional_response_seeded_anchor_flow",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def build_all(
    graph_state_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return [
        build_split(
            Path(graph_state_root) / split,
            Path(output_root) / STATE_DIRECTORY / split,
            limit=limit,
        )
        for split in ("train", "test")
    ]
