"""Build one label-free token graph artifact at a time from attention caches."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from cache import load_attention_sample
from graphs import build_original_graph, build_relation_topk_graph
from hypergraph import build_attention_hypergraph


GRAPH_KINDS = (
    "original",
    "relation_topk",
    "relation_topk_channels",
    "hypergraph",
)


@dataclass(frozen=True)
class BuildConfig:
    cache_dir: str | Path
    output_dir: str | Path
    kind: str = "relation_topk_channels"
    tau: float = 0.05
    k_prompt: int = 8
    k_history: int = 8
    device: str = "cuda"
    limit: int | None = None


class GraphDatasetBuilder:
    """Convert a cache split directory into a graph dataset without labels."""

    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        self._validate_config()
        cache_dir = Path(self.config.cache_dir)
        output_dir = Path(self.config.output_dir)
        if cache_dir.resolve() == output_dir.resolve():
            raise ValueError("cache_dir and output_dir must differ")
        if not cache_dir.is_dir():
            raise ValueError("cache_dir must be an existing directory")
        cache_paths = sorted(cache_dir.glob("*.pt"))
        if not cache_paths:
            raise ValueError("cache_dir must contain at least one top-level .pt file")
        if self.config.limit is not None:
            cache_paths = cache_paths[: self.config.limit]
        self._validate_cache_threshold(cache_paths[0])

        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError("output_dir already contains files")
        output_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir = output_dir / "graphs"
        graphs_dir.mkdir()

        index_path = output_dir / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as index_file:
            for cache_path in tqdm(cache_paths, desc=f"build {self.config.kind}"):
                sample = load_attention_sample(cache_path, map_location=self.config.device)
                graph, schema = self._build_graph(sample)
                graph_path = graphs_dir / f"{cache_path.stem}.graph.pt"
                graph_payload = {
                    "schema": schema,
                    "graph": self._cpu_graph_dict(graph.to_dict()),
                }
                torch.save(graph_payload, graph_path)
                row = self._index_row(sample, graph, graph_path, output_dir)
                index_file.write(json.dumps(row) + "\n")

        manifest = self._manifest(cache_dir, len(cache_paths))
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return {"kind": self.config.kind, "count": len(cache_paths), "output_dir": str(output_dir)}

    def _validate_config(self) -> None:
        if self.config.kind not in GRAPH_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(GRAPH_KINDS)}")
        try:
            tau = float(self.config.tau)
        except (TypeError, ValueError) as error:
            raise ValueError("tau must be a finite number in [0, 1]") from error
        if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be finite and within [0, 1]")
        if self.config.k_prompt < 0 or self.config.k_history < 0:
            raise ValueError("k_prompt and k_history must be non-negative")
        if self.config.limit is not None and (
            isinstance(self.config.limit, bool)
            or not isinstance(self.config.limit, int)
            or self.config.limit <= 0
        ):
            raise ValueError("limit must be None or a positive integer")

    def _validate_cache_threshold(self, cache_path: Path) -> None:
        if self.config.kind not in ("original", "hypergraph"):
            return
        tau = float(self.config.tau)
        sample = load_attention_sample(cache_path, map_location="cpu")
        if tau < sample.attention_floor:
            raise ValueError("tau must be at least the retained cache attention_floor")

    def _build_graph(self, sample: Any) -> tuple[Any, str]:
        if self.config.kind == "original":
            return build_original_graph(sample, self.config.tau), "token-graph-v1"
        if self.config.kind == "relation_topk":
            return (
                build_relation_topk_graph(sample, self.config.k_prompt, self.config.k_history),
                "token-graph-v1",
            )
        if self.config.kind == "relation_topk_channels":
            return (
                build_relation_topk_graph(
                    sample,
                    self.config.k_prompt,
                    self.config.k_history,
                    with_channels=True,
                ),
                "token-graph-v1",
            )
        return build_attention_hypergraph(sample, self.config.tau), "attention-hypergraph-v1"

    def _index_row(
        self,
        sample: Any,
        graph: Any,
        graph_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        row = {
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "path": graph_path.relative_to(output_dir).as_posix(),
            "num_nodes": int(graph.token_ids.numel()),
        }
        if self.config.kind == "hypergraph":
            row["num_hyperedges"] = int(graph.hyperedge_target.numel())
        else:
            row["num_edges"] = int(graph.edge_index.shape[1])
        return row

    def _manifest(self, cache_dir: Path, count: int) -> dict[str, Any]:
        schema = "attention-hypergraph-v1" if self.config.kind == "hypergraph" else "token-graph-v1"
        manifest: dict[str, Any] = {
            "schema": schema,
            "kind": self.config.kind,
            "input_cache": str(cache_dir),
            "device": self.config.device,
            "count": count,
        }
        if self.config.kind in ("original", "hypergraph"):
            manifest["tau"] = self.config.tau
        else:
            manifest["k_prompt"] = self.config.k_prompt
            manifest["k_history"] = self.config.k_history
        if self.config.kind == "original":
            manifest["compatibility_dense_channel_mode"] = True
        return manifest

    @staticmethod
    def _cpu_graph_dict(graph: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for name, value in graph.items()
        }
