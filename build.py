"""Build graph views from one canonical feature split."""

from dataclasses import dataclass
from itertools import islice
import json
import math
from numbers import Real
from pathlib import Path

import torch
from tqdm import tqdm

from cache import AttentionDataset, sha256
from graphs import build_original_graph, build_relation_topk_graph
from hypergraph import build_attention_hypergraph


GRAPH_KINDS = ("original", "relation_topk", "relation_topk_channels", "hypergraph")
TOKEN_GRAPH_SCHEMA = "ragtruth-token-graph-v1"
HYPERGRAPH_SCHEMA = "ragtruth-attention-hypergraph-v1"


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
    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def run(self):
        self._validate_config()
        cache, output = Path(self.config.cache_dir), Path(self.config.output_dir)
        if cache.resolve() == output.resolve():
            raise ValueError("cache_dir and output_dir must differ")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("output_dir must be empty")
        input_manifest_sha256 = sha256(cache / "manifest.json")
        input_index_sha256 = sha256(cache / "index.jsonl")
        dataset = AttentionDataset(cache, self.config.device, verify_hashes=True)
        if self.config.kind in ("original", "hypergraph") and self.config.tau < dataset.attention_floor:
            raise ValueError("tau cannot be lower than attention_floor")
        graphs_dir = output / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        samples = dataset if self.config.limit is None else islice(dataset, self.config.limit)
        total = len(dataset) if self.config.limit is None else min(len(dataset), self.config.limit)
        for sample in tqdm(samples, total=total, desc=f"build {self.config.kind}"):
            graph = self._build(sample)
            path = graphs_dir / f"{sample.sample_id}.pt"
            torch.save(
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in graph.to_dict().items()},
                path,
            )
            row = {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "path": path.relative_to(output).as_posix(),
                "num_nodes": graph.num_nodes,
            }
            row["num_hyperedges" if self.config.kind == "hypergraph" else "num_edges"] = (
                int(graph.hyperedge_target.numel())
                if self.config.kind == "hypergraph"
                else int(graph.edge_index.shape[1])
            )
            row["sha256"] = sha256(path)
            row["bytes"] = path.stat().st_size
            rows.append(row)

        index = output / "index.jsonl"
        index.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        index_sha256 = sha256(index)
        if (input_manifest_sha256 != sha256(cache / "manifest.json")
                or input_index_sha256 != sha256(cache / "index.jsonl")):
            raise ValueError("input manifest or index changed during graph build")
        manifest = {
            "schema": HYPERGRAPH_SCHEMA if self.config.kind == "hypergraph" else TOKEN_GRAPH_SCHEMA,
            "representation": "sparse_attention_hypergraph" if self.config.kind == "hypergraph" else "sparse_causal_token_graph",
            "kind": self.config.kind,
            "count": len(rows),
            "attention_floor": dataset.attention_floor,
            "num_layers": dataset.manifest["num_layers"],
            "num_heads": dataset.manifest["num_heads"],
            "alignment": dataset.manifest["alignment"],
            "input_manifest_sha256": input_manifest_sha256,
            "input_index_sha256": input_index_sha256,
            "index_sha256": index_sha256,
            "parameters": self._parameters(),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    def _build(self, sample):
        if self.config.kind == "original":
            return build_original_graph(sample, self.config.tau)
        if self.config.kind == "relation_topk":
            return build_relation_topk_graph(sample, self.config.k_prompt, self.config.k_history, False)
        if self.config.kind == "relation_topk_channels":
            return build_relation_topk_graph(sample, self.config.k_prompt, self.config.k_history, True)
        if self.config.kind == "hypergraph":
            return build_attention_hypergraph(sample, self.config.tau)
        raise ValueError(f"unknown graph kind: {self.config.kind}")

    def _validate_config(self) -> None:
        if self.config.kind not in GRAPH_KINDS:
            raise ValueError(f"unknown graph kind: {self.config.kind}")
        if not isinstance(self.config.tau, Real) or isinstance(self.config.tau, bool) or not math.isfinite(self.config.tau) or not 0 < self.config.tau <= 1:
            raise ValueError("tau must be finite and in (0, 1]")
        for name, value in (("k_prompt", self.config.k_prompt), ("k_history", self.config.k_history)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.config.limit is not None and (not isinstance(self.config.limit, int) or isinstance(self.config.limit, bool) or self.config.limit < 1):
            raise ValueError("limit must be a positive integer")

    def _parameters(self) -> dict[str, float | int]:
        if self.config.kind in ("original", "hypergraph"):
            return {"tau": self.config.tau}
        return {"k_prompt": self.config.k_prompt, "k_history": self.config.k_history}
