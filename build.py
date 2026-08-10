"""Build graph views from one canonical feature split."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from tqdm import tqdm

from cache import AttentionDataset
from features import NODE_FEATURE_MODES, load_node_features
from graphs import build_original_graph, build_relation_topk_graph
from hypergraph import build_attention_hypergraph


GRAPH_KINDS = ("original", "relation_topk", "relation_topk_channels", "hypergraph")


@dataclass(frozen=True)
class BuildConfig:
    cache_dir: str | Path
    output_dir: str | Path
    kind: str = "relation_topk_channels"
    tau: float = 0.05
    k_prompt: int = 8
    k_history: int = 8
    node_features: str = "attention"
    device: str = "cuda"
    limit: int | None = None


class GraphDatasetBuilder:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def run(self):
        if self.config.node_features not in NODE_FEATURE_MODES:
            raise ValueError(f"unknown node feature mode: {self.config.node_features}")
        dataset = AttentionDataset(self.config.cache_dir, self.config.device)
        output = Path(self.config.output_dir)
        graphs_dir = output / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, sample in enumerate(tqdm(dataset, desc=f"build {self.config.kind}")):
            if self.config.limit is not None and i >= self.config.limit:
                break
            x = load_node_features(self.config.cache_dir, sample, self.config.node_features)
            graph = self._build(sample, x)
            path = graphs_dir / f"{sample.sample_id}.pt"
            torch.save(
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in graph.to_dict().items()},
                path,
            )
            row = {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "path": path.relative_to(output).as_posix(),
                "num_nodes": sample.num_tokens,
            }
            row["num_hyperedges" if self.config.kind == "hypergraph" else "num_edges"] = (
                int(graph.hyperedge_target.numel())
                if self.config.kind == "hypergraph"
                else int(graph.edge_index.shape[1])
            )
            rows.append(row)

        (output / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        manifest = {
            "kind": self.config.kind,
            "node_features": self.config.node_features,
            "count": len(rows),
        }
        if self.config.kind in ("original", "hypergraph"):
            manifest["tau"] = self.config.tau
        else:
            manifest.update(k_prompt=self.config.k_prompt, k_history=self.config.k_history)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    def _build(self, sample, x):
        if self.config.kind == "original":
            return build_original_graph(sample, self.config.tau, x)
        if self.config.kind == "relation_topk":
            return build_relation_topk_graph(sample, self.config.k_prompt, self.config.k_history, False, x)
        if self.config.kind == "relation_topk_channels":
            return build_relation_topk_graph(sample, self.config.k_prompt, self.config.k_history, True, x)
        if self.config.kind == "hypergraph":
            return build_attention_hypergraph(sample, self.config.tau, x)
        raise ValueError(f"unknown graph kind: {self.config.kind}")
