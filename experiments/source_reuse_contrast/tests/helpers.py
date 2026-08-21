from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from experiments.source_reuse_contrast.config import SourceReuseConfig


@dataclass(frozen=True)
class SyntheticBlock:
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor


class SyntheticSample:
    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        response_idx: int,
        num_response_tokens: int,
        edges: tuple[list[int], list[int], list[int], list[int], list[float]],
        diagonal: torch.Tensor | None = None,
        sample_id: str = "synthetic",
        source_id: str | None = None,
        task_type: str = "QA",
        labels: list[int] | None = None,
    ):
        self.sample_id = sample_id
        self.source_id = source_id or f"source-{sample_id}"
        self.task_type = task_type
        self.labels = torch.tensor(
            labels if labels is not None else [0] * num_response_tokens,
            dtype=torch.int8,
        )
        layer, head, query, source, weight = edges
        if diagonal is None:
            diagonal = torch.zeros(
                (num_layers, num_heads, response_idx + num_response_tokens),
                dtype=torch.float32,
            )
        self._block = SyntheticBlock(
            layer=torch.tensor(layer, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            query=torch.tensor(query, dtype=torch.long),
            source=torch.tensor(source, dtype=torch.long),
            weight=torch.tensor(weight, dtype=torch.float32),
        )
        self._attention = SimpleNamespace(
            response_idx=response_idx,
            num_response_tokens=num_response_tokens,
            num_tokens=response_idx + num_response_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_floor=0.01,
            response_values=self._block.weight,
            attention_diagonal=diagonal,
        )

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=8192):
        yield self._block

    def release_attention(self):
        return None


class SyntheticLabelStore:
    def response_labels(self, sample):
        return sample.labels


class SyntheticDataset:
    def __init__(self, samples: list[SyntheticSample]):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return SyntheticLabelStore()


def tiny_config(**updates) -> SourceReuseConfig:
    values = dict(
        hidden_dim=16,
        layer_embedding_dim=4,
        head_embedding_dim=4,
        relation_embedding_dim=3,
        source_bin_embedding_dim=3,
        usage_embedding_dim=2,
        prompt_position_bins=1,
        response_lag_bins=2,
        usage_bins=3,
        memory_mode="dynamic",
        temperature=0.5,
        negative_count=2,
        negative_pool_size=4,
        prompt_position_tolerance=1.0,
        response_lag_tolerance=1.0,
        dropout=0.0,
        bptt_steps=8,
        epochs=2,
        learning_rate=2e-3,
        weight_decay=0.0,
        gradient_clip=5.0,
        validation_fraction=0.25,
        early_stopping_patience=2,
        score_rounds=2,
        block_rows=128,
        random_seed=7,
        show_progress=False,
    )
    values.update(updates)
    return SourceReuseConfig(**values)


def sequence_sample(
    *,
    sample_id="sequence",
    source_id=None,
    task_type="QA",
    labels=None,
    extra_token=False,
) -> SyntheticSample:
    query = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    source = [0, 1, 0, 2, 4, 1, 4, 5, 2, 5, 6, 4, 6]
    layer = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    head = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    weight = [0.4, 0.3, 0.5, 0.2, 0.3, 0.2, 0.35, 0.25, 0.2, 0.35, 0.25, 0.4, 0.3]
    response_tokens = 5
    if extra_token:
        query.extend([5, 5])
        source.extend([5, 7])
        layer.extend([0, 1])
        head.extend([1, 0])
        weight.extend([0.35, 0.25])
        response_tokens = 6
    return SyntheticSample(
        num_layers=2,
        num_heads=2,
        response_idx=4,
        num_response_tokens=response_tokens,
        edges=(layer, head, query, source, weight),
        sample_id=sample_id,
        source_id=source_id,
        task_type=task_type,
        labels=labels,
    )
