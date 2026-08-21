from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from experiments.source_reuse_contrast.grounding_config import GroundingGraphConfig


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
        labels: list[int] | None = None,
    ):
        self.sample_id = sample_id
        self.source_id = f"source-{sample_id}"
        self.task_type = "QA"
        self.labels = torch.tensor(labels or [0] * num_response_tokens, dtype=torch.int8)
        layer, head, query, source, weight = edges
        self._block = SyntheticBlock(
            layer=torch.tensor(layer, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            query=torch.tensor(query, dtype=torch.long),
            source=torch.tensor(source, dtype=torch.long),
            weight=torch.tensor(weight, dtype=torch.float32),
        )
        if diagonal is None:
            diagonal = torch.full(
                (num_layers, num_heads, response_idx + num_response_tokens),
                0.1,
                dtype=torch.float32,
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


def sequence_sample(sample_id="synthetic", labels=None):
    return SyntheticSample(
        num_layers=2,
        num_heads=2,
        response_idx=2,
        num_response_tokens=4,
        edges=(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 0, 1, 1, 0, 0, 2, 2, 1, 1, 3, 3, 0, 0, 4, 4],
            [0.4, 0.3, 0.25, 0.2, 0.3, 0.25, 0.35, 0.3, 0.2, 0.2, 0.4, 0.35, 0.15, 0.15, 0.45, 0.4],
        ),
        sample_id=sample_id,
        labels=labels,
    )


def tiny_config(**updates):
    values = dict(
        hidden_dim=16,
        layer_embedding_dim=4,
        head_embedding_dim=4,
        relation_embedding_dim=3,
        lag_embedding_dim=3,
        response_lag_bins=4,
        received_topk=2,
        edge_mask_rate=0.2,
        perturbation_scale=0.05,
        gate_keep_target=0.7,
        gate_regularization=0.01,
        raw_loss_weight=0.1,
        reuse_loss_weight=1.0,
        grounding_loss_weight=1.0,
        provenance_loss_weight=0.5,
        use_reuse_memory=True,
        dropout=0.0,
        bptt_steps=8,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=5.0,
        validation_fraction=0.25,
        early_stopping_patience=1,
        score_rounds=2,
        block_rows=128,
        random_seed=7,
        show_progress=False,
    )
    values.update(updates)
    return GroundingGraphConfig(**values)


class SyntheticLabelStore:
    def response_labels(self, sample):
        return sample.labels


class SyntheticDataset:
    def __init__(self, samples):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        return SyntheticLabelStore()
