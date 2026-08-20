"""Small head-preserving layer encoder with a causal token-time model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class TokenLogits:
    current_logits: torch.Tensor
    next_logits: torch.Tensor


@dataclass(frozen=True)
class HeadSequence:
    """One variable-length labeled response used by supervised training."""

    sample_id: str
    source_id: str
    task_type: str
    values: torch.Tensor
    labels: torch.Tensor

    def validate(self) -> "HeadSequence":
        if self.values.ndim != 4:
            raise ValueError("sequence values must be [token, layer, head, feature]")
        if self.labels.shape != (len(self.values),):
            raise ValueError("sequence labels do not align with token features")
        if not bool(torch.isfinite(self.values).all()):
            raise FloatingPointError("sequence values must be finite")
        if not bool(((self.labels == 0) | (self.labels == 1)).all()):
            raise ValueError("sequence labels must be binary")
        return self


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 16
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    forecast_weight: float = 0.5
    patience: int = 5
    maximum_standardized_value: float = 10.0
    seed: int = 20260820


@dataclass(frozen=True)
class HeadFeatureNormalizer:
    center: torch.Tensor
    scale: torch.Tensor
    maximum_value: float

    @classmethod
    def fit(
        cls,
        sequences: Sequence[HeadSequence],
        *,
        maximum_value: float,
    ) -> "HeadFeatureNormalizer":
        total = None
        square_total = None
        count = 0
        for sequence in sequences:
            values = sequence.validate().values.double()
            if total is None:
                total = torch.zeros_like(values[0])
                square_total = torch.zeros_like(values[0])
            elif values.shape[1:] != total.shape:
                raise ValueError("training sequence geometry changes between samples")
            total += values.sum(dim=0)
            square_total += values.square().sum(dim=0)
            count += len(values)
        if total is None or count < 2:
            raise ValueError("normalization requires at least two training tokens")
        center = total / count
        variance = (square_total / count - center.square()).clamp_min(0.0)
        scale = variance.sqrt().clamp_min(1e-4)
        return cls(
            center=center.float(),
            scale=scale.float(),
            maximum_value=float(maximum_value),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        standardized = (values.float() - self.center) / self.scale
        return standardized.clamp(-self.maximum_value, self.maximum_value)


@dataclass(frozen=True)
class SequencePrediction:
    sample_id: str
    source_id: str
    task_type: str
    labels: np.ndarray
    current_probability: np.ndarray
    forecast_probability: np.ndarray


class CausalLayerTemporalModel(nn.Module):
    """Encode ordered layers per token, then model tokens left-to-right."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        num_features: int,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(num_layers, num_heads, num_features, hidden_dim) < 1:
            raise ValueError("model geometry must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_features = int(num_features)

        # Flattening here preserves the fixed identity of every head. A head
        # permutation changes the input coordinates; no head mean is taken.
        self.head_projection = nn.Linear(num_heads * num_features, hidden_dim)
        self.layer_position = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.layer_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.current_classifier = nn.Linear(hidden_dim, 1)
        self.next_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> TokenLogits:
        """Score ``[batch, token, layer, head, feature]`` causal sequences."""

        if values.ndim != 5:
            raise ValueError("model input must be [batch, token, layer, head, feature]")
        batch, tokens, layers, heads, features = values.shape
        if (layers, heads, features) != (
            self.num_layers,
            self.num_heads,
            self.num_features,
        ):
            raise ValueError("model input geometry differs from construction")

        per_layer = values.reshape(batch * tokens, layers, heads * features)
        per_layer = torch.tanh(self.head_projection(per_layer))
        per_layer = per_layer + self.layer_position.unsqueeze(0)
        _, final_layer_state = self.layer_encoder(self.dropout(per_layer))
        token_state = final_layer_state[-1].reshape(batch, tokens, -1)
        temporal_state, _ = self.temporal_encoder(self.dropout(token_state))
        temporal_state = self.dropout(temporal_state)
        return TokenLogits(
            current_logits=self.current_classifier(temporal_state).squeeze(-1),
            next_logits=self.next_classifier(temporal_state).squeeze(-1),
        )


def _binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return {"tokens": len(labels), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "tokens": len(labels),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


class CausalLayerTemporalDetector:
    """Train with source-disjoint validation and expose token-level evaluation."""

    def __init__(self, config: TrainingConfig | None = None, *, device: str = "cpu"):
        self.config = TrainingConfig() if config is None else config
        self.device = torch.device(device)
        self.model: CausalLayerTemporalModel | None = None
        self.normalizer: HeadFeatureNormalizer | None = None
        self.best_epoch = 0

    def fit(
        self,
        train: Sequence[HeadSequence],
        validation: Sequence[HeadSequence],
    ) -> list[dict[str, float | int]]:
        """Fit only on ``train`` and select the epoch only on ``validation``."""

        if not train or not validation:
            raise ValueError("training and validation sequences are required")
        train_sources = {sequence.source_id for sequence in train}
        validation_sources = {sequence.source_id for sequence in validation}
        overlap = train_sources & validation_sources
        if overlap:
            raise ValueError(f"train/validation source overlap: {sorted(overlap)[:3]}")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        geometry = train[0].validate().values.shape[1:]
        for sequence in (*train, *validation):
            if sequence.validate().values.shape[1:] != geometry:
                raise ValueError("model geometry changes between sequences")
        layers, heads, features = geometry
        self.normalizer = HeadFeatureNormalizer.fit(
            train,
            maximum_value=self.config.maximum_standardized_value,
        )
        self.model = CausalLayerTemporalModel(
            num_layers=layers,
            num_heads=heads,
            num_features=features,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)

        positives = sum(int(sequence.labels.sum().item()) for sequence in train)
        tokens = sum(len(sequence.labels) for sequence in train)
        if positives == 0 or positives == tokens:
            raise ValueError("training labels must contain both classes")
        positive_weight = torch.tensor(
            (tokens - positives) / positives,
            dtype=torch.float32,
            device=self.device,
        )
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight,
            reduction="none",
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        generator = torch.Generator().manual_seed(self.config.seed)
        history: list[dict[str, float | int]] = []
        best_auprc = -float("inf")
        best_state = None
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            order = torch.randperm(len(train), generator=generator).tolist()
            epoch_loss = 0.0
            epoch_tokens = 0
            for start in range(0, len(order), self.config.batch_size):
                batch = [train[index] for index in order[start : start + self.config.batch_size]]
                values, labels, valid = self._collate(batch)
                output = self.model(values)
                current_loss = loss_function(output.current_logits, labels)
                loss_sum = current_loss[valid].sum()
                loss_count = int(valid.sum().item())

                forecast_valid = valid[:, :-1] & valid[:, 1:]
                if bool(forecast_valid.any()):
                    forecast_loss = loss_function(
                        output.next_logits[:, :-1],
                        labels[:, 1:],
                    )
                    loss_sum = loss_sum + self.config.forecast_weight * forecast_loss[
                        forecast_valid
                    ].sum()
                    loss_count += int(forecast_valid.sum().item())

                loss = loss_sum / max(loss_count, 1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss_sum.detach().cpu())
                epoch_tokens += loss_count

            validation_metrics = self.evaluate(validation)["current"]
            validation_auprc = float(validation_metrics["auprc"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(epoch_tokens, 1),
                    "validation_auroc": float(validation_metrics["auroc"]),
                    "validation_auprc": validation_auprc,
                }
            )
            if np.isfinite(validation_auprc) and validation_auprc > best_auprc:
                best_auprc = validation_auprc
                self.best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.config.patience:
                break

        if best_state is None:
            raise RuntimeError("validation did not produce a finite selection metric")
        self.model.load_state_dict(best_state)
        return history

    def predict(self, sequences: Sequence[HeadSequence]) -> list[SequencePrediction]:
        if self.model is None or self.normalizer is None:
            raise RuntimeError("detector must be fitted before prediction")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for sequence in sequences:
                values = self.normalizer.transform(sequence.validate().values)
                output = self.model(values.unsqueeze(0).to(self.device))
                predictions.append(
                    SequencePrediction(
                        sample_id=sequence.sample_id,
                        source_id=sequence.source_id,
                        task_type=sequence.task_type,
                        labels=sequence.labels.cpu().numpy().astype(np.int8),
                        current_probability=output.current_logits.sigmoid()[0]
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        forecast_probability=output.next_logits.sigmoid()[0]
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                    )
                )
        return predictions

    def evaluate(self, sequences: Sequence[HeadSequence]) -> dict[str, object]:
        predictions = self.predict(sequences)
        labels = np.concatenate([prediction.labels for prediction in predictions])
        current = np.concatenate(
            [prediction.current_probability for prediction in predictions]
        )
        forecast_labels = np.concatenate(
            [prediction.labels[1:] for prediction in predictions if len(prediction.labels) > 1]
        )
        forecast = np.concatenate(
            [
                prediction.forecast_probability[:-1]
                for prediction in predictions
                if len(prediction.labels) > 1
            ]
        )
        by_task = {}
        for task in sorted({prediction.task_type for prediction in predictions}):
            selected = [prediction for prediction in predictions if prediction.task_type == task]
            task_labels = np.concatenate([prediction.labels for prediction in selected])
            task_score = np.concatenate(
                [prediction.current_probability for prediction in selected]
            )
            by_task[task] = _binary_metrics(task_labels, task_score)
        return {
            "current": _binary_metrics(labels, current),
            "forecast_1": _binary_metrics(forecast_labels, forecast),
            "by_task": by_task,
        }

    def save(self, path: Path) -> None:
        if self.model is None or self.normalizer is None:
            raise RuntimeError("cannot save an unfitted detector")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "center": self.normalizer.center,
                "scale": self.normalizer.scale,
                "maximum_standardized_value": self.normalizer.maximum_value,
                "training_config": self.config.__dict__,
                "geometry": {
                    "num_layers": self.model.num_layers,
                    "num_heads": self.model.num_heads,
                    "num_features": self.model.num_features,
                },
                "best_epoch": self.best_epoch,
            },
            path,
        )

    def _collate(
        self,
        sequences: Sequence[HeadSequence],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.normalizer is None:
            raise RuntimeError("normalizer is not fitted")
        maximum = max(len(sequence.values) for sequence in sequences)
        geometry = sequences[0].values.shape[1:]
        values = torch.zeros((len(sequences), maximum, *geometry), dtype=torch.float32)
        labels = torch.zeros((len(sequences), maximum), dtype=torch.float32)
        valid = torch.zeros((len(sequences), maximum), dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            count = len(sequence.values)
            values[index, :count] = self.normalizer.transform(sequence.values)
            labels[index, :count] = sequence.labels.float()
            valid[index, :count] = True
        return values.to(self.device), labels.to(self.device), valid.to(self.device)
