"""Source-disjoint node-only supervised readability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .detectors import RobustEmbeddingScale
from .model import NodeMLP


@dataclass(frozen=True)
class ProbeConfig:
    folds: int = 5
    hidden_dim: int = 128
    dropout: float = 0.1
    epochs: int = 20
    patience: int = 4
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    split_seed: int = 20260825
    seeds: tuple[int, ...] = (20260825, 20260826, 20260827)


@dataclass(frozen=True)
class ProbePredictions:
    fold_id: np.ndarray
    score: Mapping[str, np.ndarray]
    seed_score: Mapping[str, np.ndarray]


def source_group_folds(
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign every row to one stratified fold without splitting a source."""

    label = np.asarray(label, dtype=np.int8)
    source_id = np.asarray(source_id).astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    fold_id = np.full(len(label), -1, dtype=np.int16)
    rows = np.zeros((len(label), 1), dtype=np.float32)
    for fold, (_, test) in enumerate(splitter.split(rows, label, source_id)):
        fold_id[test] = fold
    if bool((fold_id < 0).any()):
        raise RuntimeError("source-group folds did not cover every row")
    for fold in range(folds):
        if np.unique(label[fold_id == fold]).size < 2 or np.unique(
            label[fold_id != fold]
        ).size < 2:
            raise ValueError("every source-group fold must contain both classes")
    return fold_id


def fit_readability_probes(
    embeddings: Mapping[str, np.ndarray],
    label: np.ndarray,
    source_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    config: ProbeConfig | None = None,
    device: str = "cpu",
) -> ProbePredictions:
    """Return complete out-of-fold predictions from node features only."""

    config = ProbeConfig() if config is None else config
    label = np.asarray(label, dtype=np.int8)
    source_id = np.asarray(source_id).astype(str)
    fold_id = source_group_folds(
        label,
        source_id,
        folds=config.folds,
        seed=config.split_seed,
    )
    position = np.column_stack(
        (
            np.asarray(token_index, dtype=np.float32)
            / np.maximum(np.asarray(response_length, dtype=np.float32) - 1.0, 1.0),
            np.log1p(np.asarray(response_length, dtype=np.float32)),
        )
    )

    score: dict[str, np.ndarray] = {
        "linear_position": _linear_oof(position, label, fold_id, config.split_seed),
    }
    seed_score: dict[str, np.ndarray] = {}
    position_neural = []
    for seed in config.seeds:
        prediction = _mlp_oof(
            position,
            label,
            source_id,
            fold_id,
            config,
            seed,
            device,
        )
        seed_score[f"position_mlp__seed_{seed}"] = prediction
        position_neural.append(prediction)
    score["position_mlp"] = np.mean(position_neural, axis=0).astype(np.float32)
    for variant, embedding in embeddings.items():
        values = np.asarray(embedding, dtype=np.float32)
        if values.ndim != 2 or len(values) != len(label):
            raise ValueError("variant embeddings must align with labelled rows")
        score[f"linear_node__{variant}"] = _linear_oof(
            values,
            label,
            fold_id,
            config.split_seed,
        )
        neural = []
        for seed in config.seeds:
            prediction = _mlp_oof(
                values,
                label,
                source_id,
                fold_id,
                config,
                seed,
                device,
            )
            seed_score[f"node_mlp__{variant}__seed_{seed}"] = prediction
            neural.append(prediction)
        score[f"node_mlp__{variant}"] = np.mean(neural, axis=0).astype(np.float32)
    return ProbePredictions(fold_id=fold_id, score=score, seed_score=seed_score)


def _linear_oof(features, label, fold_id, seed: int) -> np.ndarray:
    prediction = np.empty(len(label), dtype=np.float32)
    for fold in range(int(fold_id.max()) + 1):
        test = fold_id == fold
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            ),
        )
        model.fit(features[train], label[train])
        prediction[test] = model.predict_proba(features[test])[:, 1]
    return prediction


def _mlp_oof(
    embedding,
    label,
    source_id,
    fold_id,
    config: ProbeConfig,
    seed: int,
    device: str,
) -> np.ndarray:
    prediction = np.empty(len(label), dtype=np.float32)
    for fold in range(int(fold_id.max()) + 1):
        outer_test = fold_id == fold
        outer_train = ~outer_test
        train_rows, validation_rows = _inner_split(
            outer_train,
            label,
            source_id,
            config.split_seed + fold + 1,
        )
        selection_scale = RobustEmbeddingScale.fit(embedding[train_rows])
        best_epochs = _select_mlp_epochs(
            selection_scale.transform(embedding[train_rows]),
            label[train_rows],
            selection_scale.transform(embedding[validation_rows]),
            label[validation_rows],
            config,
            seed + fold,
            device,
        )
        final_scale = RobustEmbeddingScale.fit(embedding[outer_train])
        model = _fit_fixed_mlp(
            final_scale.transform(embedding[outer_train]),
            label[outer_train],
            best_epochs,
            config,
            seed + fold,
            device,
        )
        prediction[outer_test] = _predict_mlp(
            model,
            final_scale.transform(embedding[outer_test]),
            config.batch_size,
            device,
        )
    return prediction


def _inner_split(outer_train, label, source_id, seed: int):
    outer_rows = np.flatnonzero(outer_train)
    group_count = len(np.unique(source_id[outer_rows]))
    splitter = StratifiedGroupKFold(
        n_splits=min(5, group_count),
        shuffle=True,
        random_state=seed,
    )
    inner_train, inner_validation = next(
        splitter.split(
            np.zeros((len(outer_rows), 1), dtype=np.float32),
            label[outer_rows],
            source_id[outer_rows],
        )
    )
    train_rows = outer_rows[inner_train]
    validation_rows = outer_rows[inner_validation]
    if (
        np.unique(label[train_rows]).size < 2
        or np.unique(label[validation_rows]).size < 2
    ):
        raise ValueError("inner source-group train/validation need both classes")
    return train_rows, validation_rows


def _select_mlp_epochs(
    train_embedding,
    train_label,
    validation_embedding,
    validation_label,
    config: ProbeConfig,
    seed: int,
    device: str,
):
    _seed(seed)
    model = NodeMLP(
        train_embedding.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positives = int(np.asarray(train_label).sum())
    positive_weight = (len(train_label) - positives) / positives
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_embedding),
            torch.from_numpy(np.asarray(train_label, dtype=np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_validation = -math.inf
    best_epoch = 1
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for embedding, label in loader:
            optimizer.zero_grad(set_to_none=True)
            logit = model(embedding.to(device))
            loss = loss_function(logit, label.to(device))
            loss.backward()
            optimizer.step()
        validation_score = _predict_mlp(
            model,
            validation_embedding,
            config.batch_size,
            device,
        )
        current = average_precision_score(validation_label, validation_score)
        if current > best_validation:
            best_validation = float(current)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    return best_epoch


def _fit_fixed_mlp(
    train_embedding,
    train_label,
    epochs: int,
    config: ProbeConfig,
    seed: int,
    device: str,
):
    _seed(seed)
    model = NodeMLP(
        train_embedding.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positives = int(np.asarray(train_label).sum())
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            (len(train_label) - positives) / positives,
            device=device,
        )
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_embedding),
            torch.from_numpy(np.asarray(train_label, dtype=np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for embedding, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(
                model(embedding.to(device)),
                label.to(device),
            )
            loss.backward()
            optimizer.step()
    return model


def _predict_mlp(model, embedding, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(embedding, dtype=np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    prediction = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            prediction.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(prediction).astype(np.float32)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
