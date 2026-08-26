"""Source-disjoint supervised probes for representation readability."""

from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .detectors import RobustScale
from .models import NodeMLP


@dataclass(frozen=True)
class ProbeConfig:
    folds: int = 5
    hidden_dim: int = 128
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    seeds: tuple[int, ...] = (20260825, 20260826, 20260827)


def source_folds(
    label: np.ndarray,
    source_id: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    fold_id = np.empty(len(label), dtype=np.int16)
    rows = np.zeros((len(label), 1), dtype=np.float32)
    for fold, (_, test) in enumerate(splitter.split(rows, label, source_id)):
        fold_id[test] = fold
    return fold_id


def linear_probe(
    feature: np.ndarray,
    label: np.ndarray,
    fold_id: np.ndarray,
    seed: int,
) -> np.ndarray:
    score = np.empty(len(label), dtype=np.float32)
    for fold in range(int(fold_id.max()) + 1):
        train = fold_id != fold
        test = ~train
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            ),
        )
        model.fit(feature[train], label[train])
        score[test] = model.predict_proba(feature[test])[:, 1]
    return score


def mlp_probe(
    feature: np.ndarray,
    label: np.ndarray,
    fold_id: np.ndarray,
    config: ProbeConfig,
    device: str,
) -> np.ndarray:
    seed_scores = []
    for seed in config.seeds:
        score = np.empty(len(label), dtype=np.float32)
        for fold in range(int(fold_id.max()) + 1):
            train = fold_id != fold
            test = ~train
            scale = RobustScale.fit(feature[train])
            model = fit_mlp(
                scale.transform(feature[train]),
                label[train],
                config,
                device,
                seed + fold,
            )
            score[test] = predict_mlp(
                model,
                scale.transform(feature[test]),
                config,
                device,
            )
        seed_scores.append(score)
    return np.mean(seed_scores, axis=0).astype(np.float32)


def fit_mlp(
    feature: np.ndarray,
    label: np.ndarray,
    config: ProbeConfig,
    device: str,
    seed: int,
) -> NodeMLP:
    set_seed(seed)
    model = NodeMLP(feature.shape[1], config.hidden_dim).to(device)
    positive = max(int(label.sum()), 1)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(label) - positive) / positive, device=device)
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(feature),
            torch.from_numpy(label.astype(np.float32)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        model.train()
        for embedding, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(embedding.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
    return model


def predict_mlp(
    model: NodeMLP,
    feature: np.ndarray,
    config: ProbeConfig,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(feature)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    score = []
    model.eval()
    with torch.no_grad():
        for (embedding,) in loader:
            score.append(torch.sigmoid(model(embedding.to(device))).cpu().numpy())
    return np.concatenate(score).astype(np.float32)


def readability_scores(
    embeddings: dict[str, np.ndarray],
    label: np.ndarray,
    source_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    config: ProbeConfig | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    config = ProbeConfig() if config is None else config
    fold_id = source_folds(label, source_id, config.folds, config.seeds[0])
    position = np.column_stack(
        (
            token_index / np.maximum(response_length - 1, 1),
            np.log1p(response_length),
        )
    ).astype(np.float32)

    scores = {
        "linear_position": linear_probe(position, label, fold_id, config.seeds[0]),
        "position_mlp": mlp_probe(position, label, fold_id, config, device),
    }
    for name, embedding in embeddings.items():
        scores[f"linear_node__{name}"] = linear_probe(
            embedding,
            label,
            fold_id,
            config.seeds[0],
        )
        scores[f"node_mlp__{name}"] = mlp_probe(
            embedding,
            label,
            fold_id,
            config,
            device,
        )
    return scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
