"""Fixed node-only anomaly detectors for frozen GroundedRoute embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from ..detection import PCAKNNConfig, PCAWhitenedKNN
from .model import DeepSVDD, EmbeddingAutoencoder


MAD_SCALE = 1.482602218505602
DETECTOR_NAMES = (
    "pca_knn",
    "isolation_forest",
    "lof",
    "one_class_svm",
    "autoencoder",
    "deep_svdd",
)


@dataclass(frozen=True)
class DetectorConfig:
    components: int = 32
    neighbors: int = 20
    max_reference: int = 20_000
    one_class_max_reference: int = 10_000
    isolation_trees: int = 300
    one_class_nu: float = 0.05
    neural_hidden_dim: int = 128
    neural_latent_dim: int = 32
    neural_epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 20260825
    neural_seeds: tuple[int, ...] = (20260825, 20260826, 20260827)


@dataclass(frozen=True)
class RobustEmbeddingScale:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, embedding: np.ndarray) -> "RobustEmbeddingScale":
        values = _matrix(embedding)
        median = np.median(values, axis=0)
        mad = MAD_SCALE * np.median(np.abs(values - median), axis=0)
        return cls(
            median=median.astype(np.float32),
            scale=np.where(mad >= 1e-6, mad, 1.0).astype(np.float32),
        )

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        return ((_matrix(embedding) - self.median) / self.scale).astype(np.float32)


def score_detectors(
    calibration_embedding: np.ndarray,
    test_embedding: np.ndarray,
    *,
    config: DetectorConfig | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Fit every registered detector without labels and score test nodes."""

    config = DetectorConfig() if config is None else config
    calibration = _reference_subset(
        _matrix(calibration_embedding),
        config.max_reference,
        config.seed,
    )
    test = _matrix(test_embedding)
    if calibration.shape[1] != test.shape[1]:
        raise ValueError("calibration and test embedding dimensions differ")

    pca_knn = PCAWhitenedKNN.fit(
        calibration,
        PCAKNNConfig(
            components=config.components,
            neighbors=config.neighbors,
            max_reference=len(calibration),
            seed=config.seed,
        ),
    )
    scores: dict[str, np.ndarray] = {
        "pca_knn": pca_knn.score(test),
    }

    scale = RobustEmbeddingScale.fit(calibration)
    reference = scale.transform(calibration)
    query = scale.transform(test)

    isolation = IsolationForest(
        n_estimators=config.isolation_trees,
        contamination="auto",
        random_state=config.seed,
        n_jobs=-1,
    ).fit(reference)
    scores["isolation_forest"] = (-isolation.decision_function(query)).astype(
        np.float32
    )

    neighbor_count = min(config.neighbors, len(reference) - 1)
    lof = LocalOutlierFactor(
        n_neighbors=neighbor_count,
        novelty=True,
        contamination="auto",
        n_jobs=-1,
    ).fit(reference)
    scores["lof"] = (-lof.decision_function(query)).astype(np.float32)

    svm_reference = _reference_subset(
        reference,
        config.one_class_max_reference,
        config.seed + 1,
    )
    one_class = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=config.one_class_nu,
        cache_size=2048,
    ).fit(svm_reference)
    scores["one_class_svm"] = (-one_class.decision_function(query)).astype(
        np.float32
    )

    scores.update(
        _neural_scores(
            reference,
            query,
            config=config,
            device=device,
        )
    )
    return scores


def _neural_scores(
    calibration: np.ndarray,
    test: np.ndarray,
    *,
    config: DetectorConfig,
    device: str,
) -> dict[str, np.ndarray]:
    calibration_tensor = torch.from_numpy(calibration)
    autoencoder_scores = []
    svdd_scores = []
    for seed in config.neural_seeds:
        _seed(seed)
        autoencoder = EmbeddingAutoencoder(
            calibration.shape[1],
            latent_dim=config.neural_latent_dim,
            hidden_dim=config.neural_hidden_dim,
        ).to(device)
        _fit_network(
            autoencoder,
            _training_loader(calibration_tensor, config, seed),
            config,
            device,
        )
        autoencoder_scores.append(
            _score_network(autoencoder, test, config.batch_size, device)
        )

        _seed(seed)
        svdd = DeepSVDD(
            calibration.shape[1],
            latent_dim=config.neural_latent_dim,
            hidden_dim=config.neural_hidden_dim,
        ).to(device)
        center_loader = DataLoader(
            TensorDataset(calibration_tensor),
            batch_size=config.batch_size,
            shuffle=False,
        )
        svdd.initialize_center(batch[0] for batch in center_loader)
        _fit_network(
            svdd,
            _training_loader(calibration_tensor, config, seed),
            config,
            device,
        )
        svdd_scores.append(_score_network(svdd, test, config.batch_size, device))
    return {
        "autoencoder": np.mean(autoencoder_scores, axis=0).astype(np.float32),
        "deep_svdd": np.mean(svdd_scores, axis=0).astype(np.float32),
    }


def _training_loader(embedding, config: DetectorConfig, seed: int):
    return DataLoader(
        TensorDataset(embedding),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _fit_network(model, loader, config: DetectorConfig, device: str) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    for _ in range(config.neural_epochs):
        for (embedding,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(embedding.to(device))
            loss.backward()
            optimizer.step()


def _score_network(model, embedding, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embedding)),
        batch_size=batch_size,
        shuffle=False,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            values.append(model.score(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _reference_subset(embedding: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(embedding) <= limit:
        return embedding
    selected = np.random.default_rng(seed).choice(len(embedding), limit, replace=False)
    return embedding[np.sort(selected)]


def _matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("node embedding must be a finite [node,dimension] matrix")
    return matrix


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
