"""A small fixed benchmark of node-only unsupervised detectors."""

from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from .models import DeepSVDD, EmbeddingAutoencoder


@dataclass(frozen=True)
class DetectorConfig:
    components: int = 32
    neighbors: int = 20
    max_reference: int = 20_000
    trees: int = 300
    hidden_dim: int = 128
    latent_dim: int = 32
    epochs: int = 20
    batch_size: int = 1024
    learning_rate: float = 1e-3
    seeds: tuple[int, ...] = (20260825, 20260826, 20260827)


@dataclass(frozen=True)
class RobustScale:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "RobustScale":
        center = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - center), axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.center) / self.scale).astype(np.float32)


def reference_subset(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    index = np.random.default_rng(seed).choice(len(values), limit, replace=False)
    return values[np.sort(index)]


def has_variation(values: np.ndarray) -> bool:
    return bool(np.max(np.ptp(values, axis=0)) > 1e-6)


def score_pca_knn(
    calibration: np.ndarray,
    test: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    if not has_variation(calibration):
        return np.zeros(len(test), dtype=np.float32)

    scale = RobustScale.fit(calibration)
    reference = scale.transform(calibration)
    query = scale.transform(test)
    components = min(config.components, len(reference) - 1, reference.shape[1])
    pca = PCA(n_components=components, whiten=True, random_state=config.seeds[0])
    reference = pca.fit_transform(reference)
    query = pca.transform(query)
    neighbors = NearestNeighbors(n_neighbors=min(config.neighbors, len(reference)))
    neighbors.fit(reference)
    distance, _ = neighbors.kneighbors(query)
    return distance.mean(axis=1).astype(np.float32)


def score_detectors(
    calibration_embedding: np.ndarray,
    test_embedding: np.ndarray,
    config: DetectorConfig | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    config = DetectorConfig() if config is None else config
    calibration = reference_subset(
        np.asarray(calibration_embedding, dtype=np.float32),
        config.max_reference,
        config.seeds[0],
    )
    test = np.asarray(test_embedding, dtype=np.float32)

    if not has_variation(calibration):
        zero = np.zeros(len(test), dtype=np.float32)
        return {
            "pca_knn": zero.copy(),
            "isolation_forest": zero.copy(),
            "lof": zero.copy(),
            "autoencoder": zero.copy(),
            "deep_svdd": zero.copy(),
        }

    scores = {"pca_knn": score_pca_knn(calibration, test, config)}
    scale = RobustScale.fit(calibration)
    reference = scale.transform(calibration)
    query = scale.transform(test)

    isolation = IsolationForest(
        n_estimators=config.trees,
        random_state=config.seeds[0],
        n_jobs=-1,
    ).fit(reference)
    scores["isolation_forest"] = (-isolation.decision_function(query)).astype(np.float32)

    lof = LocalOutlierFactor(
        n_neighbors=min(config.neighbors, len(reference) - 1),
        novelty=True,
        n_jobs=-1,
    ).fit(reference)
    scores["lof"] = (-lof.decision_function(query)).astype(np.float32)

    scores.update(score_neural_detectors(reference, query, config, device))
    return scores


def score_neural_detectors(
    calibration: np.ndarray,
    test: np.ndarray,
    config: DetectorConfig,
    device: str,
) -> dict[str, np.ndarray]:
    autoencoder_scores = []
    svdd_scores = []
    calibration_tensor = torch.from_numpy(calibration)

    for seed in config.seeds:
        set_seed(seed)
        autoencoder = EmbeddingAutoencoder(
            calibration.shape[1],
            config.hidden_dim,
            config.latent_dim,
        ).to(device)
        train_network(autoencoder, calibration_tensor, config, device, seed)
        autoencoder_scores.append(score_network(autoencoder, test, config, device))

        set_seed(seed)
        svdd = DeepSVDD(
            calibration.shape[1],
            config.hidden_dim,
            config.latent_dim,
        ).to(device)
        svdd.set_center(calibration_tensor.to(device))
        train_network(svdd, calibration_tensor, config, device, seed)
        svdd_scores.append(score_network(svdd, test, config, device))

    return {
        "autoencoder": np.mean(autoencoder_scores, axis=0).astype(np.float32),
        "deep_svdd": np.mean(svdd_scores, axis=0).astype(np.float32),
    }


def train_network(model, embedding, config: DetectorConfig, device: str, seed: int) -> None:
    loader = DataLoader(
        TensorDataset(embedding),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.epochs):
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch.to(device))
            loss.backward()
            optimizer.step()


def score_network(model, embedding, config: DetectorConfig, device: str) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embedding)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    score = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            score.append(model.score(batch.to(device)).cpu().numpy())
    return np.concatenate(score).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
