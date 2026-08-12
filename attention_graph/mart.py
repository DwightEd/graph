"""Mechanism-aware attention representation and non-GNN anomaly baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


MART_EMA_ALPHA = 0.2
MART_STATE_NAMES = (
    "retained_mass", "retained_prompt_fraction", "censored_row_entropy",
    "prompt_anchor", "censored_other_mass", "diagonal_mass",
)
MART_FEATURES = (
    "relative_position",
    *(
        f"{name}_{statistic}"
        for name in MART_STATE_NAMES
        for statistic in ("mean", "std")
    ),
    *(
        f"layer_drift_{name}_{statistic}"
        for name in MART_STATE_NAMES
        for statistic in ("mean", "std")
    ),
    "innovation_norm",
)


def _channel_state(sample):
    """Return response-token x channel states from canonical CSR rows."""
    response_tokens, channels = sample.num_response_tokens, sample.num_channels
    device = sample.response_values.device
    row_ptr = sample.response_row_ptr.long()
    lengths = row_ptr[1:] - row_ptr[:-1]
    rows = torch.repeat_interleave(
        torch.arange(channels * response_tokens, device=device), lengths
    )
    values = sample.response_values.float()
    columns = sample.response_column_indices.long()

    retained = torch.zeros(channels * response_tokens, device=device)
    prompt = torch.zeros_like(retained)
    retained.index_add_(0, rows, values)
    prompt_mask = columns < sample.response_idx
    prompt.index_add_(0, rows[prompt_mask], values[prompt_mask])
    safe_retained = retained.clamp_min(1e-12)
    q = prompt / safe_retained

    diagonal = sample.attention_diagonal.float().permute(2, 0, 1).reshape(
        sample.num_tokens, channels
    )[sample.response_idx:].transpose(0, 1).reshape(-1)
    other = (1.0 - diagonal - retained).clamp_min(0.0)
    total = (retained + diagonal + other).clamp_min(1e-12)
    probability = values / total[rows]
    entropy = torch.zeros_like(retained)
    entropy.index_add_(0, rows, -probability * probability.clamp_min(1e-12).log())
    diagonal_probability = diagonal / total
    other_probability = other / total
    entropy += -diagonal_probability * diagonal_probability.clamp_min(1e-12).log()
    entropy += -other_probability * other_probability.clamp_min(1e-12).log()
    support = lengths.float() + (diagonal > 0).float() + (other > 0).float()
    normalized_entropy = torch.where(
        support > 1.0, entropy / support.log(), torch.zeros_like(entropy)
    )
    states = torch.stack((
        retained, q, normalized_entropy, q * (1.0 - normalized_entropy), other, diagonal,
    ), dim=1).reshape(channels, response_tokens, 6).permute(1, 0, 2)
    return states


def mart_features(sample, *, ema_alpha: float = MART_EMA_ALPHA):
    """One causal mechanism vector per response token; labels are never read."""
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1]")
    states = _channel_state(sample)
    response_tokens = states.shape[0]
    summary = torch.stack((
        states.mean(1), states.std(1, correction=0),
    ), dim=2).reshape(response_tokens, -1)

    layered = states.reshape(
        response_tokens, sample.num_layers, sample.num_heads, len(MART_STATE_NAMES)
    )
    split = sample.num_layers // 2
    if split:
        drift_by_head = layered[:, split:].mean(1) - layered[:, :split].mean(1)
    else:
        drift_by_head = torch.zeros(
            (response_tokens, sample.num_heads, len(MART_STATE_NAMES)),
            device=states.device,
        )
    drift = torch.stack((
        drift_by_head.mean(1), drift_by_head.std(1, correction=0)
    ), dim=2).reshape(response_tokens, -1)

    innovation = torch.zeros(response_tokens, device=states.device)
    ema = summary[0]
    for token in range(1, response_tokens):
        innovation[token] = (summary[token] - ema).norm()
        ema = ema_alpha * summary[token] + (1.0 - ema_alpha) * ema
    position = torch.arange(response_tokens, device=states.device, dtype=torch.float32)
    position /= max(response_tokens - 1, 1)
    return torch.cat((position[:, None], summary, drift, innovation[:, None]), dim=1)


class MartDetector:
    """Train-only robust position calibration plus kNN novelty scoring."""

    def __init__(
        self, *, neighbors: int = 16, position_bins: int = 8,
        reference_size: int = 100_000, whiten: bool = True,
    ):
        if min(neighbors, position_bins, reference_size) < 1:
            raise ValueError("neighbors, position_bins and reference_size must be positive")
        self.neighbors = int(neighbors)
        self.position_bins = int(position_bins)
        self.reference_size = int(reference_size)
        self.whiten = bool(whiten)

    def _bins(self, matrix):
        return np.minimum((matrix[:, 0] * self.position_bins).astype(int), self.position_bins - 1)

    @staticmethod
    def _scale(values):
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        std = values.std(axis=0)
        return center, np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))

    def fit(self, matrices):
        values = np.concatenate(
            [np.asarray(matrix, dtype=np.float64) for matrix in matrices], axis=0
        )
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("training features must be a non-empty finite matrix list")
        self.input_dim = values.shape[1]
        self.feature_dim = self.input_dim - 1
        features = values[:, 1:]
        self.center, self.scale = self._scale(features)
        self.bin_center = np.tile(self.center, (self.position_bins, 1))
        self.bin_scale = np.tile(self.scale, (self.position_bins, 1))
        bins = self._bins(values)
        for bin_id in range(self.position_bins):
            selected = features[bins == bin_id]
            if len(selected) >= 3:
                self.bin_center[bin_id], self.bin_scale[bin_id] = self._scale(selected)
        normalized = self._transform(values)
        rank = max(1, np.linalg.matrix_rank(normalized - normalized.mean(axis=0)))
        components = min(rank, *normalized.shape)
        self.pca = PCA(
            n_components=components, whiten=self.whiten, random_state=0
        ).fit(normalized)
        reference_rows = np.linspace(
            0, len(normalized) - 1, min(self.reference_size, len(normalized)), dtype=int
        )
        self.reference = self.pca.transform(normalized[reference_rows]).astype(np.float32)
        self.knn = NearestNeighbors(
            n_neighbors=min(self.neighbors, len(self.reference))
        ).fit(self.reference)
        return self

    def _transform(self, values):
        bins = self._bins(values)
        return (values[:, 1:] - self.bin_center[bins]) / self.bin_scale[bins]

    def score(self, matrices):
        if not hasattr(self, "knn"):
            raise RuntimeError("fit must be called before score")
        values = np.concatenate(
            [np.asarray(matrix, dtype=np.float64) for matrix in matrices], axis=0
        )
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("feature matrix shape does not match fitted detector")
        embedding = self.pca.transform(self._transform(values)).astype(np.float32)
        distances = self.knn.kneighbors(embedding, return_distance=True)[0]
        return embedding, distances.mean(axis=1).astype(np.float32)


def save_mart(detector, path, *, provenance=None):
    provenance = {} if provenance is None else provenance
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        schema=np.asarray("mart-detector-v2"),
        feature_names=np.asarray(MART_FEATURES),
        ema_alpha=np.asarray(MART_EMA_ALPHA),
        neighbors=np.asarray(detector.neighbors),
        position_bins=np.asarray(detector.position_bins),
        reference_size=np.asarray(detector.reference_size),
        whiten=np.asarray(detector.whiten),
        input_dim=np.asarray(detector.input_dim),
        feature_dim=np.asarray(detector.feature_dim),
        center=detector.center, scale=detector.scale,
        bin_center=detector.bin_center, bin_scale=detector.bin_scale,
        pca_components=detector.pca.components_, pca_mean=detector.pca.mean_,
        pca_explained_variance=detector.pca.explained_variance_,
        pca_n_samples=np.asarray(detector.pca.n_samples_),
        reference=detector.reference,
        canonical_schema=np.asarray(str(provenance.get("schema", ""))),
        num_layers=np.asarray(int(provenance.get("num_layers", -1))),
        num_heads=np.asarray(int(provenance.get("num_heads", -1))),
        attention_floor=np.asarray(float(provenance.get("attention_floor", np.nan))),
        alignment=np.asarray(str(provenance.get("alignment", ""))),
        observer_model=np.asarray(str(provenance.get("observer_model", ""))),
        generator_model=np.asarray(str(provenance.get("generator_model", ""))),
        training_index_sha256=np.asarray(str(provenance.get("index_sha256", ""))),
    )
    return str(path)


def load_mart(path):
    with np.load(Path(path), allow_pickle=False) as values:
        if values["schema"].item() != "mart-detector-v2":
            raise ValueError("unsupported MART checkpoint")
        if tuple(values["feature_names"].tolist()) != MART_FEATURES:
            raise ValueError("MART checkpoint feature semantics differ from this code")
        if float(values["ema_alpha"]) != MART_EMA_ALPHA:
            raise ValueError("MART checkpoint EMA differs from this code")
        detector = MartDetector(
            neighbors=int(values["neighbors"]),
            position_bins=int(values["position_bins"]),
            reference_size=int(values["reference_size"]),
            whiten=bool(values["whiten"]),
        )
        detector.input_dim = int(values["input_dim"])
        detector.feature_dim = int(values["feature_dim"])
        detector.center = values["center"].copy()
        detector.scale = values["scale"].copy()
        detector.bin_center = values["bin_center"].copy()
        detector.bin_scale = values["bin_scale"].copy()
        components = values["pca_components"].copy()
        detector.pca = PCA(n_components=len(components), whiten=detector.whiten)
        detector.pca.components_ = components
        detector.pca.mean_ = values["pca_mean"].copy()
        detector.pca.explained_variance_ = values["pca_explained_variance"].copy()
        detector.pca.n_features_in_ = detector.feature_dim
        detector.pca.n_components_ = len(components)
        detector.pca.n_samples_ = int(values["pca_n_samples"])
        detector.reference = values["reference"].astype(np.float32, copy=True)
        detector.provenance = {
            "schema": values["canonical_schema"].item(),
            "num_layers": int(values["num_layers"]),
            "num_heads": int(values["num_heads"]),
            "attention_floor": float(values["attention_floor"]),
            "alignment": values["alignment"].item(),
            "observer_model": values["observer_model"].item(),
            "generator_model": values["generator_model"].item(),
            "index_sha256": values["training_index_sha256"].item(),
        }
    detector.knn = NearestNeighbors(
        n_neighbors=min(detector.neighbors, len(detector.reference))
    ).fit(detector.reference)
    return detector


def fit_mart(
    dataset, *, output_path, neighbors=16, position_bins=8, reference_size=100_000
):
    split = str(dataset.manifest.get("split", "")).casefold()
    if split != "train":
        raise ValueError("MART fitting requires a canonical train split")
    matrices = []
    for sample in tqdm(dataset, total=len(dataset), desc="fit MART features", unit="sample"):
        matrices.append(mart_features(sample.attention()).cpu().numpy())
        sample.release_attention()
    detector = MartDetector(
        neighbors=neighbors, position_bins=position_bins, reference_size=reference_size
    ).fit(matrices)
    return {
        "checkpoint": save_mart(detector, output_path, provenance=dataset.manifest),
        "samples": len(matrices),
        "tokens": int(sum(map(len, matrices))),
    }


def _check_dataset_geometry(dataset, detector):
    manifest = dataset.manifest
    expected = detector.provenance
    for field in (
        "schema", "num_layers", "num_heads", "alignment", "observer_model",
    ):
        if str(manifest.get(field, "")) != str(expected[field]):
            raise ValueError(f"MART checkpoint and canonical split differ in {field}")
    if not np.isclose(
        float(manifest["attention_floor"]), expected["attention_floor"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("MART checkpoint and canonical split differ in attention_floor")


def score_mart(dataset, *, checkpoint, output_path):
    detector = load_mart(checkpoint)
    _check_dataset_geometry(dataset, detector)
    records, embeddings, scores = [], [], []
    for sample in tqdm(dataset, total=len(dataset), desc="score MART", unit="sample"):
        matrix = mart_features(sample.attention()).cpu().numpy()
        embedding, score = detector.score([matrix])
        embeddings.append(embedding)
        scores.append(score)
        records.extend(
            (
                sample.sample_id, sample.source_id, index, sample.task_type,
                sample.data_source, sample.generator_model,
            )
            for index in range(len(matrix))
        )
        sample.release_attention()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,
        representation=np.asarray("mart_mechanism_pca_embedding"),
        embedding=np.concatenate(embeddings), score=np.concatenate(scores),
        sample_id=np.asarray([row[0] for row in records], dtype=str),
        source_id=np.asarray([row[1] for row in records], dtype=str),
        token_index=np.asarray([row[2] for row in records], dtype=np.int32),
        task_type=np.asarray([str(row[3]) for row in records], dtype=str),
        data_source=np.asarray([str(row[4]) for row in records], dtype=str),
        generator_model=np.asarray([str(row[5]) for row in records], dtype=str),
    )
    return {"output": str(output), "samples": len(dataset), "tokens": len(records), "labels_read": False}
