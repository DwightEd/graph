import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class _ConditionalStudentDensity(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_components: int,
        degrees_of_freedom: float,
        variance_floor: float,
        contamination_scale: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_components = num_components
        self.degrees_of_freedom = degrees_of_freedom
        self.variance_floor = variance_floor
        self.contamination_scale = contamination_scale
        self.context = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.mixture_logits = nn.Linear(hidden_dim, num_components)
        self.locations = nn.Linear(hidden_dim, num_components * embedding_dim)
        self.raw_scales = nn.Linear(hidden_dim, num_components * embedding_dim)

    def log_probabilities(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.zeros_like(embeddings)
        shifted[:, 1:] = embeddings[:, :-1]
        context, _ = self.context(shifted)

        batch, length, _ = embeddings.shape
        locations = self.locations(context).reshape(
            batch, length, self.num_components, self.embedding_dim
        )
        variances = F.softplus(self.raw_scales(context)).reshape_as(locations)
        variances = variances + self.variance_floor
        scales = variances.sqrt()
        component_log_prob = _diagonal_student_log_prob(
            embeddings.unsqueeze(2),
            locations,
            scales,
            self.degrees_of_freedom,
        )
        log_weights = F.log_softmax(self.mixture_logits(context), dim=-1)
        inlier_log_prob = torch.logsumexp(log_weights + component_log_prob, dim=-1)

        contamination_log_prob = _diagonal_student_log_prob(
            embeddings,
            torch.zeros_like(embeddings),
            torch.full_like(embeddings, self.contamination_scale),
            degrees_of_freedom=1.0,
        )
        return inlier_log_prob, contamination_log_prob


def _diagonal_student_log_prob(
    value: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: float,
) -> torch.Tensor:
    log_normalizer = (
        math.lgamma((degrees_of_freedom + 1.0) / 2.0)
        - math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom * math.pi)
        - scale.log()
    )
    standardized = (value - location) / scale
    log_kernel = -0.5 * (degrees_of_freedom + 1.0) * torch.log1p(
        standardized.square() / degrees_of_freedom
    )
    return (log_normalizer + log_kernel).sum(dim=-1)


class ConditionalStudentMixture:
    """Causal Student-t density estimator for variable-length token embeddings."""

    def __init__(
        self,
        num_components: int = 4,
        contamination: float = 0.05,
        variance_floor: float = 1e-4,
        hidden_dim: int = 32,
        degrees_of_freedom: float = 5.0,
        contamination_scale: float = 10.0,
        fit_steps: int = 75,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if num_components < 1:
            raise ValueError("num_components must be positive")
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be between zero and one")
        if variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        if hidden_dim < 1 or fit_steps < 1:
            raise ValueError("hidden_dim and fit_steps must be positive")
        if degrees_of_freedom <= 0.0 or contamination_scale <= 0.0:
            raise ValueError("Student-t parameters must be positive")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        self.num_components = num_components
        self.contamination = contamination
        self.variance_floor = variance_floor
        self.hidden_dim = hidden_dim
        self.degrees_of_freedom = degrees_of_freedom
        self.contamination_scale = contamination_scale
        self.fit_steps = fit_steps
        self.learning_rate = learning_rate
        self.seed = seed
        self._density: _ConditionalStudentDensity | None = None
        self._center: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None

    def fit(self, samples: Sequence[torch.Tensor], *, progress: bool = False):
        embeddings = _validated_samples(samples)
        all_embeddings = torch.cat(embeddings)
        self._center = all_embeddings.median(dim=0).values
        absolute_deviation = (all_embeddings - self._center).abs()
        robust_scale = 1.4826 * absolute_deviation.median(dim=0).values
        fallback_scale = absolute_deviation.square().mean(dim=0).sqrt()
        self._scale = torch.where(robust_scale > 1e-6, robust_scale, fallback_scale)
        self._scale = self._scale.clamp_min(1e-6)

        normalized = [(sample - self._center) / self._scale for sample in embeddings]
        padded, valid = _padded_batch(normalized)
        rng_devices = [padded.device] if padded.is_cuda else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(self.seed)
            if padded.is_cuda:
                torch.cuda.manual_seed(self.seed)
            self._density = _ConditionalStudentDensity(
                embedding_dim=padded.shape[-1],
                hidden_dim=self.hidden_dim,
                num_components=self.num_components,
                degrees_of_freedom=self.degrees_of_freedom,
                variance_floor=self.variance_floor,
                contamination_scale=self.contamination_scale,
            ).to(padded.device)

        optimizer = torch.optim.Adam(self._density.parameters(), lr=self.learning_rate)
        log_inlier_weight = math.log1p(-self.contamination)
        log_contamination_weight = math.log(self.contamination)
        self._density.train()
        steps = tqdm(
            range(self.fit_steps),
            desc="causal density",
            unit="step",
            leave=False,
            disable=not progress,
        )
        for _ in steps:
            optimizer.zero_grad()
            inlier, contamination = self._density.log_probabilities(padded)
            observed = torch.logaddexp(
                inlier + log_inlier_weight,
                contamination + log_contamination_weight,
            )
            loss = -observed[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite anomaly-model loss")
            loss.backward()
            nn.utils.clip_grad_norm_(self._density.parameters(), max_norm=10.0)
            optimizer.step()
            steps.set_postfix(nll=f"{float(loss.detach()):.4f}")

        self._density.eval()
        return self

    def score(self, samples: Sequence[torch.Tensor]):
        if self._density is None or self._center is None or self._scale is None:
            raise RuntimeError("fit must be called before score")
        embeddings = _validated_samples(samples, expected_dim=self._center.numel())
        normalized = [(sample - self._center) / self._scale for sample in embeddings]
        padded, _ = _padded_batch(normalized)

        with torch.no_grad():
            inlier, _ = self._density.log_probabilities(padded)
        return [-inlier[index, : sample.shape[0]] for index, sample in enumerate(embeddings)]


class EmpiricalTailCalibrator:
    """Smoothed empirical upper-tail probabilities from a fixed fit sample."""

    def __init__(self) -> None:
        self._fit_scores: torch.Tensor | None = None

    def fit(self, scores: torch.Tensor):
        if scores.ndim != 1 or scores.numel() == 0:
            raise ValueError("fit scores must be a non-empty one-dimensional tensor")
        if not torch.isfinite(scores).all():
            raise ValueError("fit scores must be finite")
        self._fit_scores = scores.detach().to(dtype=torch.float32, device="cpu").sort().values
        return self

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        if self._fit_scores is None:
            raise RuntimeError("fit must be called before transform")
        output_device = scores.device
        query = scores.detach().to(dtype=torch.float32, device="cpu")
        insertion = torch.searchsorted(self._fit_scores, query.reshape(-1), right=False)
        tail_count = self._fit_scores.numel() - insertion
        probabilities = (tail_count + 1).float() / (self._fit_scores.numel() + 1)
        return probabilities.reshape(scores.shape).to(output_device)


def _validated_samples(
    samples: Sequence[torch.Tensor], expected_dim: int | None = None
) -> list[torch.Tensor]:
    if not samples:
        raise ValueError("samples must not be empty")
    embeddings = [torch.as_tensor(sample, dtype=torch.float32) for sample in samples]
    embedding_dim = embeddings[0].shape[1] if embeddings[0].ndim == 2 else None
    if expected_dim is not None:
        embedding_dim = expected_dim
    if embedding_dim is None or embedding_dim < 1:
        raise ValueError("each sample must have shape [tokens, embedding_dim]")
    for sample in embeddings:
        if sample.ndim != 2 or sample.shape[0] < 1 or sample.shape[1] != embedding_dim:
            raise ValueError("samples must be non-empty matrices with one embedding dimension")
        if not torch.isfinite(sample).all():
            raise ValueError("sample embeddings must be finite")
    return embeddings


def _padded_batch(samples: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence(samples, batch_first=True)
    lengths = torch.tensor(
        [sample.shape[0] for sample in samples], device=padded.device
    )
    positions = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0)
    return padded, positions < lengths.unsqueeze(1)
