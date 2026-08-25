"""Token residuals and position-conditioned one-class detection."""

from dataclasses import dataclass

import numpy as np
import torch

from .config import HoloRouteConfig
from .graph import EventGraph
from .learning import autocast_context, reconstruction_error
from .model import DEPTH, EVENT, QUERY, RELAY, HoloRoute

RESIDUAL_NAMES = (
    "event",
    "depth",
    "relay",
    "query",
    "depth_relay",
    "holonomy",
)
CONDITION_NAMES = (
    "log_position",
    "relative_position",
    "relative_position_squared",
    "relative_position_cubed",
    "log_response_length",
    "log_event_count",
    "log_relay_count",
    "log_diamond_count",
    "retained_mass",
    "observed_head_fraction",
    "unresolved_mass",
)
MAD_SCALE = 1.482602218505602


@dataclass(frozen=True)
class TokenResiduals:
    value: np.ndarray
    coverage: np.ndarray


class ResidualReservoir:
    """Keep aligned rows from calibration graphs without storing every token."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.random = np.random.default_rng(seed)
        self.seen = 0
        self.size = 0
        self.blocks: dict[str, np.ndarray] | None = None

    def add(self, **blocks) -> None:
        arrays = {name: np.asarray(value) for name, value in blocks.items()}
        rows = len(next(iter(arrays.values())))
        if self.blocks is None:
            self.blocks = {
                name: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                for name, value in arrays.items()
            }
        for row in range(rows):
            self.seen += 1
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                index = int(self.random.integers(self.seen))
                if index >= self.capacity:
                    continue
            for name, value in arrays.items():
                self.blocks[name][index] = value[row]

    def values(self) -> dict[str, np.ndarray]:
        if self.blocks is None or self.size < 2:
            raise RuntimeError("calibration needs at least two token rows")
        return {name: value[: self.size].copy() for name, value in self.blocks.items()}


def column_median(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        if len(finite):
            result[column] = np.median(finite)
    return result


def fill_missing(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    row, column = np.where(~np.isfinite(output))
    if len(row):
        output[row, column] = fill[column]
    return output


def design_matrix(
    conditions: np.ndarray,
    task: np.ndarray,
    task_names: tuple[str, ...],
) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.float64)
    task = np.asarray(task).astype(str)
    task_indicator = np.column_stack([task == name for name in task_names]).astype(np.float64)
    return np.column_stack((np.ones(len(conditions)), conditions, task_indicator))


@dataclass(frozen=True)
class ConditionalReference:
    task_names: tuple[str, ...]
    fill: np.ndarray
    coefficient: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    energy_reference: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(
        cls,
        residuals: np.ndarray,
        conditions: np.ndarray,
        task: np.ndarray,
        config,
    ) -> "ConditionalReference":
        raw = np.asarray(residuals, dtype=np.float64)
        active = np.mean(np.isfinite(raw), axis=0) >= 0.05
        fill = column_median(raw)
        values = fill_missing(raw, fill)
        task_names = tuple(sorted(set(np.asarray(task).astype(str).tolist())))
        design = design_matrix(conditions, task, task_names)
        penalty = np.eye(design.shape[1]) * float(config.ridge_alpha)
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ values)

        centered = values - design @ coefficient
        median = np.median(centered, axis=0)
        mad = np.median(np.abs(centered - median), axis=0)
        scale = np.maximum(MAD_SCALE * mad, float(config.scale_floor))
        standardized = (centered - median) / scale
        standardized[:, ~active] = 0.0

        covariance = np.cov(standardized, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]])
        diagonal = np.diag(np.diag(covariance))
        shrinkage = float(config.covariance_shrinkage)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-4
        precision = np.linalg.pinv(covariance)

        positive = np.maximum(standardized, 0.0)
        energy = np.einsum("nf,fg,ng->n", positive, precision, positive)
        return cls(
            task_names=task_names,
            fill=fill.astype(np.float32),
            coefficient=coefficient.astype(np.float32),
            median=median.astype(np.float32),
            scale=scale.astype(np.float32),
            precision=precision.astype(np.float32),
            energy_reference=np.sort(energy.astype(np.float32)),
            active=active.astype(bool),
        )

    def transform(
        self,
        residuals: np.ndarray,
        conditions: np.ndarray,
        task: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = fill_missing(residuals, self.fill)
        design = design_matrix(conditions, task, self.task_names)
        standardized = (
            (values - design @ self.coefficient - self.median) / self.scale
        ).astype(np.float32)
        standardized[:, ~self.active] = 0.0
        positive = np.maximum(standardized, 0.0)
        energy = np.einsum("nf,fg,ng->n", positive, self.precision, positive)
        rank = np.searchsorted(self.energy_reference, energy, side="left")
        probability = (
            len(self.energy_reference) - rank + 1
        ) / float(len(self.energy_reference) + 1)
        score = -np.log10(np.clip(probability, 1e-12, 1.0)).astype(np.float32)
        return score, standardized

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "reference_task_names": np.asarray(self.task_names),
            "reference_fill": self.fill,
            "reference_coefficient": self.coefficient,
            "reference_median": self.median,
            "reference_scale": self.scale,
            "reference_precision": self.precision,
            "reference_energy": self.energy_reference,
            "reference_active": self.active,
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "ConditionalReference":
        return cls(
            task_names=tuple(np.asarray(arrays["reference_task_names"]).astype(str).tolist()),
            fill=np.asarray(arrays["reference_fill"]),
            coefficient=np.asarray(arrays["reference_coefficient"]),
            median=np.asarray(arrays["reference_median"]),
            scale=np.asarray(arrays["reference_scale"]),
            precision=np.asarray(arrays["reference_precision"]),
            energy_reference=np.asarray(arrays["reference_energy"]),
            active=np.asarray(arrays["reference_active"]).astype(bool),
        )


def token_conditions(graph: EventGraph) -> np.ndarray:
    tokens = graph.response_count
    position = np.arange(tokens, dtype=np.float32)
    relative = position / max(tokens - 1, 1)
    event_token = graph.event_query.detach().cpu().numpy()
    event_count = np.bincount(event_token, minlength=tokens).astype(np.float32)

    relay_count = np.zeros(tokens, dtype=np.float32)
    if graph.relay_edges.shape[1]:
        relay_token = graph.event_query[graph.relay_edges[1]].detach().cpu().numpy()
        relay_count = np.bincount(relay_token, minlength=tokens).astype(np.float32)

    diamond_count = np.zeros(tokens, dtype=np.float32)
    if graph.diamonds.shape[1]:
        diamond_token = graph.event_query[graph.diamonds[3]].detach().cpu().numpy()
        diamond_count = np.bincount(diamond_token, minlength=tokens).astype(np.float32)

    mass = graph.events.mass.detach().cpu().numpy().astype(np.float32)
    observed = (
        graph.events.observed.float().mean(dim=-1).detach().cpu().numpy().astype(np.float32)
    )
    denominator = np.maximum(event_count, 1.0)
    retained = (
        np.bincount(event_token, weights=mass, minlength=tokens).astype(np.float32)
        / denominator
    )
    observed_fraction = (
        np.bincount(event_token, weights=observed, minlength=tokens).astype(np.float32)
        / denominator
    )
    unresolved = graph.unresolved.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)

    return np.column_stack(
        (
            np.log1p(position),
            relative,
            relative**2,
            relative**3,
            np.full(tokens, np.log1p(tokens), dtype=np.float32),
            np.log1p(event_count),
            np.log1p(relay_count),
            np.log1p(diamond_count),
            retained,
            observed_fraction,
            unresolved,
        )
    ).astype(np.float32)


def aggregate_events(
    graph: EventGraph,
    values: torch.Tensor,
    available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = values.new_zeros(graph.response_count)
    count = values.new_zeros(graph.response_count)
    selected = available & torch.isfinite(values)
    if bool(selected.any()):
        token = graph.event_query[selected]
        total.index_add_(0, token, values[selected])
        count.index_add_(0, token, torch.ones_like(values[selected]))
    output = values.new_full((graph.response_count,), torch.nan)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output, count


def score_schedule(graph: EventGraph, folds: int, seed: int) -> list[torch.Tensor]:
    event_count = graph.event_count
    if not event_count:
        return []
    generator = torch.Generator(device=graph.device).manual_seed(int(seed))
    order = torch.randperm(event_count, generator=generator, device=graph.device)
    fold_count = max(1, min(int(folds), event_count))
    schedule = []
    for selected in torch.tensor_split(order, fold_count):
        if not len(selected):
            continue
        mask = torch.zeros(event_count, dtype=torch.bool, device=graph.device)
        mask[selected] = True
        schedule.append(mask)
    return schedule


@torch.no_grad()
def score_graph(
    model: HoloRoute,
    graph: EventGraph,
    config: HoloRouteConfig,
    seed: int,
) -> TokenResiduals:
    model.eval()
    event_sum = graph.events.value.new_zeros((graph.event_count, 5))
    event_count = graph.events.value.new_zeros((graph.event_count, 5))

    for mask in score_schedule(graph, config.detection.score_folds, seed):
        values = graph.events.value.clone()
        observed = graph.events.observed.clone()
        values[mask] = 0.0
        observed[mask] = False
        with autocast_context(model, config.train.mixed_precision):
            output = model(
                graph,
                values=values,
                observed=observed,
                query_keep=~mask,
            )

        errors = tuple(
            reconstruction_error(
                output.predictions.value[:, index],
                output.predictions.support[:, index],
                graph,
                config,
            ).float()
            for index in (EVENT, DEPTH, RELAY, QUERY)
        )
        disagreement = (
            output.contexts[:, 0] - output.contexts[:, 1]
        ).square().mean(dim=-1).float()
        masks = (
            mask,
            mask & output.coverage[:, 0],
            mask & output.coverage[:, 1],
            mask & output.coverage[:, 2],
            mask & output.coverage[:, 0] & output.coverage[:, 1],
        )
        for column, (error, available) in enumerate(
            zip((*errors, disagreement), masks, strict=True)
        ):
            event_sum[available, column] += error[available]
            event_count[available, column] += 1.0

    event_value = event_sum / event_count.clamp_min(1.0)
    token_values: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    for column in range(5):
        value, coverage = aggregate_events(
            graph,
            event_value[:, column],
            event_count[:, column] > 0,
        )
        token_values.append(value)
        token_counts.append(coverage)

    with autocast_context(model, config.train.mixed_precision):
        clean = model(graph)
    holonomy = graph.events.value.new_full((graph.response_count,), torch.nan)
    holonomy_count = graph.events.value.new_zeros(graph.response_count)
    if clean.holonomy.numel():
        clean_holonomy = clean.holonomy.float()
        total = graph.events.value.new_zeros(graph.response_count)
        total.index_add_(0, clean.holonomy_token, clean_holonomy)
        holonomy_count.index_add_(
            0,
            clean.holonomy_token,
            torch.ones_like(clean_holonomy),
        )
        available = holonomy_count > 0
        holonomy[available] = total[available] / holonomy_count[available]
    token_values.append(holonomy)
    token_counts.append(holonomy_count)

    return TokenResiduals(
        value=torch.stack(token_values, dim=-1).cpu().numpy().astype(np.float32),
        coverage=torch.stack(token_counts, dim=-1).cpu().numpy().astype(np.float32),
    )
