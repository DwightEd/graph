"""Channel-resolved variable-order De Bruijn route grammar."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GrammarConfig


def _normalize(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError("route sequence must have shape [token, channel, state]")
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-30)


@dataclass(frozen=True)
class RouteGrammar:
    prior: torch.Tensor
    order1: torch.Tensor
    order2: torch.Tensor
    order2_context_count: torch.Tensor
    backoff_tau: float
    token_count: int

    @property
    def num_channels(self) -> int:
        return int(self.prior.shape[0])

    @property
    def num_states(self) -> int:
        return int(self.prior.shape[1])

    def to(self, device: str | torch.device) -> "RouteGrammar":
        return RouteGrammar(
            prior=self.prior.to(device),
            order1=self.order1.to(device),
            order2=self.order2.to(device),
            order2_context_count=self.order2_context_count.to(device),
            backoff_tau=self.backoff_tau,
            token_count=self.token_count,
        )

    def predict_order1(self, q: torch.Tensor) -> torch.Tensor:
        q = _normalize(q).to(self.prior)
        predicted = self.prior[None].expand(len(q), -1, -1).clone()
        if len(q) > 1:
            predicted[1:] = torch.einsum("tca,cad->tcd", q[:-1], self.order1)
        return _normalize(predicted)

    def predict(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return backoff prediction and its order-2 interpolation weight."""

        q = _normalize(q).to(self.prior)
        order1 = self.predict_order1(q)
        predicted = order1.clone()
        weight = q.new_zeros((len(q), self.num_channels))
        if len(q) > 2:
            order2 = torch.einsum(
                "tca,tcb,cabd->tcd",
                q[:-2],
                q[1:-1],
                self.order2,
            )
            support = torch.einsum(
                "tca,tcb,cab->tc",
                q[:-2],
                q[1:-1],
                self.order2_context_count,
            )
            current_weight = support / (support + float(self.backoff_tau))
            predicted[2:] = (
                current_weight[..., None] * order2
                + (1.0 - current_weight[..., None]) * order1[2:]
            )
            weight[2:] = current_weight
        return _normalize(predicted), weight

    @staticmethod
    def cross_entropy(q: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        q = _normalize(q).to(predicted)
        return -(q * predicted.clamp_min(1e-30).log()).sum(dim=-1)

    def score(
        self,
        q: torch.Tensor,
        *,
        use_order2: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        order1_prediction = self.predict_order1(q)
        backoff_prediction, order2_weight = self.predict(q)
        order1_surprisal = self.cross_entropy(q, order1_prediction)
        if use_order2:
            prediction = backoff_prediction
            surprisal = self.cross_entropy(q, prediction)
        else:
            prediction = order1_prediction
            surprisal = order1_surprisal
            order2_weight = torch.zeros_like(order2_weight)
        return surprisal, order1_surprisal, prediction, order2_weight


class GrammarAccumulator:
    """Streaming sufficient statistics for complete response sequences."""

    def __init__(
        self,
        num_channels: int,
        num_states: int,
        *,
        config: GrammarConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = GrammarConfig() if config is None else config
        self.prior_count = torch.zeros(
            (num_channels, num_states), dtype=torch.float64, device=device
        )
        self.order1_count = torch.zeros(
            (num_channels, num_states, num_states),
            dtype=torch.float64,
            device=device,
        )
        self.order2_count = torch.zeros(
            (num_channels, num_states, num_states, num_states),
            dtype=torch.float64,
            device=device,
        )
        self.token_count = 0

    @torch.no_grad()
    def update(self, q: torch.Tensor) -> None:
        q = _normalize(q).to(self.prior_count)
        self.prior_count += q.sum(dim=0)
        if len(q) > 1:
            self.order1_count += torch.einsum(
                "tca,tcd->cad", q[:-1], q[1:]
            )
        if len(q) > 2:
            self.order2_count += torch.einsum(
                "tca,tcb,tcd->cabd",
                q[:-2],
                q[1:-1],
                q[2:],
            )
        self.token_count += len(q)

    @torch.no_grad()
    def freeze(self) -> RouteGrammar:
        if self.token_count == 0:
            raise ValueError("grammar fitting received no response tokens")
        alpha = float(self.config.alpha)
        prior = self.prior_count + alpha
        order1 = self.order1_count + alpha
        order2 = self.order2_count + alpha
        prior /= prior.sum(dim=-1, keepdim=True)
        order1 /= order1.sum(dim=-1, keepdim=True)
        order2 /= order2.sum(dim=-1, keepdim=True)
        return RouteGrammar(
            prior=prior.float(),
            order1=order1.float(),
            order2=order2.float(),
            order2_context_count=self.order2_count.sum(dim=-1).float(),
            backoff_tau=float(self.config.backoff_tau),
            token_count=self.token_count,
        )
