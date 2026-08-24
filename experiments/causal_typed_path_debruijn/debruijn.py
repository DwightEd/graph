"""Streaming, channel-independent soft De Bruijn transition counts.

There is no neural network, optimizer, label, or gradient path in this module.
Each channel owns a separate order-1 or order-2 Markov table. Soft route states
are top-k truncated and renormalized before fractional transition counting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import DeBruijnConfig


def _prepare_q(
    q: torch.Tensor,
    *,
    channels: int,
    states: int,
    top_k: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if q.ndim != 3 or q.shape[1:] != (channels, states):
        raise ValueError(f"q must have shape [R,{channels},{states}]")
    if q.shape[0] < 1:
        raise ValueError("q must contain at least one response token")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("q must be finite and non-negative")
    if top_k > states:
        raise ValueError("soft_top_k cannot exceed the route-state count")
    q = q.detach()
    if device is not None or dtype is not None:
        q = q.to(
            device=q.device if device is None else device,
            dtype=q.dtype if dtype is None else dtype,
        )
    values, indices = torch.topk(q, k=top_k, dim=-1, largest=True, sorted=False)
    truncated = torch.zeros_like(q).scatter(-1, indices, values)
    total = truncated.sum(dim=-1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("every q row/channel must have positive route mass")
    return truncated / total


@dataclass(frozen=True)
class FrozenDeBruijn:
    """Smoothed prior and per-channel transition probabilities."""

    config: DeBruijnConfig
    prior: torch.Tensor
    transition: torch.Tensor
    token_count: int
    transition_window_count: int

    @property
    def num_channels(self) -> int:
        return int(self.prior.shape[0])

    @property
    def num_states(self) -> int:
        return int(self.prior.shape[1])

    def validate(self) -> "FrozenDeBruijn":
        self.config.validate()
        if self.prior.ndim != 2:
            raise ValueError("prior must be [C,M]")
        contexts = self.num_states ** int(self.config.order)
        if self.transition.shape != (self.num_channels, contexts, self.num_states):
            raise ValueError("transition has the wrong De Bruijn geometry")
        if self.transition.device != self.prior.device:
            raise ValueError("prior and transition must share one device")
        for tensor in (self.prior, self.transition):
            if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
                raise ValueError("smoothed probabilities must be finite and positive")
        if not torch.allclose(
            self.prior.sum(dim=-1),
            torch.ones(self.num_channels, dtype=self.prior.dtype, device=self.prior.device),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("prior rows must sum to one")
        if not torch.allclose(
            self.transition.sum(dim=-1),
            torch.ones(
                self.transition.shape[:-1],
                dtype=self.transition.dtype,
                device=self.transition.device,
            ),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("transition rows must sum to one")
        if self.token_count < 1 or self.transition_window_count < 0:
            raise ValueError("frozen count audit is invalid")
        return self

    def to(self, device: str | torch.device) -> "FrozenDeBruijn":
        return FrozenDeBruijn(
            config=self.config,
            prior=self.prior.to(device),
            transition=self.transition.to(device),
            token_count=self.token_count,
            transition_window_count=self.transition_window_count,
        )

    @torch.no_grad()
    def predict_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Return the prefix-only predicted next-state distribution ``[R,C,M]``.

        Tokens without a complete order-``k`` context use the frozen marginal
        prior.  A sample is always passed independently, so no n-gram can cross
        a response boundary.
        """

        q = _prepare_q(
            q,
            channels=self.num_channels,
            states=self.num_states,
            top_k=self.config.soft_top_k,
            dtype=self.prior.dtype,
            device=self.prior.device,
        )
        predicted = self.prior.unsqueeze(0).expand(q.shape[0], -1, -1).clone()
        order = int(self.config.order)
        if order == 1 and q.shape[0] > 1:
            transition = self.transition.reshape(
                self.num_channels, self.num_states, self.num_states
            )
            predicted[1:] = torch.einsum(
                "tca,cad->tcd", q[:-1], transition
            )
        elif order == 2 and q.shape[0] > 2:
            transition = self.transition.reshape(
                self.num_channels,
                self.num_states,
                self.num_states,
                self.num_states,
            )
            predicted[2:] = torch.einsum(
                "tca,tcb,cabd->tcd",
                q[:-2],
                q[1:-1],
                transition,
            )
        predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return predicted

    @torch.no_grad()
    def score(self, q: torch.Tensor) -> torch.Tensor:
        """Return soft-state predictive cross-entropy in ``[R,C]`` form."""

        observed = _prepare_q(
            q,
            channels=self.num_channels,
            states=self.num_states,
            top_k=self.config.soft_top_k,
            dtype=self.prior.dtype,
            device=self.prior.device,
        )
        predicted = self.predict_distribution(q)
        tiny = max(torch.finfo(predicted.dtype).tiny, 1e-30)
        return -(observed * torch.log(predicted.clamp_min(tiny))).sum(dim=-1)


class DeBruijnAccumulator:
    """Mutable streaming sufficient statistics, frozen before scoring."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_states: int,
        config: DeBruijnConfig | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = DeBruijnConfig() if config is None else config
        self.config.validate()
        if isinstance(num_channels, bool) or int(num_channels) < 1:
            raise ValueError("num_channels must be positive")
        if isinstance(num_states, bool) or int(num_states) < 2:
            raise ValueError("num_states must be at least two")
        if self.config.soft_top_k > int(num_states):
            raise ValueError("soft_top_k cannot exceed num_states")
        if not dtype.is_floating_point:
            raise ValueError("count dtype must be floating point")
        self.num_channels = int(num_channels)
        self.num_states = int(num_states)
        contexts = self.num_states ** int(self.config.order)
        self.prior_count = torch.zeros(
            (self.num_channels, self.num_states), dtype=dtype, device=device
        )
        self.transition_count = torch.zeros(
            (self.num_channels, contexts, self.num_states), dtype=dtype, device=device
        )
        self.token_count = 0
        self.transition_window_count = 0
        self._frozen = False

    @property
    def device(self) -> torch.device:
        return self.prior_count.device

    @torch.no_grad()
    def update(self, q: torch.Tensor) -> None:
        """Accumulate one complete sample; calls never create cross-sample edges."""

        if self._frozen:
            raise RuntimeError("cannot update a frozen De Bruijn accumulator")
        q = _prepare_q(
            q,
            channels=self.num_channels,
            states=self.num_states,
            top_k=self.config.soft_top_k,
            dtype=self.prior_count.dtype,
            device=self.device,
        )
        length = int(q.shape[0])
        self.prior_count += q.sum(dim=0)
        order = int(self.config.order)
        windows = max(length - order, 0)
        if order == 1 and windows:
            count = torch.einsum("tca,tcd->cad", q[:-1], q[1:])
            self.transition_count += count
        elif order == 2 and windows:
            count = torch.einsum(
                "tca,tcb,tcd->cabd", q[:-2], q[1:-1], q[2:]
            )
            self.transition_count += count.reshape(
                self.num_channels, self.num_states**2, self.num_states
            )
        self.token_count += length
        self.transition_window_count += windows

    @torch.no_grad()
    def freeze(self) -> FrozenDeBruijn:
        """Apply symmetric Dirichlet smoothing and prevent further updates."""

        if self.token_count < 1:
            raise RuntimeError("at least one update is required before freeze")
        self._frozen = True
        alpha = float(self.config.alpha)
        prior = self.prior_count + alpha
        prior = prior / prior.sum(dim=-1, keepdim=True)
        transition = self.transition_count + alpha
        transition = transition / transition.sum(dim=-1, keepdim=True)
        return FrozenDeBruijn(
            config=self.config,
            prior=prior,
            transition=transition,
            token_count=self.token_count,
            transition_window_count=self.transition_window_count,
        ).validate()
