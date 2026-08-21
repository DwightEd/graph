"""Runtime helpers for grounding-sensitive graph interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .counterfactual import CounterfactualScores, build_scores, reconstruction_loss
from .data import SourceReuseGraph
from .edge_refinement import PairEncoding
from .graph_encoder import RelationalGraphEncoder, SourceStateBank, SourceStateEncoder, StructuredDecoder
from .grounding_config import GroundingGraphConfig
from .grounding_nulls import matched_endpoint_rewire
from .provenance import TokenGroundingTargets


@dataclass(frozen=True)
class TokenViews:
    raw: torch.Tensor
    full: torch.Tensor
    no_prompt: torch.Tensor
    no_response: torch.Tensor
    perturbed: torch.Tensor
    no_state: torch.Tensor
    shuffled_state: torch.Tensor
    endpoint_rewired: torch.Tensor
    changed_fraction: torch.Tensor


def observed_weights(
    config: GroundingGraphConfig,
    base_weight: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    keep = torch.ones_like(base_weight, dtype=torch.bool)
    if base_weight.numel() > 1 and config.edge_mask_rate > 0:
        keep = torch.rand(
            base_weight.shape,
            generator=generator,
            device=base_weight.device,
        ) >= config.edge_mask_rate
        if not bool(keep.any()):
            keep[base_weight.argmax()] = True
    return (base_weight * keep).detach().clone().requires_grad_(True)


def encode(
    encoder: RelationalGraphEncoder,
    graph: SourceReuseGraph,
    *,
    token: int,
    pair: PairEncoding,
    source_state: torch.Tensor,
    view: str = "full",
    multiplier: torch.Tensor | None = None,
) -> torch.Tensor:
    return encoder(
        graph,
        token=token,
        pair=pair,
        source_state=source_state,
        view=view,
        pair_multiplier=multiplier,
    )


def perturbation(
    config: GroundingGraphConfig,
    pair: PairEncoding,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if pair.mass.numel() == 0:
        return pair.mass
    noise = torch.randn(
        pair.mass.shape,
        generator=generator,
        device=pair.mass.device,
    ) * config.perturbation_scale
    multiplier = noise.exp()
    reference = pair.mass * pair.gate
    scale = (reference * multiplier).sum() / reference.sum().clamp_min(1e-8)
    return multiplier / scale.clamp_min(1e-8)


def shuffle_source_state(
    graph: SourceReuseGraph,
    pair: PairEncoding,
    state: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if state.shape[0] < 2:
        return state
    relation = (pair.sources >= graph.response_idx).long()
    origin_bin = torch.floor(pair.origin * 4).long().clamp(0, 3)
    strata = relation * 4 + origin_bin
    shuffled = state.clone()
    for key in torch.unique(strata):
        selected = torch.nonzero(strata == key, as_tuple=False).flatten()
        if selected.numel() > 1:
            order = selected[
                torch.randperm(
                    selected.numel(),
                    generator=generator,
                    device=selected.device,
                )
            ]
            shuffled[selected] = state[order]
    return shuffled


def rewired_state(
    graph: SourceReuseGraph,
    *,
    token: int,
    pair: PairEncoding,
    bank: SourceStateBank,
    source_state_encoder: SourceStateEncoder,
    response_provenance: torch.Tensor,
    config: GroundingGraphConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    available = torch.arange(
        graph.response_idx + token,
        device=graph.device,
        dtype=torch.long,
    )
    available_state = source_state_encoder.representations(bank, available)
    source_origin = graph.weight.new_ones(graph.num_tokens)
    source_origin[graph.response_idx :] = response_provenance
    rewired, changed = matched_endpoint_rewire(
        graph,
        token=token,
        sources=pair.sources,
        pair_origin=pair.origin,
        all_source_state=available_state,
        source_origin=source_origin,
        config=config,
        generator=generator,
    )
    return source_state_encoder.representations(bank, rewired), changed


def build_views(
    graph: SourceReuseGraph,
    *,
    token: int,
    raw_pair: PairEncoding,
    pair: PairEncoding,
    raw_state: torch.Tensor,
    state: torch.Tensor,
    bank: SourceStateBank,
    response_provenance: torch.Tensor,
    source_state_encoder: SourceStateEncoder,
    graph_encoder: RelationalGraphEncoder,
    config: GroundingGraphConfig,
    generator: torch.Generator,
) -> TokenViews:
    endpoint_state, changed = rewired_state(
        graph,
        token=token,
        pair=pair,
        bank=bank,
        source_state_encoder=source_state_encoder,
        response_provenance=response_provenance,
        config=config,
        generator=generator,
    )
    return TokenViews(
        raw=encode(
            graph_encoder,
            graph,
            token=token,
            pair=raw_pair,
            source_state=raw_state,
        ),
        full=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=state,
        ),
        no_prompt=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=state,
            view="no_prompt",
        ),
        no_response=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=state,
            view="no_response",
        ),
        perturbed=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=state,
            multiplier=perturbation(config, pair, generator=generator),
        ),
        no_state=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=torch.zeros_like(state),
        ),
        shuffled_state=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=shuffle_source_state(
                graph,
                pair,
                state,
                generator=generator,
            ),
        ),
        endpoint_rewired=encode(
            graph_encoder,
            graph,
            token=token,
            pair=pair,
            source_state=endpoint_state,
        ),
        changed_fraction=changed.float().mean(),
    )


def view_losses(
    views: TokenViews,
    target: TokenGroundingTargets,
    *,
    decoder: StructuredDecoder,
    config: GroundingGraphConfig,
) -> tuple[CounterfactualScores, torch.Tensor, torch.Tensor]:
    losses = {
        name: reconstruction_loss(
            decoder(getattr(views, name)),
            target,
            config,
        )
        for name in (
            "raw",
            "full",
            "no_prompt",
            "no_response",
            "perturbed",
            "no_state",
            "shuffled_state",
            "endpoint_rewired",
        )
    }
    scores = build_scores(
        raw=losses["raw"],
        full=losses["full"],
        no_prompt=losses["no_prompt"],
        no_response=losses["no_response"],
        perturbed=losses["perturbed"],
        no_state=losses["no_state"],
        shuffled_state=losses["shuffled_state"],
        endpoint_rewired=losses["endpoint_rewired"],
    )
    return scores, losses["raw"].total, losses["full"].total


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0 or float(weight.sum().detach()) <= 0:
        return value.new_tensor(0.0)
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)
