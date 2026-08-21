"""Grounding-sensitive edge refinement and counterfactual graph scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .counterfactual import CounterfactualScores, build_scores, reconstruction_loss
from .data import SourceReuseGraph
from .edge_refinement import SparseLayerHeadEncoder
from .graph_encoder import RelationalGraphEncoder, SourceStateEncoder, StructuredDecoder
from .grounding_config import GroundingGraphConfig
from .grounding_nulls import matched_endpoint_rewire
from .provenance import compute_grounding_targets


@dataclass(frozen=True)
class GroundingSequenceOutput:
    loss: torch.Tensor
    valid: torch.Tensor
    reconstruction: torch.Tensor
    raw_reconstruction: torch.Tensor
    prompt_gain: torch.Tensor
    response_gain: torch.Tensor
    closure: torch.Tensor
    fragility: torch.Tensor
    refinement_gain: torch.Tensor
    state_gain: torch.Tensor
    memory_specificity: torch.Tensor
    endpoint_specificity: torch.Tensor
    rewire_changed_fraction: torch.Tensor
    sensitivity_mean: torch.Tensor
    gate_mean: torch.Tensor
    prompt_gate_mean: torch.Tensor
    response_gate_mean: torch.Tensor
    embedding: torch.Tensor


class GroundingSensitiveGraphModel(nn.Module):
    """Label-free edge sensitivity, refinement, and origin intervention model."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: GroundingGraphConfig | None = None,
    ):
        super().__init__()
        self.config = GroundingGraphConfig() if config is None else config
        self.config.validate()
        self.pair_encoder = SparseLayerHeadEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            config=self.config,
        )
        self.source_state = SourceStateEncoder(self.config)
        self.graph_encoder = RelationalGraphEncoder(self.config)
        self.decoder = StructuredDecoder(
            num_layers=num_layers,
            num_heads=num_heads,
            config=self.config,
        )

    def _observed_weights(
        self,
        base_weight: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if base_weight.numel() <= 1 or self.config.edge_mask_rate <= 0.0:
            return base_weight.detach().clone().requires_grad_(True)
        keep = torch.rand(
            base_weight.shape,
            generator=generator,
            device=base_weight.device,
        ) >= self.config.edge_mask_rate
        if not bool(keep.any()):
            keep[base_weight.argmax()] = True
        return (base_weight * keep).detach().clone().requires_grad_(True)

    def _perturbation(
        self,
        pair_mass: torch.Tensor,
        pair_gate: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if pair_mass.numel() == 0:
            return pair_mass
        noise = torch.randn(
            pair_mass.shape,
            generator=generator,
            device=pair_mass.device,
        ) * self.config.perturbation_scale
        multiplier = noise.exp()
        reference = pair_mass * pair_gate
        normalizer = (reference * multiplier).sum() / reference.sum().clamp_min(1e-8)
        return multiplier / normalizer.clamp_min(1e-8)

    def _shuffle_source_state(
        self,
        graph: SourceReuseGraph,
        pair,
        source_state: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Shuffle source states only within relation/origin strata."""
        if source_state.shape[0] < 2:
            return source_state
        relation = (pair.sources >= graph.response_idx).long()
        origin_bin = torch.floor(pair.origin * 4.0).long().clamp(0, 3)
        key = relation * 4 + origin_bin
        shuffled = source_state.clone()
        for value in torch.unique(key):
            selected = torch.nonzero(key == value, as_tuple=False).flatten()
            if selected.numel() > 1:
                order = selected[torch.randperm(selected.numel(), generator=generator, device=selected.device)]
                shuffled[selected] = source_state[order]
        return shuffled

    @staticmethod
    def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0 or float(weight.sum().detach()) <= 0.0:
            return value.new_tensor(0.0)
        return (value * weight).sum() / weight.sum().clamp_min(1e-8)

    def forward(
        self,
        graph: SourceReuseGraph,
        *,
        seed: int | None = None,
    ) -> GroundingSequenceOutput:
        seed = self.config.random_seed if seed is None else seed
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(seed)
        targets = compute_grounding_targets(
            graph,
            received_topk=self.config.received_topk,
        )
        bank = self.source_state.initialize(graph)

        training_losses: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        scores: list[CounterfactualScores] = []
        sensitivity_mean: list[torch.Tensor] = []
        gate_mean: list[torch.Tensor] = []
        prompt_gate_mean: list[torch.Tensor] = []
        response_gate_mean: list[torch.Tensor] = []
        rewire_changed_fraction: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []

        for token in range(graph.num_response_tokens):
            current = graph.token_slice(token)
            base_weight = graph.weight[current]
            origin = targets.edge_origin[current]
            if base_weight.numel() == 0:
                zero = graph.weight.new_tensor(0.0)
                token_embedding = graph.weight.new_zeros(self.config.hidden_dim)
                empty_score = CounterfactualScores(
                    reconstruction=zero,
                    raw_reconstruction=zero,
                    prompt_removed=zero,
                    response_removed=zero,
                    perturbed=zero,
                    prompt_gain=zero,
                    response_gain=zero,
                    closure=zero,
                    fragility=zero,
                    refinement_gain=zero,
                    state_gain=zero,
                    memory_specificity=zero,
                    endpoint_specificity=zero,
                )
                scores.append(empty_score)
                valid.append(torch.tensor(False, device=graph.device))
                sensitivity_mean.append(zero)
                gate_mean.append(zero)
                prompt_gate_mean.append(zero)
                response_gate_mean.append(zero)
                rewire_changed_fraction.append(zero)
                embeddings.append(token_embedding)
                response_source = graph.response_idx + token
                self.source_state.seed_response(
                    bank,
                    source_index=response_source,
                    token_embedding=token_embedding,
                )
                continue

            observed_weight = self._observed_weights(base_weight, generator=generator)
            raw_pair = self.pair_encoder(
                graph,
                token=token,
                observed_weight=observed_weight,
                base_weight=base_weight,
                origin=origin,
                sensitivity=None,
                refine=False,
            )
            raw_source_state = self.source_state.²È="24€€€€€€€€€Í•¹Í¥Ñ¥Ù¥ÑäõÍ•¹Í¥Ñ¥Ù¥Ñä°(€€€€€€€€€€€€€€€É•™¥¹”õQÉÕ”°(€€€€€€€€€€€€¤(€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”€ôÍ•±˜¹Í½ÕÉ•}ÍÑ…Ñ”¹É•ÁÉ•Í•¹Ñ…Ñ¥½¹Ì¡‰…¹¬°Á…¥È¹Í½ÕÉ•Ì¤(€€€€€€€€€€€™Õ±±}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÍ½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰™Õ±°ˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}ÁÉ½µÁÑ}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÍ½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰¹½}ÁÉ½µÁÐˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}É•ÍÁ½¹Í•}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÍ½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰¹½}É•ÍÁ½¹Í”ˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€Á•ÉÑÕÉ‰…Ñ¥½¸€ôÍ•±˜¹}Á•ÉÑÕÉ‰…Ñ¥½¸¡Á…¥È¹µ…ÍÌ°Á…¥È¹…Ñ”°•¹•É…Ñ½Èõ•¹•É…Ñ½È¤(€€€€€€€€€€€Á•ÉÑÕÉ‰•‘}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÍ½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰™Õ±°ˆ°(€€€€€€€€€€€€€€€Á…¥É}µÕ±Ñ¥Á±¥•ÈõÁ•ÉÑÕÉ‰…Ñ¥½¸°(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}ÍÑ…Ñ•}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÑ½É ¹é•É½Í}±¥­”¡Í½ÕÉ•}ÍÑ…Ñ”¤°(€€€€€€€€€€€€€€€Ù¥•Üô‰™Õ±°ˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€Í¡Õ™™±•‘}ÍÑ…Ñ”€ôÍ•±˜¹}Í¡Õ™™±•}Í½ÕÉ•}ÍÑ…Ñ” (€€€€€€€€€€€€€€€É…Á °Á…¥È°Í½ÕÉ•}ÍÑ…Ñ”°•¹•É…Ñ½Èõ•¹•É…Ñ½È(€€€€€€€€€€€€¤(€€€€€€€€€€€Í¡Õ™™±•‘}ÍÑ…Ñ•}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÍ¡Õ™™±•‘}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰™Õ±°ˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€…Ù…¥±…‰±”€ôÑ½É ¹…É…¹” (€€€€€€€€€€€€€€€É…Á ¹É•ÍÁ½¹Í•}¥‘à€¬Ñ½­•¸°‘•Ù¥”õÉ…Á ¹‘•Ù¥”°‘ÑåÁ”õÑ½É ¹±½¹œ(€€€€€€€€€€€€¤(€€€€€€€€€€€…±±}Í½ÕÉ•}ÍÑ…Ñ”€ôÍ•±˜¹Í½ÕÉ•}ÍÑ…Ñ”¹É•ÁÉ•Í•¹Ñ…Ñ¥½¹Ì¡‰…¹¬°…Ù…¥±…‰±”¤(€€€€€€€€€€€Í½ÕÉ•}½É¥¥¸€ôÉ…Á ¹Ý•¥¡Ð¹¹•Ý}½¹•Ì¡É…Á ¹¹Õµ}Ñ½­•¹Ì¤(€€€€€€€€€€€Í½ÕÉ•}½É¥¥¹mÉ…Á ¹É•ÍÁ½¹Í•}¥‘à€ét€ôÑ…É•ÑÌ¹ÁÉ½Ù•¹…¹•lè°€´Åt(€€€€€€€€€€€É•Ý¥É•‘}Í½ÕÉ”°É•Ý¥É•‘}¡…¹•€ôµ…Ñ¡•‘}•¹‘Á½¥¹Ñ}É•Ý¥É” (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Í½ÕÉ•ÌõÁ…¥È¹Í½ÕÉ•Ì°(€€€€€€€€€€€€€€€Á…¥É}½É¥¥¸õÁ…¥È¹½É¥¥¸°(€€€€€€€€€€€€€€€…±±}Í½ÕÉ•}ÍÑ…Ñ”õ…±±}Í½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Í½ÕÉ•}½É¥¥¸õÍ½ÕÉ•}½É¥¥¸°(€€€€€€€€€€€€€€€½¹™¥œõÍ•±˜¹½¹™¥œ°(€€€€€€€€€€€€€€€•¹•É…Ñ½Èõ•¹•É…Ñ½È°(€€€€€€€€€€€€¤(€€€€€€€€€€€É•Ý¥É•‘}Í½ÕÉ•}ÍÑ…Ñ”€ôÍ•±˜¹Í½ÕÉ•}ÍÑ…Ñ”¹É•ÁÉ•Í•¹Ñ…Ñ¥½¹Ì (€€€€€€€€€€€€€€€‰…¹¬°É•Ý¥É•‘}Í½ÕÉ”(€€€€€€€€€€€€¤(€€€€€€€€€€€•¹‘Á½¥¹Ñ}É•Ý¥É•‘}•µ‰•‘‘¥¹œ€ôÍ•±˜¹É…Á¡}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Á…¥ÈõÁ…¥È°(€€€€€€€€€€€€€€€Í½ÕÉ•}ÍÑ…Ñ”õÉ•Ý¥É•‘}Í½ÕÉ•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€Ù¥•Üô‰™Õ±°ˆ°(€€€€€€€€€€€€¤((€€€€€€€€€€€™Õ±±}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡™Õ±±}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}ÁÉ½µÁÑ}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡¹½}ÁÉ½µÁÑ}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}É•ÍÁ½¹Í•}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡¹½}É•ÍÁ½¹Í•}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€Á•ÉÑÕÉ‰•‘}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡Á•ÉÑÕÉ‰•‘}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€¹½}ÍÑ…Ñ•}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡¹½}ÍÑ…Ñ•}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€Í¡Õ™™±•‘}ÍÑ…Ñ•}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡Í¡Õ™™±•‘}ÍÑ…Ñ•}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€•¹‘Á½¥¹Ñ}É•Ý¥É•‘}±½ÍÌ€ôÉ•½¹ÍÑÉÕÑ¥½¹}±½ÍÌ (€€€€€€€€€€€€€€€Í•±˜¹‘•½‘•È¡•¹‘Á½¥¹Ñ}É•Ý¥É•‘}•µ‰•‘‘¥¹œ¤°Ñ½­•¹}Ñ…É•Ð°Í•±˜¹½¹™¥œ(€€€€€€€€€€€€¤(€€€€€€€€€€€ÕÉÉ•¹Ñ}Í½É•Ì€ô‰Õ¥±‘}Í½É•Ì (€€€€€€€€€€€€€€€É…ÜõÉ…Ý}±½ÍÌ°(€€€€€€€€€€€€€€€™Õ±°õ™Õ±±}±½ÍÌ°(€€€€€€€€€€€€€€€¹½}ÁÉ½µÁÐõ¹½}ÁÉ½µÁÑ}±½ÍÌ°(€€€€€€€€€€€€€€€¹½}É•ÍÁ½¹Í”õ¹½}É•ÍÁ½¹Í•}±½ÍÌ°(€€€€€€€€€€€€€€€Á•ÉÑÕÉ‰•õÁ•ÉÑÕÉ‰•‘}±½ÍÌ°(€€€€€€€€€€€€€€€¹½}ÍÑ…Ñ”õ¹½}ÍÑ…Ñ•}±½ÍÌ°(€€€€€€€€€€€€€€€Í¡Õ™™±•‘}ÍÑ…Ñ”õÍ¡Õ™™±•‘}ÍÑ…Ñ•}±½ÍÌ°(€€€€€€€€€€€€€€€•¹‘Á½¥¹Ñ}É•Ý¥É•õ•¹‘Á½¥¹Ñ}É•Ý¥É•‘}±½ÍÌ°(€€€€€€€€€€€€¤(€€€€€€€€€€€…Ñ•}Á•¹…±Ñä€ô€ (€€€€€€€€€€€€€€€Á…¥È¹…Ñ”¹µ•…¸ ¤€´Í•±˜¹½¹™¥œ¹…Ñ•}­••Á}Ñ…É•Ð(€€€€€€€€€€€€¤¹ÍÅÕ…É” ¤(€€€€€€€€€€€ÑÉ…¥¹¥¹}±½ÍÍ•Ì¹…ÁÁ•¹ (€€€€€€€€€€€€€€€™Õ±±}±½ÍÌ¹Ñ½Ñ…°(€€€€€€€€€€€€€€€€¬Í•±˜¹½¹™¥œ¹É…Ý}±½ÍÍ}Ý•¥¡Ð€¨É…Ý}±½ÍÌ¹Ñ½Ñ…°(€€€€€€€€€€€€€€€€¬Í•±˜¹½¹™¥œ¹…Ñ•}É•Õ±…É¥é…Ñ¥½¸€¨…Ñ•}Á•¹…±Ñä(€€€€€€€€€€€€¤(€€€€€€€€€€€Í½É•Ì¹…ÁÁ•¹¡ÕÉÉ•¹Ñ}Í½É•Ì¤(€€€€€€€€€€€Ù…±¥¹…ÁÁ•¹¡Ñ½É ¹Ñ•¹Í½È¡QÉÕ”°‘•Ù¥”õÉ…Á ¹‘•Ù¥”¤¤(€€€€€€€€€€€Í•¹Í¥Ñ¥Ù¥Ñå}µ•…¸¹…ÁÁ•¹ (€€€€€€€€€€€€€€€Í•±˜¹}Ý•¥¡Ñ•‘}µ•…¸¡Í•¹Í¥Ñ¥Ù¥Ñä°‰…Í•}Ý•¥¡Ð¤(€€€€€€€€€€€€¤(€€€€€€€€€€€…Ñ•}µ•…¸¹…ÁÁ•¹¡Á…¥È¹…Ñ”¹µ•…¸ ¤¤(€€€€€€€€€€€ÁÉ½µÁÑ}Ý•¥¡Ð€ôÁ…¥È¹µ…ÍÌ€¨Á…¥È¹½É¥¥¸(€€€€€€€€€€€É•ÍÁ½¹Í•}Ý•¥¡Ð€ôÁ…¥È¹µ…ÍÌ€¨€ Ä¸À€´Á…¥È¹½É¥¥¸¤(€€€€€€€€€€€ÁÉ½µÁÑ}…Ñ•}µ•…¸¹…ÁÁ•¹¡Í•±˜¹}Ý•¥¡Ñ•‘}µ•…¸¡Á…¥È¹…Ñ”°ÁÉ½µÁÑ}Ý•¥¡Ð¤¤(€€€€€€€€€€€É•ÍÁ½¹Í•}…Ñ•}µ•…¸¹…ÁÁ•¹¡Í•±˜¹}Ý•¥¡Ñ•‘}µ•…¸¡Á…¥È¹…Ñ”°É•ÍÁ½¹Í•}Ý•¥¡Ð¤¤(€€€€€€€€€€€É•Ý¥É•}¡…¹•‘}™É…Ñ¥½¸¹…ÁÁ•¹¡É•Ý¥É•‘}¡…¹•¹™±½…Ð ¤¹µ•…¸ ¤¤(€€€€€€€€€€€•µ‰•‘‘¥¹Ì¹…ÁÁ•¹¡™Õ±±}•µ‰•‘‘¥¹œ¤((€€€€€€€€€€€µ•µ½Éå}Á…¥È€ôÍ•±˜¹Á…¥É}•¹½‘•È (€€€€€€€€€€€€€€€É…Á °(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€½‰Í•ÉÙ•‘}Ý•¥¡Ðõ‰…Í•}Ý•¥¡Ð°(€€€€€€€€€€€€€€€‰…Í•}Ý•¥¡Ðõ‰…Í•}Ý•¥¡Ð°(€€€€€€€€€€€€€€€½É¥¥¸õ½É¥¥¸°(€€€€€€€€€€€€€€€Í•¹Í¥Ñ¥Ù¥Ñäõ9½¹”°(€€€€€€€€€€€€€€€É•™¥¹”õ…±Í”°(€€€€€€€€€€€€¤(€€€€€€€€€€€Í•±˜¹Í½ÕÉ•}ÍÑ…Ñ”¹ÕÁ‘…Ñ•}É•ÕÍ” (€€€€€€€€€€€€€€€‰…¹¬°(€€€€€€€€€€€€€€€Ñ½­•¸õÑ½­•¸°(€€€€€€€€€€€€€€€Í½ÕÉ•Ìõµ•µ½Éå}Á…¥È¹Í½ÕÉ•Ì°(€€€€€€€€€€€€€€€Á…¥É}•µ‰•‘‘¥¹œõµ•µ½Éå}Á…¥È¹•µ‰•‘‘¥¹œ°(€€€€€€€€€€€€€€€Á…¥É}µ…ÍÌõµ•µ½Éå}Á…¥È¹µ…ÍÌ°(€€€€€€€€€€€€€€€Ñ½­•¹}•µ‰•‘‘¥¹œõ™Õ±±}•µ‰•‘‘¥¹œ°(€€€€€€€€€€€€¤(€€€€€€€€€€€É•ÍÁ½¹Í•}Í½ÕÉ”€ôÉ…Á ¹É•ÍÁ½¹Í•}¥‘à€¬Ñ½­•¸(€€€€€€€€€€€Í•±˜¹Í½ÕÉ•}ÍÑ…Ñ”¹Í••‘}É•ÍÁ½¹Í” (€€€€€€€€€€€€€€€‰…¹¬°(€€€€€€€€€€€€€€€Í½ÕÉ•}¥¹‘•àõÉ•ÍÁ½¹Í•}Í½ÕÉ”°(€€€€€€€€€€€€€€€Ñ½­•¹}•µ‰•‘‘¥¹œõ™Õ±±}•µ‰•‘‘¥¹œ°(€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜€¡Ñ½­•¸€¬€Ä¤€”Í•±˜¹½¹™¥œ¹‰ÁÑÑ}ÍÑ•ÁÌ€ôô€Àè(€€€€€€€€€€€€€€€‰…¹¬¹‘•Ñ…  ¤((€€€€€€€¥˜ÑÉ…¥¹¥¹}±½ÍÍ•Ìè(€€€€€€€€€€€±½ÍÌ€ôÑ½É ¹ÍÑ…¬¡ÑÉ…¥¹¥¹}±½ÍÍ•Ì¤¹µ•…¸ ¤(€€€€€€€•±Í”è(€€€€€€€€€€€±½ÍÌ€ôÉ…Á ¹Ý•¥¡Ð¹ÍÕ´ ¤€¨€À¸À((€€€€€€€‘•˜ÍÑ…­}Í½É”¡¹…µ”èÍÑÈ¤€´øÑ½É ¹Q•¹Í½Èè(€€€€€€€€€€€É•ÑÕÉ¸Ñ½É ¹ÍÑ…¬¡m•Ñ…ÑÑÈ¡Ù…±Õ”°¹…µ”¤™½ÈÙ…±Õ”¥¸Í½É•Ít¤((€€€€€€€É•ÑÕÉ¸É½Õ¹‘¥¹M•ÅÕ•¹•=ÕÑÁÕÐ (€€€€€€€€€€€±½ÍÌõ±½ÍÌ°(€€€€€€€€€€€Ù…±¥õÑ½É ¹ÍÑ…¬¡Ù…±¥¤¹‰½½° ¤°(€€€€€€€€€€€É•½¹ÍÑÉÕÑ¥½¸õÍÑ…­}Í½É” ‰É•½¹ÍÑÉÕÑ¥½¸ˆ¤°(€€€€€€€€€€€É…Ý}É•½¹ÍÑÉÕÑ¥½¸õÍÑ…­}Í½É” ‰É…Ý}É•½¹ÍÑÉÕÑ¥½¸ˆ¤°(€€€€€€€€€€€ÁÉ½µÁÑ}…¥¸õÍÑ…­}Í½É” ‰ÁÉ½µÁÑ}…¥¸ˆ¤°(€€€€€€€€€€€É•ÍÁ½¹Í•}…¥¸õÍÑ…­}Í½É” ‰É•ÍÁ½¹Í•}…¥¸ˆ¤°(€€€€€€€€€€€±½ÍÕÉ”õÍÑ…­}Í½É” ‰±½ÍÕÉ”ˆ¤°(€€€€€€€€€€€™É…¥±¥ÑäõÍÑ…­}Í½É” ‰™É…¥±¥Ñäˆ¤°(€€€€€€€€€€€É•™¥¹•µ•¹Ñ}…¥¸õÍÑ…­}Í½É” ‰É•™¥¹•µ•¹Ñ}…¥¸ˆ¤°(€€€€€€€€€€€ÍÑ…Ñ•}…¥¸õÍÑ…­}Í½É” ‰ÍÑ…Ñ•}…¥¸ˆ¤°(€€€€€€€€€€€µ•µ½Éå}ÍÁ•¥™¥¥ÑäõÍÑ…­}Í½É” ‰µ•µ½Éå}ÍÁ•¥™¥¥Ñäˆ¤°(€€€€€€€€€€€•¹‘Á½¥¹Ñ}ÍÁ•¥™¥¥ÑäõÍÑ…­}Í½É” ‰•¹‘Á½¥¹Ñ}ÍÁ•¥™¥¥Ñäˆ¤°(€€€€€€€€€€€É•Ý¥É•}¡…¹•‘}™É…Ñ¥½¸õÑ½É ¹ÍÑ…¬¡É•Ý¥É•}¡…¹•‘}™É…Ñ¥½¸¤°(€€€€€€€€€€€Í•¹Í¥Ñ¥Ù¥Ñå}µ•…¸õÑ½É ¹ÍÑ…¬¡Í•¹Í¥Ñ¥Ù¥Ñå}µ•…¸¤°(€€€€€€€€€€€…Ñ•}µ•…¸õÑ½É ¹ÍÑ…¬¡…Ñ•}µ•…¸¤°(€€€€€€€€€€€ÁÉ½µÁÑ}…Ñ•}µ•…¸õÑ½É ¹ÍÑ…¬¡ÁÉ½µÁÑ}…Ñ•}µ•…¸¤°(€€€€€€€€€€€É•ÍÁ½¹Í•}…Ñ•}µ•…¸õÑ½É ¹ÍÑ…¬¡É•ÍÁ½¹Í•}…Ñ•}µ•…¸¤°(€€€€€€€€€€€•µ‰•‘‘¥¹œõÑ½É ¹ÍÑ…¬¡•µ‰•‘‘¥¹Ì¤°(€€€€€€€€¤(