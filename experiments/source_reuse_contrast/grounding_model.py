"""Grounding-sensitive edge refinement and counterfactual graph scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .counterfactual import CounterfactualScores, reconstruction_loss
from .data import SourceReuseGraph
from .edge_refinement import SparseLayerHeadEncoder
from .graph_encoder import RelationalGraphEncoder, SourceStateEncoder, StructuredDecoder
from .grounding_config import GroundingGraphConfig
from .grounding_runtime import (
    build_views,
    encode,
    observed_weights,
    view_losses,
    weighted_mean,
)
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
    """Label-free sensitivity, soft edge refinement, and origin intervention."""

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

    def _empty_token(
        self,
        graph: SourceReuseGraph,
        *,
        token: int,
        bank,
    ) -> tuple[CounterfactualScores, torch.Tensor]:
        zero = graph.weight.new_tensor(0.0)
        embedding = graph.weight.new_zeros(self.config.hidden_dim)
        self.source_state.seed_response(
            bank,
            source_index=graph.response_idx + token,
            token_embedding=embedding,
        )
        return CounterfactualScores(*([zero] * 14)), embedding

    def forward(
        self,
        graph: SourceReuseGraph,
        *,
        seed: int | None = None,
    ) -> GroundingSequenceOutput:
        generator = torch.Generator(device=graph.device)
        generator.manual_seed(
            self.config.random_seed if seed is None else seed
        )
        targets = compute_grounding_targets(
            graph,
            received_topk=self.config.received_topk,
        )
        bank = self.source_state.initialize(graph)
        rows: list[CounterfactualScores] = []
        training_losses: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        diagnostics = {
            name: []
            for name in (
                "rewire",
                "sensitivity",
                "gate",
                "prompt_gate",
                "response_gate",
            )
        }
        embeddings: list[torch.Tensor] = []

        for token in range(graph.num_response_tokens):
            current = graph.token_slice(token)
            base_weight = graph.weight[current]
            if base_weight.numel() == 0:
                score, embedding = self._empty_token(
                    graph,
                    token=token,
                    bank=bank,
                )
                rows.append(score)
                valid.append(torch.tensor(False, device=graph.device))
                for values in diagnostics.values():
                    values.append(base_weight.new_tensor(0.0))
                embeddings.append(embedding)
                continue

            origin = targets.edge_origin[current]
            observed = observed_weights(
                self.config,
                base_weight,
                generator=generator,
            )
            raw_pair = self.pair_encoder(
                graph,
                token=token,
                observed_weight=observed,
                base_weight=base_weight,
                origin=origin,
                sensitivity=None,
                refine=False,
            )
            raw_state = self.source_state.representations(bank, raw_pair.sources)
            raw_embedding = encode(
                self.graph_encoder,
                graph,
                token=token,
                pair=raw_pair,
                source_state=raw_state,
            )
            target = targets.token(token)
            raw_loss = reconstruction_loss(
                self.decoder(raw_embedding),
                target,
                self.config,
            )
            gradient = torch.autograd.grad(
                raw_loss.total,
                observed,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            sensitivity = (
                observed.new_zeros(observed.shape)
                if gradient is None
                else (observed * gradient).abs().detach()
            )
            pair = self.pair_encoder(
                graph,
                token=token,
                observed_weight=observed,
                base_weight=base_weight,
                origin=origin,
                sensitivity=sensitivity,
                refine=True,
            )
            state = self.source_state.representations(bank, pair.sources)
            views = build_views(
                graph,
                token=token,
                raw_pair=raw_pair,
                pair=pair,
                raw_state=raw_state,
                state=state,
                bank=bank,
                response_provenance=targets.provenance[:, -1],
                source_state_encoder=self.source_state,
                graph_encoder=self.graph_encoder,
                config=self.config,
                generator=generator,
            )
            score, raw_total, full_total = view_losses(
                views,
                target,
                decoder=self.decoder,
                config=self.config,
            )
            gate_penalty = (
                pair.gate.mean() - self.config.gate_keep_target
            ).square()
            training_losses.append(
                full_total
                + self.config.raw_loss_weight * raw_total
                + self.config.gate_regularization * gate_penalty
            )
            rows.append(score)
            valid.append(torch.tensor(True, device=graph.device))
            diagnostics["rewire"].append(views.changed_fraction)
            diagnostics["sensitivity"].append(
                weighted_mean(sensitivity, base_weight)
            )
            diagnostics["gate"].append(pair.gate.mean())
            diagnostics["prompt_gate"].append(
                weighted_mean(pair.gate, pair.mass * pair.origin)
            )
            diagnostics["response_gate"].append(
                weighted_mean(pair.gate, pair.mass * (1 - pair.origin))
            )
            embeddings.append(views.full)

            memory_pair = self.pair_encoder(
                graph,
                token=token,
                observed_weight=base_weight,
                base_weight=base_weight,
                origin=origin,
                sensitivity=None,
                refine=False,
            )
            self.source_state.update_reuse(
                bank,
                token=token,
                sources=memory_pair.sources,
                pair_embedding=memory_pair.embedding,
                pair_mass=memory_pair.mass,
                token_embedding=views.full,
            )
            self.source_state.seed_response(
                bank,
                source_index=graph.response_idx + token,
                token_embedding=views.full,
            )
            if (token + 1) % self.config.bptt_steps == 0:
                bank.detach()

        loss = (
            torch.stack(training_losses).mean()
            if training_losses
            else graph.weight.sum() * 0
        )

        def score(name: str) -> torch.Tensor:
            return torch.stack([getattr(row, name) for row in rows])

        return GroundingSequenceOutput(
            loss=loss,
            valid=torch.stack(valid).bool(),
            reconstruction=score("reconstruction"),
            raw_reconstruction=score("raw_reconstruction"),
            prompt_gain=score("prompt_gain"),
            response_gain=score("response_gain"),
            closure=score("closure"),
            fragility=score("fragility"),
            refinement_gain=score("refinement_gain"),
            state_gain=score("state_gain"),
            memory_specificity=score("memory_specificity"),
            endpoint_specificity=score("endpoint_specificity"),
            rewire_changed_fraction=torch.stack(diagnostics["rewire"]),
            sensitivity_mean=torch.stack(diagnostics["sensitivity"]),
            gate_mean=torch.stack(diagnostics["gate"]),
            prompt_gate_mean=torch.stack(diagnostics["prompt_gate"]),
            response_gate_mean=torch.stack(diagnostics["response_gate"]),
            embedding=torch.stack(embeddings),
        )
