"""Masked exact-source prediction from causal source-reuse history."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SourceReuseConfig
from .data import PROMPT, RESPONSE, SourceReuseGraph
from .sampler import matched_candidate_batch


@dataclass(frozen=True)
class PredictabilityScores:
    endpoint_nll: torch.Tensor
    shuffled_nll: torch.Tensor
    margin: torch.Tensor
    accuracy: torch.Tensor
    valid_pairs: torch.Tensor
    candidate_count: torch.Tensor
    positive_logit: torch.Tensor
    hardest_negative_logit: torch.Tensor
    mean_match_distance: torch.Tensor
    query_embedding: torch.Tensor
    source_embedding: torch.Tensor

    @property
    def valid(self) -> torch.Tensor:
        return self.valid_pairs > 0


class SetReadout(nn.Module):
    """Compact permutation-invariant set encoder."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.empty = nn.Parameter(torch.zeros(hidden_dim))
        self.gate = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        if values.shape[0] == 0:
            return self.empty
        log_mass = mass.clamp_min(1e-8).log()
        weight = F.softmax(self.gate(values).squeeze(-1) + log_mass, dim=0)
        weighted = (weight.unsqueeze(-1) * values).sum(dim=0)
        mean = values.mean(dim=0)
        maximum = values.max(dim=0).values
        return self.output(torch.cat((weighted, mean, maximum), dim=-1))


class IncidenceEncoder(nn.Module):
    """Encode current layer/head marks without exact source identity."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: SourceReuseConfig,
    ):
        super().__init__()
        self.config = config
        self.layer = nn.Embedding(num_layers, config.layer_embedding_dim)
        self.head = nn.Embedding(num_heads, config.head_embedding_dim)
        self.relation = nn.Embedding(2, config.relation_embedding_dim)
        self.source_bin = nn.Embedding(
            max(config.prompt_position_bins, config.response_lag_bins),
            config.source_bin_embedding_dim,
        )
        self.usage = nn.Embedding(config.usage_bins, config.usage_embedding_dim)
        input_dim = (
            config.layer_embedding_dim
            + config.head_embedding_dim
            + config.relation_embedding_dim
            + config.source_bin_embedding_dim
            + config.usage_embedding_dim
            + 2
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        *,
        layer: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        source_bin: torch.Tensor,
        usage_bucket: torch.Tensor,
        weight: torch.Tensor,
        attention_floor: float,
    ) -> torch.Tensor:
        numeric = torch.stack(
            (
                weight,
                torch.log1p(weight / max(attention_floor, 1e-8)),
            ),
            dim=-1,
        )
        value = torch.cat(
            (
                self.layer(layer),
                self.head(head),
                self.relation(relation),
                self.source_bin(source_bin),
                self.usage(usage_bucket),
                numeric,
            ),
            dim=-1,
        )
        return self.network(value)


class SourceStateEncoder(nn.Module):
    """Separate immutable birth state from learned subsequent reuse state."""

    def __init__(self, config: SourceReuseConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.prompt_bin = nn.Embedding(config.prompt_position_bins, hidden)
        self.prompt_birth = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.response_birth = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.reuse_input = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.reuse_update = nn.GRUCell(hidden, hidden)
        self.relation = nn.Embedding(2, hidden)
        self.source_bin = nn.Embedding(
            max(config.prompt_position_bins, config.response_lag_bins), hidden
        )
        self.usage = nn.Embedding(config.usage_bins, hidden)
        self.stats = nn.Sequential(
            nn.Linear(hidden * 3 + 3, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.memory = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )

    def initialize(
        self, graph: SourceReuseGraph
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hidden = self.config.hidden_dim
        if graph.response_idx:
            position = torch.arange(graph.response_idx, device=graph.device)
            prompt_bin = torch.minimum(
                position * self.config.prompt_position_bins // graph.response_idx,
                torch.full_like(position, self.config.prompt_position_bins - 1),
            )
            normalized = position.float() / float(max(graph.response_idx - 1, 1))
            prompt = self.prompt_birth(
                torch.cat((self.prompt_bin(prompt_bin), normalized[:, None]), dim=-1)
            )
            birth = [prompt[index] for index in range(graph.response_idx)]
        else:
            birth = []
        zero = torch.zeros(hidden, device=graph.device)
        birth.extend(zero.clone() for _ in range(graph.num_response_tokens))
        reuse = [zero.clone() for _ in range(graph.num_tokens)]
        return birth, reuse

    def seed_response(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return self.response_birth(token_embedding)

    def update_reuse(
        self,
        reuse: list[torch.Tensor],
        sources: torch.Tensor,
        pair_embedding: torch.Tensor,
        token_embedding: torch.Tensor,
        *,
        token: int,
        cumulative_mass: torch.Tensor,
        last_used: torch.Tensor,
    ) -> None:
        if sources.numel() == 0:
            return
        previous = torch.stack([reuse[int(source)] for source in sources.tolist()])
        repeated_token = token_embedding.expand(pair_embedding.shape[0], -1)
        gap = torch.tensor(
            [
                token + 1
                if int(last_used[int(source)]) < 0
                else token - int(last_used[int(source)])
                for source in sources.tolist()
            ],
            dtype=pair_embedding.dtype,
            device=pair_embedding.device,
        )
        numeric = torch.stack(
            (
                torch.log1p(gap),
                torch.log1p(cumulative_mass[sources]),
                torch.full_like(gap, math.log1p(token + 1)),
            ),
            dim=-1,
        )
        update_input = self.reuse_input(
            torch.cat((pair_embedding, repeated_token, numeric), dim=-1)
        )
        updated = self.reuse_update(update_input, previous)
        for index, source in enumerate(sources.tolist()):
            reuse[source] = updated[index]

    def candidate_embeddings(
        self,
        *,
        graph: SourceReuseGraph,
        query: int,
        candidate_source: torch.Tensor,
        candidate_mask: torch.Tensor,
        use_count: torch.Tensor,
        cumulative_mass: torch.Tensor,
        last_used: torch.Tensor,
        birth: list[torch.Tensor],
        reuse: list[torch.Tensor],
        source_bin: torch.Tensor,
        usage_bucket: torch.Tensor,
        shuffled: bool,
    ) -> torch.Tensor:
        safe_source = candidate_source.clamp_min(0)
        relation = (safe_source >= graph.response_idx).long()
        gap = torch.where(
            last_used[safe_source] < 0,
            torch.full_like(safe_source, query + 1),
            query - last_used[safe_source],
        ).float()
        numeric = torch.stack(
            (
                torch.log1p(use_count[safe_source].float()),
                torch.log1p(cumulative_mass[safe_source]),
                torch.log1p(gap.clamp_min(0.0)),
            ),
            dim=-1,
        )
        stats = self.stats(
            torch.cat(
                (
                    self.relation(relation),
                    self.source_bin(source_bin),
                    self.usage(usage_bucket),
                    numeric,
                ),
                dim=-1,
            )
        )

        birth_state = torch.stack(
            [birth[int(source)] for source in safe_source.flatten().tolist()]
        ).reshape(*safe_source.shape, -1)
        reuse_state = torch.stack(
            [reuse[int(source)] for source in safe_source.flatten().tolist()]
        ).reshape(*safe_source.shape, -1)
        if self.config.memory_mode == "current":
            memory = torch.zeros_like(stats)
        elif self.config.memory_mode == "birth":
            memory = birth_state
        else:
            memory = self.memory(torch.cat((birth_state, reuse_state), dim=-1))

        if shuffled and self.config.memory_mode != "current":
            shuffled_memory = memory.clone()
            for row in range(memory.shape[0]):
                count = int(candidate_mask[row].sum())
                if count > 1:
                    shuffled_memory[row, :count] = memory[row, :count].roll(1, dims=0)
            memory = shuffled_memory

        value = self.candidate(torch.cat((stats, memory), dim=-1))
        return value * candidate_mask.unsqueeze(-1)


class SourceReusePredictor(nn.Module):
    """CaSH v2: predict masked exact sources from causal reuse history."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: SourceReuseConfig | None = None,
    ):
        super().__init__()
        self.config = SourceReuseConfig() if config is None else config
        self.config.validate()
        hidden = self.config.hidden_dim
        self.incidence = IncidenceEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            config=self.config,
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.PReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.readout = SetReadout(hidden, self.config.dropout)
        self.control = nn.Sequential(
            nn.Linear(4, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.query = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.PReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.source_state = SourceStateEncoder(self.config)

    def _source_bins(
        self, graph: SourceReuseGraph, token: int, source: torch.Tensor
    ) -> torch.Tensor:
        prompt = source < graph.response_idx
        prompt_bin = torch.minimum(
            source * self.config.prompt_position_bins // max(graph.response_idx, 1),
            torch.full_like(source, self.config.prompt_position_bins - 1),
        )
        lag = token - (source - graph.response_idx)
        response_bin = torch.minimum(
            torch.floor(torch.log2(lag.clamp_min(1).float())).long(),
            torch.full_like(source, self.config.response_lag_bins - 1),
        )
        return torch.where(prompt, prompt_bin, response_bin)

    def _usage_bins(self, use_count: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        value = torch.floor(torch.log2(use_count[source].float() + 1.0)).long()
        return value.clamp_max(self.config.usage_bins - 1)

    def _pair_embeddings(
        self,
        graph: SourceReuseGraph,
        token: int,
        use_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = graph.token_slice(token)
        if current.start == current.stop:
            empty_source = torch.empty(0, dtype=torch.long, device=graph.device)
            empty_embedding = graph.weight.new_empty((0, self.config.hidden_dim))
            empty_mass = graph.weight.new_empty(0)
            return empty_source, empty_embedding, empty_mass

        source = graph.source[current]
        weight = graph.weight[current]
        relation = torch.where(
            source < graph.response_idx,
            torch.full_like(source, PROMPT),
            torch.full_like(source, RESPONSE),
        )
        event = self.incidence(
            layer=graph.layer[current],
            head=graph.head[current],
            relation=relation,
            source_bin=self._source_bins(graph, token, source),
            usage_bucket=self._usage_bins(use_count, source),
            weight=weight,
            attention_floor=graph.attention_floor,
        )
        unique_source, inverse = torch.unique(source, sorted=True, return_inverse=True)
        pair_mass = weight.new_zeros(unique_source.shape[0])
        pair_mass.index_add_(0, inverse, weight)
        pair_count = weight.new_zeros(unique_source.shape[0])
        pair_count.index_add_(0, inverse, torch.ones_like(weight))
        pair_sum = event.new_zeros((unique_source.shape[0], event.shape[-1]))
        pair_sum.index_add_(0, inverse, event * weight.unsqueeze(-1))
        pair_mean = pair_sum / pair_mass.clamp_min(1e-8).unsqueeze(-1)
        pair_feature = torch.stack(
            (
                torch.log1p(pair_mass / max(graph.attention_floor, 1e-8)),
                pair_count / float(graph.num_layers * graph.num_heads),
            ),
            dim=-1,
        )
        pair_embedding = self.pair_projection(
            torch.cat((pair_mean, pair_feature), dim=-1)
        )
        return unique_source, pair_embedding, pair_mass

    def _token_control(
        self, graph: SourceReuseGraph, token: int, pair_mass: torch.Tensor
    ) -> torch.Tensor:
        channel_count = float(graph.num_layers * graph.num_heads)
        diagonal = graph.diagonal[token].mean()
        retained = pair_mass.sum() / channel_count
        known = (diagonal + retained).clamp(0.0, 1.0)
        unresolved = 1.0 - known
        causal_position = diagonal.new_tensor(math.log1p(token + 1) / 8.0)
        return self.control(
            torch.stack((diagonal, retained, unresolved, causal_position))
        )

    def _score_candidates(
        self,
        query_embedding: torch.Tensor,
        candidate_embedding: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(query_embedding, dim=-1)
        candidate = F.normalize(candidate_embedding, dim=-1)
        logits = torch.einsum("ph,pch->pc", query, candidate)
        logits = logits / self.config.temperature
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def forward(
        self, graph: SourceReuseGraph, *, seed: int | None = None
    ) -> PredictabilityScores:
        rng = random.Random(self.config.random_seed if seed is None else seed)
        birth, reuse = self.source_state.initialize(graph)
        use_count = torch.zeros(graph.num_tokens, dtype=torch.long, device=graph.device)
        cumulative_mass = torch.zeros(graph.num_tokens, device=graph.device)
        last_used = torch.full(
            (graph.num_tokens,), -1, dtype=torch.long, device=graph.device
        )

        rows: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "endpoint_nll",
                "shuffled_nll",
                "margin",
                "accuracy",
                "valid_pairs",
                "candidate_count",
                "positive_logit",
                "hardest_negative_logit",
                "mean_match_distance",
                "query_embedding",
                "source_embedding",
            )
        }

        for token in range(graph.num_response_tokens):
            sources, pair_embedding, pair_mass = self._pair_embeddings(
                graph, token, use_count
            )
            token_set = self.readout(pair_embedding, pair_mass)
            control = self._token_control(graph, token, pair_mass)
            token_embedding = self.query(torch.cat((token_set, control, token_set)))

            if sources.numel():
                pair_query = self.query(
                    torch.cat(
                        (
                            token_set.expand(pair_embedding.shape[0], -1),
                            control.expand(pair_embedding.shape[0], -1),
                            pair_embedding,
                        ),
                        dim=-1,
                    )
                )
                gap = torch.where(
                    last_used < 0,
                    torch.full_like(last_used, token + 1),
                    token - last_used,
                ).float()
                history_norm = torch.sqrt(
                    torch.log1p(use_count.float()).square()
                    + torch.log1p(cumulative_mass).square()
                    + torch.log1p(gap.clamp_min(0.0)).square()
                )
                candidates = matched_candidate_batch(
                    graph,
                    query=token,
                    true_sources=sources,
                    use_count=use_count,
                    cumulative_mass=cumulative_mass,
                    last_used=last_used,
                    memory_norm=history_norm,
                    config=self.config,
                    rng=rng,
                )
                candidate_source = candidates.candidate_source
                candidate_mask = candidates.candidate_mask
                safe_candidate = candidate_source.clamp_min(0)
                source_bin = self._source_bins(graph, token, safe_candidate)
                usage_bin = self._usage_bins(use_count, safe_candidate)
                candidate_embedding = self.source_state.candidate_embeddings(
                    graph=graph,
                    query=token,
                    candidate_source=candidate_source,
                    candidate_mask=candidate_mask,
                    use_count=use_count,
                    cumulative_mass=cumulative_mass,
                    last_used=last_used,
                    birth=birth,
                    reuse=reuse,
                    source_bin=source_bin,
                    usage_bucket=usage_bin,
                    shuffled=False,
                )
                shuffled_embedding = self.source_state.candidate_embeddings(
                    graph=graph,
                    query=token,
                    candidate_source=candidate_source,
                    candidate_mask=candidate_mask,
                    use_count=use_count,
                    cumulative_mass=cumulative_mass,
                    last_used=last_used,
                    birth=birth,
                    reuse=reuse,
                    source_bin=source_bin,
                    usage_bucket=usage_bin,
                    shuffled=True,
                )
                logits = self._score_candidates(
                    pair_query, candidate_embedding, candidate_mask
                )
                shuffled_logits = self._score_candidates(
                    pair_query, shuffled_embedding, candidate_mask
                )
                valid = candidates.valid
                if bool(valid.any()):
                    target = torch.zeros(
                        int(valid.sum()), dtype=torch.long, device=graph.device
                    )
                    nll = F.cross_entropy(logits[valid], target, reduction="none")
                    shuffled_nll = F.cross_entropy(
                        shuffled_logits[valid], target, reduction="none"
                    )
                    positive = logits[valid, 0]
                    hardest = logits[valid, 1:].max(dim=1).values
                    distance = candidates.candidate_distance[valid, 1:]
                    finite_distance = distance[torch.isfinite(distance)]
                    rows["endpoint_nll"].append(nll.mean())
                    rows["shuffled_nll"].append(shuffled_nll.mean())
                    rows["margin"].append((positive - hardest).mean())
                    rows["accuracy"].append(
                        (logits[valid].argmax(dim=1) == 0).float().mean()
                    )
                    rows["valid_pairs"].append(valid.sum().float())
                    rows["candidate_count"].append(
                        candidate_mask[valid].sum(dim=1).float().mean()
                    )
                    rows["positive_logit"].append(positive.mean())
                    rows["hardest_negative_logit"].append(hardest.mean())
                    rows["mean_match_distance"].append(
                        finite_distance.mean()
                        if finite_distance.numel()
                        else graph.weight.new_tensor(0.0)
                    )
                    rows["query_embedding"].append(pair_query[valid].mean(dim=0))
                    rows["source_embedding"].append(
                        candidate_embedding[valid, 0].mean(dim=0)
                    )
                else:
                    self._append_empty(rows, graph)
            else:
                self._append_empty(rows, graph)

            self.source_state.update_reuse(
                reuse,
                sources,
                pair_embedding,
                token_embedding,
                token=token,
                cumulative_mass=cumulative_mass,
                last_used=last_used,
            )
            if sources.numel():
                use_count[sources] += 1
                cumulative_mass.index_add_(0, sources, pair_mass.detach())
                last_used[sources] = token
            response_source = graph.response_idx + token
            birth[response_source] = self.source_state.seed_response(token_embedding)

            if (token + 1) % self.config.bptt_steps == 0:
                birth = [value.detach() for value in birth]
                reuse = [value.detach() for value in reuse]

        return PredictabilityScores(
            **{name: torch.stack(value) for name, value in rows.items()}
        )

    def _append_empty(
        self, rows: dict[str, list[torch.Tensor]], graph: SourceReuseGraph
    ) -> None:
        zero = graph.weight.new_tensor(0.0)
        for name in (
            "endpoint_nll",
            "shuffled_nll",
            "margin",
            "accuracy",
            "valid_pairs",
            "candidate_count",
            "positive_logit",
            "hardest_negative_logit",
            "mean_match_distance",
        ):
            rows[name].append(zero)
        empty = graph.weight.new_zeros(self.config.hidden_dim)
        rows["query_embedding"].append(empty)
        rows["source_embedding"].append(empty)

    @staticmethod
    def loss(output: PredictabilityScores) -> torch.Tensor:
        if not bool(output.valid.any()):
            return output.endpoint_nll.sum() * 0.0
        return output.endpoint_nll[output.valid].mean()

    @staticmethod
    def anomaly_score(output: PredictabilityScores) -> torch.Tensor:
        return output.endpoint_nll
