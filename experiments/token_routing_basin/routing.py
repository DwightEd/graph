"""Sparse, prefix-causal routing features for each generated token."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


FEATURE_NAMES = (
    "prompt_mass_share",
    "prompt_effective_source_fraction",
    "prompt_top1_share",
    "response_effective_source_fraction",
    "response_top1_share",
    "recent_response_share",
    "prompt_anchor_repeat",
    "prompt_anchor_run_fraction",
    "multiplex_route_effective_rank_fraction",
    "multiplex_route_dominant_mode_share",
    "multiplex_prompt_route_effective_rank_fraction",
    "multiplex_prompt_route_dominant_mode_share",
    "multiplex_response_route_effective_rank_fraction",
    "multiplex_response_route_dominant_mode_share",
    "relative_route_effective_rank_fraction",
    "relative_route_dominant_mode_share",
    "relative_route_velocity",
)

CONTROL_NAMES = (
    "log_token_index",
    "log_prompt_length",
    "log_excess_edge_count",
    "log_excess_attention_mass",
)


@dataclass(frozen=True)
class RoutingFeatureConfig:
    """Fixed geometry of the compact causal routing operator."""

    window: int = 8
    prompt_bins: int = 8
    lag_bins: int = 6
    recent_lag_max: int = 4
    anchor_run_cap: int = 8
    operator_sketch_width: int = 512
    block_rows: int = 8192
    epsilon: float = 1e-8

    def validate(self) -> None:
        for name in (
            "window",
            "prompt_bins",
            "lag_bins",
            "recent_lag_max",
            "anchor_run_cap",
            "operator_sketch_width",
            "block_rows",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class RoutingSequence:
    """One response's fixed-width token states and causal controls."""

    sample_id: str
    source_id: str
    task_type: str
    data_source: str
    names: tuple[str, ...]
    values: torch.Tensor
    control_names: tuple[str, ...]
    controls: torch.Tensor
    valid: torch.Tensor


class CausalRoutingFeatureExtractor:
    """Encode each observed prefix as source- and channel-preserving route states."""

    def __init__(self, config: RoutingFeatureConfig | None = None) -> None:
        self.config = config or RoutingFeatureConfig()
        self.config.validate()

    def extract(self, sample) -> RoutingSequence:
        attention = sample.attention()
        response_count = int(attention.num_response_tokens)
        prompt_count = int(attention.response_idx)
        layers = int(attention.num_layers)
        heads = int(attention.num_heads)
        floor = float(attention.attention_floor)
        if min(response_count, prompt_count, layers, heads) < 1:
            raise ValueError("routing features require non-empty attention geometry")
        if not math.isfinite(floor) or floor < 0:
            raise ValueError("attention_floor must be finite and non-negative")

        fields = {
            name: [] for name in ("layer", "head", "query", "source", "weight")
        }
        try:
            for block in sample.iter_sparse_attention_blocks(
                block_rows=self.config.block_rows
            ):
                layer = block.layer.long()
                head = block.head.long()
                query = block.query.long()
                source = block.source.long()
                weight = block.weight.float()
                self._validate_block(
                    layer,
                    head,
                    query,
                    source,
                    weight,
                    layers=layers,
                    heads=heads,
                    response_count=response_count,
                    prompt_count=prompt_count,
                )
                excess = (weight - floor).clamp_min(0)
                keep = excess > 0
                if not bool(keep.any()):
                    continue
                fields["layer"].append(layer[keep])
                fields["head"].append(head[keep])
                fields["query"].append(query[keep])
                fields["source"].append(source[keep])
                fields["weight"].append(excess[keep])
        finally:
            sample.release_attention()

        device = self._device(attention)
        if not fields["weight"]:
            return self._empty(sample, response_count, prompt_count, device)

        layer = torch.cat(fields["layer"])
        head = torch.cat(fields["head"])
        query = torch.cat(fields["query"])
        source = torch.cat(fields["source"])
        weight = torch.cat(fields["weight"])
        values, valid = self._source_features(
            query,
            source,
            weight,
            response_count=response_count,
            prompt_count=prompt_count,
        )
        spectrum = self._operator_spectra(
            layer,
            head,
            query,
            source,
            weight,
            response_count=response_count,
            prompt_count=prompt_count,
            heads=heads,
            valid=valid,
        )
        values = torch.cat((values, spectrum), dim=1)

        edge_count = torch.zeros(
            response_count, dtype=torch.float32, device=weight.device
        )
        edge_count.index_add_(0, query, torch.ones_like(weight))
        mass = torch.zeros_like(edge_count)
        mass.index_add_(0, query, weight)
        token_index = torch.arange(
            response_count, dtype=torch.float32, device=weight.device
        )
        controls = torch.stack(
            (
                torch.log1p(token_index),
                torch.full_like(token_index, math.log1p(prompt_count)),
                torch.log1p(edge_count),
                torch.log1p(mass),
            ),
            dim=1,
        )
        values = torch.where(valid[:, None], values, torch.zeros_like(values))
        return RoutingSequence(
            sample_id=str(sample.sample_id),
            source_id=str(sample.source_id),
            task_type=str(sample.task_type),
            data_source=str(sample.data_source),
            names=FEATURE_NAMES,
            values=values,
            control_names=CONTROL_NAMES,
            controls=controls,
            valid=valid,
        )

    @staticmethod
    def _device(attention) -> torch.device:
        for name in ("response_values", "attention_diagonal", "token_ids"):
            value = getattr(attention, name, None)
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _validate_block(
        layer,
        head,
        query,
        source,
        weight,
        *,
        layers,
        heads,
        response_count,
        prompt_count,
    ) -> None:
        count = len(weight)
        if any(len(value) != count for value in (layer, head, query, source)):
            raise ValueError("sparse attention block fields must align")
        if not count:
            return
        if bool((layer < 0).any()) or bool((layer >= layers).any()):
            raise ValueError("sparse attention layer is outside the manifest geometry")
        if bool((head < 0).any()) or bool((head >= heads).any()):
            raise ValueError("sparse attention head is outside the manifest geometry")
        if bool((query < 0).any()) or bool((query >= response_count).any()):
            raise ValueError("sparse attention query is outside the response")
        if bool((source < 0).any()):
            raise ValueError("sparse attention source must be non-negative")
        response = source >= prompt_count
        if bool(response.any()):
            relative = source[response] - prompt_count
            if bool((relative >= query[response]).any()):
                raise ValueError("response routes must point to an earlier token")
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError("sparse attention weights must be finite and non-negative")

    def _empty(self, sample, response_count, prompt_count, device) -> RoutingSequence:
        values = torch.zeros(
            (response_count, len(FEATURE_NAMES)), dtype=torch.float32, device=device
        )
        token_index = torch.arange(
            response_count, dtype=torch.float32, device=device
        )
        controls = torch.stack(
            (
                torch.log1p(token_index),
                torch.full_like(token_index, math.log1p(prompt_count)),
                torch.zeros_like(token_index),
                torch.zeros_like(token_index),
            ),
            dim=1,
        )
        return RoutingSequence(
            sample_id=str(sample.sample_id),
            source_id=str(sample.source_id),
            task_type=str(sample.task_type),
            data_source=str(sample.data_source),
            names=FEATURE_NAMES,
            values=values,
            control_names=CONTROL_NAMES,
            controls=controls,
            valid=torch.zeros(response_count, dtype=torch.bool, device=device),
        )

    def _source_features(
        self, query, source, weight, *, response_count, prompt_count
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = prompt_count + response_count
        route_id = query * token_count + source
        unique_route, inverse = torch.unique(
            route_id, sorted=True, return_inverse=True
        )
        route_weight = torch.zeros(
            len(unique_route), dtype=torch.float32, device=weight.device
        )
        route_weight.index_add_(0, inverse, weight)
        route_query = torch.div(unique_route, token_count, rounding_mode="floor")
        route_source = unique_route.remainder(token_count)

        prompt = route_source < prompt_count
        response = ~prompt
        prompt_total, prompt_count_visible, prompt_entropy_sum, prompt_max = (
            self._distribution_statistics(
                route_query[prompt], route_weight[prompt], response_count
            )
        )
        response_total, response_count_visible, response_entropy_sum, response_max = (
            self._distribution_statistics(
                route_query[response], route_weight[response], response_count
            )
        )
        total = prompt_total + response_total
        valid = total > self.config.epsilon

        prompt_effective = self._effective_fraction(
            prompt_total,
            prompt_entropy_sum,
            torch.full_like(prompt_total, float(prompt_count)),
        )
        available_response = torch.arange(
            response_count, dtype=torch.float32, device=weight.device
        )
        response_effective = self._effective_fraction(
            response_total, response_entropy_sum, available_response
        )
        prompt_top1 = self._safe_ratio(prompt_max, prompt_total)
        response_top1 = self._safe_ratio(response_max, response_total)
        prompt_share = self._safe_ratio(prompt_total, total)

        recent_mass = torch.zeros_like(response_total)
        if bool(response.any()):
            response_query = route_query[response]
            response_source = route_source[response] - prompt_count
            lag = response_query - response_source
            recent = (lag > 0) & (lag <= self.config.recent_lag_max)
            recent_mass.index_add_(
                0, response_query[recent], route_weight[response][recent]
            )
        recent_share = self._safe_ratio(recent_mass, response_total)
        anchor_repeat, anchor_run = self._anchor_features(
            route_query[prompt],
            route_source[prompt],
            route_weight[prompt],
            prompt_max,
            prompt_count_visible > 0,
            response_count,
            prompt_count,
        )
        return (
            torch.stack(
                (
                    prompt_share,
                    prompt_effective,
                    prompt_top1,
                    response_effective,
                    response_top1,
                    recent_share,
                    anchor_repeat,
                    anchor_run,
                ),
                dim=1,
            ),
            valid,
        )

    @staticmethod
    def _distribution_statistics(query, weight, rows):
        total = torch.zeros(rows, dtype=torch.float32, device=weight.device)
        count = torch.zeros_like(total)
        entropy_sum = torch.zeros_like(total)
        maximum = torch.zeros_like(total)
        if len(weight):
            total.index_add_(0, query, weight)
            count.index_add_(0, query, torch.ones_like(weight))
            entropy_sum.index_add_(0, query, weight * weight.log())
            maximum.scatter_reduce_(
                0, query, weight, reduce="amax", include_self=True
            )
        return total, count, entropy_sum, maximum

    def _effective_fraction(self, total, entropy_sum, available):
        entropy = total.clamp_min(self.config.epsilon).log() - (
            entropy_sum / total.clamp_min(self.config.epsilon)
        )
        fraction = entropy.exp() / available.clamp_min(1)
        return torch.where(
            (total > self.config.epsilon) & (available > 0),
            fraction.clamp(0, 1),
            torch.zeros_like(fraction),
        )

    def _safe_ratio(self, numerator, denominator):
        return torch.where(
            denominator > self.config.epsilon,
            numerator / denominator.clamp_min(self.config.epsilon),
            torch.zeros_like(numerator),
        )

    def _anchor_features(
        self,
        query,
        source,
        weight,
        prompt_max,
        has_prompt,
        response_count,
        prompt_count,
    ):
        sentinel = int(prompt_count)
        anchor = torch.full(
            (response_count,), sentinel, dtype=torch.long, device=weight.device
        )
        if len(weight):
            strongest = torch.isclose(
                weight, prompt_max[query], rtol=1e-6, atol=1e-8
            )
            anchor.scatter_reduce_(
                0,
                query[strongest],
                source[strongest],
                reduce="amin",
                include_self=True,
            )
        repeat = torch.zeros(
            response_count, dtype=torch.float32, device=weight.device
        )
        if response_count > 1:
            repeat[1:] = (
                has_prompt[1:]
                & has_prompt[:-1]
                & (anchor[1:] == anchor[:-1])
            ).float()
        run = torch.zeros_like(repeat)
        current = torch.zeros((), dtype=torch.float32, device=weight.device)
        for token in range(response_count):
            if not bool(has_prompt[token]):
                current = torch.zeros_like(current)
            elif token and bool(repeat[token]):
                current = current + 1
            else:
                current = torch.ones_like(current)
            run[token] = current / float(self.config.anchor_run_cap)
        return repeat, run.clamp(0, 1)

    def _operator_spectra(
        self,
        layer,
        head,
        query,
        source,
        weight,
        *,
        response_count,
        prompt_count,
        heads,
        valid,
    ):
        """Compute rolling singular spectra without materializing dense operators.

        The exact operator uses ``(layer, head, absolute_source)`` columns.  A
        second operator uses ``(layer, head, role, prompt_position/RR_lag)``
        columns, so source recurrence and relative routing remain separate.
        """

        channel = layer * heads + head
        pair_sum = channel + source
        exact_column = torch.div(
            pair_sum * (pair_sum + 1), 2, rounding_mode="floor"
        ) + source
        prompt = source < prompt_count

        exact_gram = self._sketched_operator_gram(
            query, exact_column, weight, response_count
        )
        prompt_gram = self._sketched_operator_gram(
            query[prompt], exact_column[prompt], weight[prompt], response_count
        )
        response_gram = self._sketched_operator_gram(
            query[~prompt], exact_column[~prompt], weight[~prompt], response_count
        )

        role_bins = self.config.prompt_bins + self.config.lag_bins
        local_bin = torch.empty_like(source)
        local_bin[prompt] = torch.div(
            source[prompt] * self.config.prompt_bins,
            prompt_count,
            rounding_mode="floor",
        ).clamp_max(self.config.prompt_bins - 1)
        if bool((~prompt).any()):
            response_source = source[~prompt] - prompt_count
            lag = query[~prompt] - response_source
            local_bin[~prompt] = self.config.prompt_bins + torch.floor(
                torch.log2(lag.float())
            ).long().clamp(0, self.config.lag_bins - 1)
        relative_column = channel * role_bins + local_bin
        relative_gram = self._sketched_operator_gram(
            query, relative_column, weight, response_count
        )

        exact_spectrum = self._rolling_gram_spectrum(exact_gram, valid)
        prompt_spectrum = self._rolling_gram_spectrum(prompt_gram, valid)
        response_spectrum = self._rolling_gram_spectrum(response_gram, valid)
        relative_spectrum = self._rolling_gram_spectrum(relative_gram, valid)
        velocity = torch.zeros(
            response_count, dtype=torch.float32, device=weight.device
        )
        if response_count > 1:
            previous_valid = relative_gram.diagonal()[:-1] > self.config.epsilon
            current_valid = relative_gram.diagonal()[1:] > self.config.epsilon
            comparable = previous_valid & current_valid
            adjacent_similarity = relative_gram.diagonal(offset=-1).clamp(0, 1)
            velocity[1:] = torch.where(
                comparable,
                1 - adjacent_similarity,
                torch.zeros_like(adjacent_similarity),
            )
        return torch.cat(
            (
                exact_spectrum,
                prompt_spectrum,
                response_spectrum,
                relative_spectrum,
                velocity[:, None],
            ),
            dim=1,
        )

    def _sketched_operator_gram(self, query, column, weight, rows):
        """Approximate a multiplex Gram matrix with a signed CountSketch.

        Columns represent exact channel/source identities before hashing.  The
        deterministic signed sketch approximates inner products while its
        memory does not grow with prompt length.
        """

        if not len(weight):
            return torch.zeros((rows, rows), dtype=torch.float32, device=weight.device)
        width = self.config.operator_sketch_width
        squared_norm = torch.zeros(rows, dtype=torch.float32, device=weight.device)
        squared_norm.index_add_(0, query, weight.square())
        normalized_weight = weight / squared_norm[query].sqrt().clamp_min(
            self.config.epsilon
        )
        hash_input = torch.remainder(column, 2_147_483_647)
        bucket = torch.remainder(
            torch.remainder(
                hash_input * 1_103_515_245 + 12_345,
                2_147_483_647,
            ),
            width,
        )
        sign = (
            1
            - 2
            * torch.remainder(
                torch.div(
                    hash_input * 2_654_435_761 + 2_246_822_519,
                    2_147_483_647,
                    rounding_mode="floor",
                ),
                2,
            )
        ).to(dtype=torch.float32)
        sketch = torch.zeros(
            (rows, width), dtype=torch.float32, device=weight.device
        )
        sketch.view(-1).index_add_(
            0,
            query * width + bucket,
            normalized_weight * sign,
        )
        gram = sketch @ sketch.transpose(0, 1)
        gram.diagonal().copy_((squared_norm > self.config.epsilon).float())
        return gram

    def _rolling_gram_spectrum(self, full_gram, valid):
        response_count = len(full_gram)
        window = self.config.window
        row_valid = valid & (full_gram.diagonal() > self.config.epsilon)
        grams = torch.zeros(
            (response_count, window, window),
            dtype=torch.float32,
            device=full_gram.device,
        )
        available_rank = torch.zeros(
            response_count, dtype=torch.float32, device=full_gram.device
        )
        consecutive = 0
        for token in range(response_count):
            if not bool(row_valid[token]):
                consecutive = 0
                continue
            consecutive = min(window, consecutive + 1)
            available = consecutive
            start = token + 1 - available
            grams[token, -available:, -available:] = full_gram[
                start : token + 1, start : token + 1
            ]
            available_rank[token] = available
        eigenvalues = torch.linalg.eigvalsh(grams).clamp_min(0)
        energy = eigenvalues.sum(dim=1)
        probability = eigenvalues / energy[:, None].clamp_min(self.config.epsilon)
        entropy = -torch.where(
            probability > 0,
            probability * probability.clamp_min(self.config.epsilon).log(),
            torch.zeros_like(probability),
        ).sum(dim=1)
        effective_rank = torch.where(
            energy > self.config.epsilon,
            entropy.exp() / available_rank.clamp_min(1),
            torch.zeros_like(energy),
        ).clamp(0, 1)
        dominant = self._safe_ratio(eigenvalues[:, -1], energy).clamp(0, 1)
        spectrum = torch.stack((effective_rank, dominant), dim=1)
        return torch.where(row_valid[:, None], spectrum, torch.zeros_like(spectrum))
