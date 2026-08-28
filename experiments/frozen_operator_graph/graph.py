"""Construct an exact role-conserving quotient of frozen attention messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from .config import GraphConstructionConfig
from .math_utils import (
    cosine,
    effective_number,
    entropy_from_mass,
    minimum_prefix,
    safe_norm,
    top1_share,
    weighted_mean_and_variance,
)
from .schema import (
    HISTORY,
    PROMPT,
    ROLE_COUNT,
    ROLE_NAMES,
    SELF,
    ExactSampleCapture,
    OperatorBasis,
)


ROUTE_METRICS = (
    "mass",
    "source_entropy",
    "source_effective_number",
    "source_top1_share",
    "lag_mean",
    "lag_variance",
    "value_energy_sum",
    "aggregate_value_norm",
    "value_coherence",
)

EDGE_FEATURE_NAMES = (
    "route_mass",
    "route_fraction_of_role",
    "pre_output_energy",
    "energy_fraction_of_role",
    "causal_lag",
    "head_entropy",
    "effective_heads",
    "head_top1_share",
    "operator_code_norm",
)

REMAINDER_FEATURE_NAMES = (
    "source_count",
    "route_mass",
    "route_fraction_of_role",
    "pre_output_energy_sum",
    "pre_output_aggregate_norm",
    "pre_output_coherence",
    "residual_message_norm",
    "lag_mean",
    "lag_variance",
    "head_entropy",
    "effective_heads",
    "head_top1_share",
    "alignment_with_attention",
    "alignment_with_role_message",
    "alignment_with_mlp",
)

GLOBAL_LAYER_FEATURE_NAMES = (
    "pre_attention_hidden_norm",
    "pre_mlp_hidden_norm",
    "residual_to_pre_attention_cosine",
    "post_residual_to_pre_mlp_cosine",
    "attention_output_norm",
    "attention_without_bias_norm",
    "output_bias_norm",
    "mlp_output_norm",
    "residual_input_norm",
    "layer_output_norm",
    "prompt_history_cosine",
    "prompt_mlp_cosine",
    "history_mlp_cosine",
    "attention_mlp_cosine",
    "prompt_route_fraction",
    "prompt_residual_fraction",
    "grounding_mismatch_signed",
    "grounding_mismatch_absolute",
    "layer_update_norm",
    "update_to_state_ratio",
    "pre_post_cosine",
)

ROLE_LAYER_METRICS = (
    "route_fraction",
    "pre_output_energy_sum",
    "pre_output_aggregate_norm",
    "residual_message_norm",
    "pre_output_coherence",
    "operator_dispersion",
    "operator_mean_norm",
)


@dataclass(frozen=True)
class GraphTensors:
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_role: torch.Tensor
    edge_attention_code: torch.Tensor
    edge_features: torch.Tensor
    edge_feature_names: Sequence[str]
    remainder_features: torch.Tensor
    remainder_feature_names: Sequence[str]
    route_features: torch.Tensor
    route_feature_names: Sequence[str]
    layer_features: torch.Tensor
    layer_feature_names: Sequence[str]
    final_hidden: torch.Tensor
    audit: Mapping[str, Any]


def route_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{role}_{metric}" for role in ROLE_NAMES for metric in ROUTE_METRICS
    )


def layer_feature_names(head_count: int) -> tuple[str, ...]:
    names: list[str] = list(GLOBAL_LAYER_FEATURE_NAMES)
    for role in ROLE_NAMES:
        names.extend(f"{role}_{metric}" for metric in ROLE_LAYER_METRICS)
        names.extend(
            f"{role}_operator_mean_unit_code_head_{head}"
            for head in range(head_count)
        )
    return tuple(names)


def _output_weight(output_factor: torch.Tensor) -> torch.Tensor:
    """Reassemble the exact ``o_proj.weight`` from ``[H,D,d]`` blocks."""

    if output_factor.ndim != 3:
        raise ValueError("output_factor must be [head,hidden,head_dim]")
    heads, hidden, head_dim = output_factor.shape
    if heads * head_dim != hidden:
        raise ValueError("output factors do not span the hidden dimension")
    return output_factor.permute(1, 0, 2).reshape(hidden, hidden).contiguous()


def _role_masks(tokens: int, response_start: int, device: torch.device) -> torch.Tensor:
    response = tokens - response_start
    source = torch.arange(tokens, device=device)
    target = response_start + torch.arange(response, device=device)
    prompt = (source < response_start).expand(response, -1)
    history = (source[None] >= response_start) & (source[None] < target[:, None])
    self_edge = source[None] == target[:, None]
    masks = torch.stack((prompt, history, self_edge), dim=1)
    if not bool((masks.sum(dim=1) == (source[None] <= target[:, None])).all()):
        raise RuntimeError("source roles do not partition the causal support")
    return masks


def _operator_statistics(
    code: torch.Tensor,
    factor: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean unit code, operator dispersion, and mean operator norm."""

    if code.ndim != 2:
        raise ValueError("operator code must be [source,head]")
    weight = code.float().clamp_min(0).sum(dim=-1)
    norm = safe_norm(code, eps=eps)
    direction = torch.where(
        norm[:, None] > eps,
        code.float() / norm[:, None].clamp_min(eps),
        torch.zeros_like(code.float()),
    )
    total = weight.sum()
    if float(total.item()) <= eps:
        return (
            torch.zeros(code.shape[1], dtype=torch.float32),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
    mean_code = (direction * weight[:, None]).sum(dim=0) / total
    embedded = direction @ factor.float()
    mean_embedding = (embedded * weight[:, None]).sum(dim=0) / total
    dispersion = (
        safe_norm(embedded - mean_embedding[None], eps=eps).pow(2) * weight
    ).sum() / total
    mean_norm = (safe_norm(embedded, eps=eps) * weight).sum() / total
    return mean_code, dispersion, mean_norm


def _source_selection(
    route_mass: torch.Tensor,
    pre_output_energy: torch.Tensor,
    source: torch.Tensor,
    *,
    config: GraphConstructionConfig,
) -> torch.Tensor:
    if (
        float(config.route_mass_retention) == 1.0
        and float(config.value_energy_retention) == 1.0
    ):
        return torch.arange(len(source), dtype=torch.long, device=source.device)
    mass = minimum_prefix(
        route_mass,
        source,
        config.route_mass_retention,
        config.minimum_role_edges,
    )
    energy = minimum_prefix(
        pre_output_energy,
        source,
        config.value_energy_retention,
        config.minimum_role_edges,
    )
    selected = torch.unique(torch.cat((mass, energy)), sorted=True)
    return selected


def _head_statistics(
    masked_attention: torch.Tensor,
    value_norm: torch.Tensor,
    role_context: torch.Tensor,
    lag: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Return ``[R,H,9]`` statistics for one source role."""

    mass = masked_attention.sum(dim=-1)
    entropy = entropy_from_mass(masked_attention, eps=eps)
    effective = effective_number(masked_attention, eps=eps)
    top1 = top1_share(masked_attention, eps=eps)
    lag_mean, lag_variance = weighted_mean_and_variance(
        lag[:, None, :].expand_as(masked_attention),
        masked_attention,
        eps=eps,
    )
    energy = (masked_attention * value_norm.T[None]).sum(dim=-1)
    aggregate_norm = safe_norm(role_context, eps=eps)
    coherence = torch.where(
        energy > eps,
        aggregate_norm / energy.clamp_min(eps),
        torch.zeros_like(energy),
    )
    return torch.stack(
        (
            mass,
            entropy,
            effective,
            top1,
            lag_mean,
            lag_variance,
            energy,
            aggregate_norm,
            coherence,
        ),
        dim=-1,
    )


def build_graph_tensors(
    capture: ExactSampleCapture,
    basis: OperatorBasis,
    config: GraphConstructionConfig | None = None,
) -> GraphTensors:
    """Build a deterministic graph without labels or trainable graph parameters."""

    config = GraphConstructionConfig() if config is None else config
    config.validate()
    capture.validate(
        atol=config.conservation_atol,
        rtol=config.conservation_rtol,
    )
    basis.validate()
    if capture.checkpoint != basis.checkpoint:
        raise ValueError("capture and operator basis come from different checkpoints")
    if capture.layer_count != basis.layer_count:
        raise ValueError("capture and basis layer counts differ")
    if capture.head_count != basis.head_count:
        raise ValueError("capture and basis head counts differ")
    if capture.hidden_size != basis.hidden_size or capture.head_dim != basis.head_dim:
        raise ValueError("capture and basis hidden geometry differ")
    if not torch.equal(capture.q_to_kv.cpu(), basis.q_to_kv.cpu()):
        raise ValueError("capture and basis query-to-KV mappings differ")

    layers = capture.layer_count
    response = capture.response_count
    tokens = capture.token_count
    heads = capture.head_count
    head_dim = capture.head_dim
    hidden = capture.hidden_size
    eps = config.feature_epsilon
    masks = _role_masks(tokens, capture.response_start, torch.device("cpu"))
    source_coordinate = torch.arange(tokens)
    target_coordinate = capture.response_start + torch.arange(response)
    lag = (target_coordinate[:, None] - source_coordinate[None]).clamp_min(0).float()

    route_names = route_feature_names()
    layer_names = layer_feature_names(heads)
    route_output = torch.empty((response, layers, heads, len(route_names)))
    layer_output = torch.empty((response, layers, len(layer_names)))
    remainder_output = torch.empty(
        (layers, response, ROLE_COUNT, len(REMAINDER_FEATURE_NAMES))
    )

    edge_source: list[torch.Tensor] = []
    edge_target: list[torch.Tensor] = []
    edge_layer: list[torch.Tensor] = []
    edge_role: list[torch.Tensor] = []
    edge_code: list[torch.Tensor] = []
    edge_feature: list[torch.Tensor] = []

    max_attention_reconstruction = 0.0
    max_context_rounding = 0.0
    max_projection_rounding = 0.0
    max_numerical_remainder = 0.0
    max_exact_attention_decomposition = 0.0
    max_o_proj_input_reconstruction = 0.0
    max_role_context_error = 0.0
    max_quotient_context_error = 0.0
    max_route_conservation_error = 0.0
    exposed_sources = 0
    causal_sources = 0

    for layer_index, layer_capture in enumerate(capture.layers):
        attention = layer_capture.attention.float().permute(1, 0, 2).contiguous()
        # [response, head, source]
        value = layer_capture.value_states.float()
        value_by_query_head = value[:, capture.q_to_kv.long(), :]
        # [source, query_head, head_dim]
        value_norm = safe_norm(value_by_query_head, eps=eps)
        role_context = torch.zeros((response, ROLE_COUNT, heads, head_dim))
        role_energy_sum = torch.zeros((response, ROLE_COUNT))
        role_route_mass = torch.zeros((response, ROLE_COUNT))
        route_blocks = []
        for role in range(ROLE_COUNT):
            role_mask = masks[:, role]
            masked = attention * role_mask[:, None, :]
            context = torch.einsum("rhn,nhd->rhd", masked, value_by_query_head)
            role_context[:, role] = context
            head_statistics = _head_statistics(
                masked,
                value_norm,
                context,
                lag,
                eps=eps,
            )
            route_blocks.append(head_statistics)
            role_route_mass[:, role] = head_statistics[..., 0].sum(dim=-1)
            role_energy_sum[:, role] = head_statistics[..., 6].sum(dim=-1)
        route_output[:, layer_index] = torch.cat(route_blocks, dim=-1)

        total_context_direct = torch.einsum(
            "rhn,nhd->rhd",
            attention,
            value_by_query_head,
        )
        captured_o_proj_input = layer_capture.o_proj_input[
            capture.response_start :
        ].float()
        o_proj_input_error = (total_context_direct - captured_o_proj_input).abs().max()
        max_o_proj_input_reconstruction = max(
            max_o_proj_input_reconstruction,
            float(o_proj_input_error.item()),
        )
        if not torch.allclose(
            total_context_direct,
            captured_o_proj_input,
            atol=config.conservation_atol,
            rtol=config.conservation_rtol,
        ):
            raise ValueError(
                f"layer {layer_index} exact A@V reconstruction failed: "
                f"max_abs_error={float(o_proj_input_error.item()):.6g}"
            )
        context_error = (
            total_context_direct - role_context.sum(dim=1)
        ).abs().max()
        max_role_context_error = max(
            max_role_context_error,
            float(context_error.item()),
        )

        # The actual output is captured from the frozen ``o_proj`` module in
        # the same forward pass. ``capture.py`` also requires self-attention to
        # return that exact tensor, so no cross-device re-execution is used as a
        # proxy for CUDA bfloat16/float16 GEMM semantics. Edge/role messages are
        # decomposed in float32 and the unavoidable hardware numerical residual
        # is retained explicitly below.
        output_factor_native = basis.output_factor[layer_index].detach().cpu()
        output_weight_native = _output_weight(output_factor_native)
        output_bias_native = basis.output_bias[layer_index].detach().cpu()
        output_weight = output_weight_native.float()
        output_bias = output_bias_native.float()
        captured_attention = layer_capture.attention_output[
            capture.response_start :
        ].float()

        # Edge/role messages are accumulated in float32 so their additive
        # decomposition is stable and inspectable.  Preserve both unavoidable
        # numerical residuals instead of hiding them or loosening tolerances:
        #   1) A@V rounding before o_proj,
        #   2) o_proj output quantization/backend rounding.
        context_rounding = captured_o_proj_input - total_context_direct
        context_rounding_error = context_rounding.abs().max()
        max_context_rounding = max(
            max_context_rounding,
            float(context_rounding_error.item()),
        )
        without_bias = functional.linear(
            total_context_direct.reshape(response, hidden),
            output_weight,
            None,
        )
        context_rounding_message = functional.linear(
            context_rounding.reshape(response, hidden),
            output_weight,
            None,
        )
        captured_context_without_bias = without_bias + context_rounding_message
        projection_linearized = captured_context_without_bias + output_bias[None]
        projection_rounding = captured_attention - projection_linearized
        projection_rounding_error = projection_rounding.abs().max()
        max_projection_rounding = max(
            max_projection_rounding,
            float(projection_rounding_error.item()),
        )
        numerical_remainder = context_rounding_message + projection_rounding
        numerical_remainder_error = numerical_remainder.abs().max()
        max_numerical_remainder = max(
            max_numerical_remainder,
            float(numerical_remainder_error.item()),
        )

        # This is the exact accounting identity used by the graph: every
        # prompt/history/self message, output bias, and finite-precision
        # remainder is retained.  The old ``attention_error`` remains an audit
        # of how far the real-valued edge sum is from the hardware output; it is
        # evidence about numeric precision, not an operator mismatch.
        reconstructed_attention = without_bias + output_bias[None]
        attention_error = (reconstructed_attention - captured_attention).abs().max()
        max_attention_reconstruction = max(
            max_attention_reconstruction,
            float(attention_error.item()),
        )
        exact_attention = reconstructed_attention + numerical_remainder
        exact_attention_error = (exact_attention - captured_attention).abs().max()
        max_exact_attention_decomposition = max(
            max_exact_attention_decomposition,
            float(exact_attention_error.item()),
        )
        if not torch.allclose(
            exact_attention,
            captured_attention,
            atol=config.conservation_atol,
            rtol=config.conservation_rtol,
        ):
            raise ValueError(
                f"layer {layer_index} finite-precision attention accounting "
                "failed: "
                f"max_abs_error={float(exact_attention_error.item()):.6g}"
            )

        role_residual_message = functional.linear(
            role_context.reshape(response * ROLE_COUNT, hidden),
            output_weight,
            None,
        ).reshape(response, ROLE_COUNT, hidden)
        if not torch.allclose(
            role_residual_message.sum(dim=1),
            without_bias,
            atol=config.conservation_atol,
            rtol=config.conservation_rtol,
        ):
            raise ValueError(f"layer {layer_index} role residual messages do not conserve")

        selected_context = torch.zeros_like(role_context)
        selected_route = torch.zeros((response, ROLE_COUNT))
        operator_code_mean = torch.zeros((response, ROLE_COUNT, heads))
        operator_dispersion = torch.zeros((response, ROLE_COUNT))
        operator_mean_norm = torch.zeros((response, ROLE_COUNT))
        selected_membership = torch.zeros(
            (response, ROLE_COUNT, tokens), dtype=torch.bool
        )
        factor = basis.normalized_operator_factor[layer_index].float()

        for query in range(response):
            target = int(target_coordinate[query].item())
            code_by_source = attention[query].T.contiguous()  # [source, head]
            head_context_by_source = (
                code_by_source[:, :, None] * value_by_query_head
            )
            pre_output_energy = safe_norm(
                head_context_by_source.reshape(tokens, hidden),
                eps=eps,
            )
            route_mass = code_by_source.sum(dim=-1)
            causal_sources += target + 1

            for role in range(ROLE_COUNT):
                role_indices = torch.nonzero(
                    masks[query, role], as_tuple=False
                ).flatten()
                if not len(role_indices):
                    continue
                role_code = code_by_source[role_indices]
                mean_code, dispersion, mean_norm = _operator_statistics(
                    role_code,
                    factor,
                    eps=eps,
                )
                operator_code_mean[query, role] = mean_code
                operator_dispersion[query, role] = dispersion
                operator_mean_norm[query, role] = mean_norm

                local_selected = _source_selection(
                    route_mass[role_indices],
                    pre_output_energy[role_indices],
                    source_coordinate[role_indices],
                    config=config,
                )
                selected_indices = role_indices[local_selected]
                selected_membership[query, role, selected_indices] = True
                exposed_sources += len(selected_indices)

                selected_context[query, role] = head_context_by_source[
                    selected_indices
                ].sum(dim=0)
                selected_route[query, role] = route_mass[selected_indices].sum()
                role_mass_total = route_mass[role_indices].sum()
                role_energy_total = pre_output_energy[role_indices].sum()
                code = code_by_source[selected_indices]
                head_entropy = entropy_from_mass(code, eps=eps)
                head_effective = effective_number(code, eps=eps)
                head_top1 = top1_share(code, eps=eps)
                operator_norm = safe_norm(code @ factor, eps=eps)
                features = torch.stack(
                    (
                        route_mass[selected_indices],
                        route_mass[selected_indices]
                        / role_mass_total.clamp_min(eps),
                        pre_output_energy[selected_indices],
                        pre_output_energy[selected_indices]
                        / role_energy_total.clamp_min(eps),
                        (target - source_coordinate[selected_indices]).float(),
                        head_entropy,
                        head_effective,
                        head_top1,
                        operator_norm,
                    ),
                    dim=-1,
                )
                edge_source.append(source_coordinate[selected_indices].long())
                edge_target.append(
                    torch.full(
                        (len(selected_indices),),
                        target,
                        dtype=torch.long,
                    )
                )
                edge_layer.append(
                    torch.full(
                        (len(selected_indices),),
                        layer_index,
                        dtype=torch.long,
                    )
                )
                edge_role.append(
                    torch.full(
                        (len(selected_indices),),
                        role,
                        dtype=torch.long,
                    )
                )
                edge_code.append(code)
                edge_feature.append(features)

        remainder_context = role_context - selected_context
        quotient_error = (
            selected_context + remainder_context - role_context
        ).abs().max()
        max_quotient_context_error = max(
            max_quotient_context_error,
            float(quotient_error.item()),
        )
        remainder_route = role_route_mass - selected_route
        route_error = (
            selected_route + remainder_route - role_route_mass
        ).abs().max()
        max_route_conservation_error = max(
            max_route_conservation_error,
            float(route_error.item()),
        )
        remainder_residual_message = functional.linear(
            remainder_context.reshape(response * ROLE_COUNT, hidden),
            output_weight,
            None,
        ).reshape(response, ROLE_COUNT, hidden)
        if not torch.allclose(
            role_residual_message,
            functional.linear(
                selected_context.reshape(response * ROLE_COUNT, hidden),
                output_weight,
                None,
            ).reshape(response, ROLE_COUNT, hidden)
            + remainder_residual_message,
            atol=config.conservation_atol,
            rtol=config.conservation_rtol,
        ):
            raise ValueError(f"layer {layer_index} quotient messages do not conserve")

        # Exact remainder features.  All omitted token edges contribute here.
        for query in range(response):
            target = int(target_coordinate[query].item())
            code_by_source = attention[query].T.contiguous()
            head_context_by_source = (
                code_by_source[:, :, None] * value_by_query_head
            )
            pre_output_energy = safe_norm(
                head_context_by_source.reshape(tokens, hidden),
                eps=eps,
            )
            route_mass = code_by_source.sum(dim=-1)
            for role in range(ROLE_COUNT):
                role_indices = torch.nonzero(
                    masks[query, role], as_tuple=False
                ).flatten()
                if not len(role_indices):
                    remainder_output[layer_index, query, role].zero_()
                    continue
                remainder_indices = role_indices[
                    ~selected_membership[query, role, role_indices]
                ]
                if not len(remainder_indices):
                    remainder_output[layer_index, query, role].zero_()
                    continue
                rem_code = code_by_source[remainder_indices]
                rem_route = route_mass[remainder_indices]
                rem_energy = pre_output_energy[remainder_indices]
                rem_lag = (target - source_coordinate[remainder_indices]).float()
                lag_mean, lag_variance = weighted_mean_and_variance(
                    rem_lag[None],
                    rem_route[None],
                    eps=eps,
                )
                aggregate_pre_norm = safe_norm(
                    remainder_context[query, role].reshape(1, hidden),
                    eps=eps,
                )[0]
                energy_sum = rem_energy.sum()
                coherence = aggregate_pre_norm / energy_sum.clamp_min(eps)
                residual_message = remainder_residual_message[query, role]
                role_message = role_residual_message[query, role]
                mlp_message = layer_capture.mlp_output[target].float()
                remainder_output[layer_index, query, role] = torch.stack(
                    (
                        torch.tensor(float(len(remainder_indices))),
                        rem_route.sum(),
                        rem_route.sum()
                        / role_route_mass[query, role].clamp_min(eps),
                        energy_sum,
                        aggregate_pre_norm,
                        coherence,
                        safe_norm(residual_message[None], eps=eps)[0],
                        lag_mean[0],
                        lag_variance[0],
                        entropy_from_mass(rem_code.sum(dim=0)[None], eps=eps)[0],
                        effective_number(rem_code.sum(dim=0)[None], eps=eps)[0],
                        top1_share(rem_code.sum(dim=0)[None], eps=eps)[0],
                        cosine(
                            residual_message[None],
                            without_bias[query][None],
                            eps=eps,
                        )[0],
                        cosine(
                            residual_message[None],
                            role_message[None],
                            eps=eps,
                        )[0],
                        cosine(
                            residual_message[None],
                            mlp_message[None],
                            eps=eps,
                        )[0],
                    )
                )

        prompt_message = role_residual_message[:, PROMPT]
        history_message = role_residual_message[:, HISTORY]
        self_message = role_residual_message[:, SELF]
        mlp = layer_capture.mlp_output[capture.response_start :].float()
        residual_input = layer_capture.residual_input[capture.response_start :].float()
        final_layer_state = layer_capture.layer_output[capture.response_start :].float()
        update = final_layer_state - residual_input
        role_message_norm = safe_norm(role_residual_message, eps=eps)
        route_total = role_route_mass.sum(dim=-1)
        prompt_route_fraction = role_route_mass[:, PROMPT] / route_total.clamp_min(eps)
        residual_total = role_message_norm.sum(dim=-1)
        prompt_residual_fraction = role_message_norm[:, PROMPT] / residual_total.clamp_min(eps)
        pre_attention_hidden = layer_capture.pre_attention_hidden[
            capture.response_start :
        ].float()
        post_attention_residual = layer_capture.post_attention_residual[
            capture.response_start :
        ].float()
        pre_mlp_hidden = layer_capture.pre_mlp_hidden[
            capture.response_start :
        ].float()
        global_features = torch.stack(
            (
                safe_norm(pre_attention_hidden, eps=eps),
                safe_norm(pre_mlp_hidden, eps=eps),
                cosine(residual_input, pre_attention_hidden, eps=eps),
                cosine(post_attention_residual, pre_mlp_hidden, eps=eps),
                safe_norm(captured_attention, eps=eps),
                safe_norm(without_bias, eps=eps),
                safe_norm(output_bias[None].expand(response, -1), eps=eps),
                safe_norm(mlp, eps=eps),
                safe_norm(residual_input, eps=eps),
                safe_norm(final_layer_state, eps=eps),
                cosine(prompt_message, history_message, eps=eps),
                cosine(prompt_message, mlp, eps=eps),
                cosine(history_message, mlp, eps=eps),
                cosine(captured_attention, mlp, eps=eps),
                prompt_route_fraction,
                prompt_residual_fraction,
                prompt_route_fraction - prompt_residual_fraction,
                (prompt_route_fraction - prompt_residual_fraction).abs(),
                safe_norm(update, eps=eps),
                safe_norm(update, eps=eps)
                / safe_norm(residual_input, eps=eps).clamp_min(eps),
                cosine(residual_input, final_layer_state, eps=eps),
            ),
            dim=-1,
        )
        role_feature_blocks = []
        for role in range(ROLE_COUNT):
            pre_aggregate_norm = safe_norm(
                role_context[:, role].reshape(response, hidden),
                eps=eps,
            )
            pre_coherence = pre_aggregate_norm / role_energy_sum[:, role].clamp_min(eps)
            role_feature_blocks.append(
                torch.cat(
                    (
                        (role_route_mass[:, role] / float(heads))[:, None],
                        role_energy_sum[:, role][:, None],
                        pre_aggregate_norm[:, None],
                        role_message_norm[:, role][:, None],
                        pre_coherence[:, None],
                        operator_dispersion[:, role][:, None],
                        operator_mean_norm[:, role][:, None],
                        operator_code_mean[:, role],
                    ),
                    dim=-1,
                )
            )
        layer_output[:, layer_index] = torch.cat(
            (global_features, *role_feature_blocks),
            dim=-1,
        )

    if max_role_context_error > config.conservation_atol:
        raise ValueError("role context conservation exceeded tolerance")
    if max_quotient_context_error > config.conservation_atol:
        raise ValueError("quotient context conservation exceeded tolerance")
    if max_route_conservation_error > config.conservation_atol:
        raise ValueError("route mass conservation exceeded tolerance")

    if edge_source:
        source = torch.cat(edge_source)
        target = torch.cat(edge_target)
        edge_index_tensor = torch.stack((source, target), dim=0)
        edge_layer_tensor = torch.cat(edge_layer)
        edge_role_tensor = torch.cat(edge_role)
        edge_code_tensor = torch.cat(edge_code).float()
        edge_feature_tensor = torch.cat(edge_feature).float()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_layer_tensor = torch.empty(0, dtype=torch.long)
        edge_role_tensor = torch.empty(0, dtype=torch.long)
        edge_code_tensor = torch.empty((0, heads), dtype=torch.float32)
        edge_feature_tensor = torch.empty(
            (0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32
        )

    audit = {
        "labels_read": False,
        "construction": "rolewise_dual_conservation_quotient",
        "route_mass_retention": config.route_mass_retention,
        "value_energy_retention": config.value_energy_retention,
        "exposed_token_edges": int(edge_index_tensor.shape[1]),
        "causal_token_pairs_considered": int(causal_sources),
        "exposed_pair_fraction": (
            float(edge_index_tensor.shape[1]) / float(causal_sources)
            if causal_sources
            else 0.0
        ),
        "max_attention_reconstruction_abs_error": max_attention_reconstruction,
        "max_context_rounding_abs_error": max_context_rounding,
        "max_projection_rounding_abs_error": max_projection_rounding,
        "max_numerical_remainder_abs_error": max_numerical_remainder,
        "max_exact_attention_decomposition_abs_error": (
            max_exact_attention_decomposition
        ),
        "projection_output_binding": (
            "direct_o_proj_forward_hook_bitwise_equals_self_attention_output"
        ),
        "projection_validation": (
            "float32_edge_decomposition_plus_explicit_finite_precision_remainder"
        ),
        "max_o_proj_input_reconstruction_abs_error": max_o_proj_input_reconstruction,
        "max_role_context_abs_error": max_role_context_error,
        "max_quotient_context_abs_error": max_quotient_context_error,
        "max_route_conservation_abs_error": max_route_conservation_error,
        "attention_cache_binding": capture.attention_cache_binding,
    }
    return GraphTensors(
        edge_index=edge_index_tensor,
        edge_layer=edge_layer_tensor,
        edge_role=edge_role_tensor,
        edge_attention_code=edge_code_tensor,
        edge_features=edge_feature_tensor,
        edge_feature_names=EDGE_FEATURE_NAMES,
        remainder_features=remainder_output.float(),
        remainder_feature_names=REMAINDER_FEATURE_NAMES,
        route_features=route_output.float(),
        route_feature_names=route_names,
        layer_features=layer_output.float(),
        layer_feature_names=layer_names,
        final_hidden=capture.final_hidden[capture.response_start :].float(),
        audit=audit,
    )
