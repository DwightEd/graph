"""Input-dependent functional contribution on the cached sparse attention graph.

The sparse cache contains retained off-diagonal attention endpoints, an exact
diagonal, and unresolved attention mass.  A replay of the frozen language
model supplies the actual value vectors and the chosen-answer gradient at the
input of every ``o_proj``.  For an observed edge ``q <- j`` we compute

    phi[l, q, j, h] = A[l, q, j, h] * <dJ/dc[l, q, h], v[l, j, kv(h)]>.

Only observed endpoints receive a functional attribution.  Unresolved mass is
reported through attention coverage and is never silently converted to a zero
functional contribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from experiments.grounded_route.graph import TokenGraph


PROMPT_ROLE_NAMES = ("evidence", "question", "constraint", "other_prompt")
HISTORY_ROLE = len(PROMPT_ROLE_NAMES)
FUNCTIONAL_ROLE_NAMES = (*PROMPT_ROLE_NAMES, "history")
EPSILON = 1e-12


def _field(value: Any, *names: str) -> Any:
    """Read the first named field from a dataclass-like object or mapping."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    joined = ", ".join(names)
    raise AttributeError(f"capture does not provide any of: {joined}")


def _optional_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def prompt_role_array(
    prompt_roles: Any,
    prompt_length: int,
    cached_token_ids: Any | None = None,
) -> np.ndarray:
    """Return canonical prompt role ids from an array or ``PromptRoleMap``.

    The four role ids are fixed by the audit schema.  A missing or out-of-range
    role is an input error rather than an invitation to collapse back to an
    ambiguous prompt bucket.
    """

    if cached_token_ids is not None and hasattr(prompt_roles, "validate"):
        prompt_roles.validate(np.asarray(cached_token_ids, dtype=np.int64))
    if isinstance(prompt_roles, Mapping):
        value = prompt_roles.get("role_ids", prompt_roles.get("token_roles"))
    else:
        value = getattr(
            prompt_roles,
            "role_ids",
            getattr(prompt_roles, "token_roles", prompt_roles),
        )
    roles = np.asarray(value, dtype=np.int64)
    if roles.shape != (int(prompt_length),):
        raise ValueError(
            "prompt role ids must have shape "
            f"({int(prompt_length)},), received {roles.shape}"
        )
    if roles.size and ((roles < 0).any() or (roles >= len(PROMPT_ROLE_NAMES)).any()):
        raise ValueError("prompt role ids are outside the canonical four-role schema")
    return roles


def _capture_tensors(
    graph: TokenGraph,
    capture: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    """Validate and return value, gradient, q-to-kv and predictor tensors."""

    values = _as_tensor(_field(capture, "value_states"))
    gradients = _as_tensor(
        _field(capture, "o_proj_input_gradients", "context_gradients")
    )
    probe_value = _optional_field(capture, "o_proj_input_gradient_probes")
    gradient_probes = None if probe_value is None else _as_tensor(probe_value)
    q_to_kv = _as_tensor(_field(capture, "q_to_kv"), dtype=torch.long)

    if values.ndim == 3:
        kv_heads = int(_field(capture, "kv_head_count", "kv_heads"))
        head_dim = int(_field(capture, "head_dim"))
        values = values.reshape(values.shape[0], values.shape[1], kv_heads, head_dim)
    if gradients.ndim == 3:
        q_heads = int(_field(capture, "head_count", "q_head_count", "q_heads"))
        head_dim = int(_field(capture, "head_dim"))
        gradients = gradients.reshape(
            gradients.shape[0], gradients.shape[1], q_heads, head_dim
        )

    if values.ndim != 4 or gradients.ndim != 4:
        raise ValueError("value states and o_proj gradients must be rank-four tensors")
    if values.shape[0] != graph.layer_count or gradients.shape[0] != graph.layer_count:
        raise ValueError("capture and graph disagree on the number of layers")
    if values.shape[1] != graph.token_count:
        raise ValueError("value states must cover the complete prompt-response sequence")
    if gradients.shape[2] != graph.head_count or q_to_kv.shape != (graph.head_count,):
        raise ValueError("capture and graph disagree on query heads")
    if values.shape[3] != gradients.shape[3]:
        raise ValueError("value states and gradients disagree on head dimension")
    if q_to_kv.numel() and (
        int(q_to_kv.min()) < 0 or int(q_to_kv.max()) >= values.shape[2]
    ):
        raise ValueError("q_to_kv contains an invalid key/value head")

    token_ids = _as_tensor(_field(capture, "token_ids"), dtype=torch.long).cpu()
    if token_ids.shape != graph.token_ids.shape or not torch.equal(
        token_ids, graph.token_ids.detach().cpu().long()
    ):
        raise ValueError("functional replay token ids do not match the cached graph")

    default_predictors = torch.cat(
        (
            torch.tensor([graph.response_start - 1], dtype=torch.long),
            torch.arange(
                graph.response_start,
                graph.token_count - 1,
                dtype=torch.long,
            ),
        )
    )
    predictors = _as_tensor(
        _field(capture, "predictor_indices"), dtype=torch.long
    ).cpu()
    if predictors.shape != (graph.response_count,) or not torch.equal(
        predictors, default_predictors
    ):
        raise ValueError(
            "predictor alignment must be [last prompt, response[:-1]]"
        )

    target_ids = _as_tensor(_field(capture, "target_ids"), dtype=torch.long).cpu()
    if target_ids.shape != (graph.response_count,) or not torch.equal(
        target_ids, graph.response_token_ids.detach().cpu().long()
    ):
        raise ValueError("functional replay targets do not match response token ids")

    # New captures retain gradients only for predictor rows.  A full-sequence
    # tensor is also accepted for backwards-compatible synthetic audits.
    if gradients.shape[1] == graph.response_count:
        aligned_gradients = gradients
    elif gradients.shape[1] == graph.token_count:
        aligned_gradients = gradients.index_select(
            1, predictors.to(gradients.device)
        )
    else:
        raise ValueError(
            "o_proj gradients must index response predictors or the full sequence"
        )
    aligned_probe_gradients = None
    if gradient_probes is not None:
        if gradient_probes.ndim != 5 or gradient_probes.shape[1:] != (
            graph.layer_count,
            graph.response_count,
            graph.head_count,
            values.shape[3],
        ):
            raise ValueError(
                "gradient probes must have shape [probe, layer, response, head, dim]"
            )
        if gradient_probes.shape[0] < 1:
            raise ValueError("functional capture contains no gradient probes")
        if not torch.allclose(
            gradient_probes.mean(dim=0),
            aligned_gradients,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError("stored mean gradient differs from signed probe mean")
        aligned_probe_gradients = gradient_probes
    return values, aligned_gradients, aligned_probe_gradients, q_to_kv, predictors


def _entropy_and_hhi(
    energy_sum: torch.Tensor,
    energy_log_sum: torch.Tensor,
    energy_square_sum: torch.Tensor,
    endpoint_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute observed-only normalized entropy and HHI from sufficient stats."""

    positive = energy_sum > 0
    entropy = torch.full_like(energy_sum, torch.nan)
    hhi = torch.full_like(energy_sum, torch.nan)
    safe = energy_sum.clamp_min(EPSILON)
    raw_entropy = safe.log() - energy_log_sum / safe
    normalizer = endpoint_count.to(energy_sum.dtype).clamp_min(1).log()
    normalized = torch.where(
        endpoint_count > 1,
        raw_entropy / normalizer.clamp_min(EPSILON),
        torch.zeros_like(raw_entropy),
    )
    entropy[positive] = normalized[positive].clamp(0.0, 1.0)
    hhi[positive] = (energy_square_sum[positive] / safe[positive].square()).clamp(
        0.0, 1.0
    )
    return entropy, hhi


def _head_role_js(role_energy: torch.Tensor) -> torch.Tensor:
    """Generalized JSD among heads over functional source roles.

    ``role_energy`` has shape ``[..., H, K]``.  Heads with no observed
    functional energy are unavailable and excluded, rather than represented
    by an artificial all-zero categorical distribution.
    """

    shape = role_energy.shape[:-2]
    heads = role_energy.shape[-2]
    roles = role_energy.shape[-1]
    flat = role_energy.reshape(-1, heads, roles)
    result = torch.full(
        (flat.shape[0],),
        torch.nan,
        dtype=role_energy.dtype,
        device=role_energy.device,
    )
    for index, row in enumerate(flat):
        totals = row.sum(dim=-1)
        valid = totals > 0
        count = int(valid.sum())
        if count == 0:
            continue
        if count == 1:
            result[index] = 0.0
            continue
        distributions = row[valid] / totals[valid, None]
        mixture = distributions.mean(dim=0)
        mixture_entropy = -torch.xlogy(mixture, mixture).sum()
        head_entropy = -torch.xlogy(distributions, distributions).sum(dim=-1).mean()
        maximum = row.new_tensor(float(min(count, roles))).log()
        result[index] = ((mixture_entropy - head_entropy) / maximum).clamp(0.0, 1.0)
    return result.reshape(shape)


@torch.no_grad()
def functional_flow(
    graph: TokenGraph,
    prompt_roles: Any,
    capture: Any,
) -> dict[str, np.ndarray]:
    """Compute predecessor-aligned functional-flow traces for one answer.

    Every returned response trajectory has ``R`` rows.  Row zero is NaN and
    marked unavailable because the sparse cache does not contain the
    last-prompt query which predicts the first answer token.  Cached query
    ``i`` is written to response-token row ``i + 1``; the final cached query is
    intentionally unused because it predicts beyond the saved answer.
    """

    graph = graph.canonicalize().check()
    roles = prompt_role_array(
        prompt_roles,
        graph.response_start,
        graph.token_ids.detach().cpu().numpy(),
    )
    values, gradients, gradient_probes, q_to_kv, predictors = _capture_tensors(
        graph, capture
    )
    device = gradients.device
    dtype = torch.promote_types(values.dtype, gradients.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    values = values.to(device=device, dtype=dtype)
    gradients = gradients.to(device=device, dtype=dtype)
    if gradient_probes is not None:
        gradient_probes = gradient_probes.to(device=device, dtype=dtype)
    q_to_kv = q_to_kv.to(device)
    predictors_device = predictors.to(device)

    response_count = graph.response_count
    layers = graph.layer_count
    heads = graph.head_count
    role_count = len(FUNCTIONAL_ROLE_NAMES)
    shape = (response_count, layers, heads)

    signed_role = torch.zeros((*shape, role_count), dtype=dtype, device=device)
    absolute_role = torch.zeros_like(signed_role)
    energy_sum = torch.zeros(shape, dtype=dtype, device=device)
    energy_log_sum = torch.zeros_like(energy_sum)
    energy_square_sum = torch.zeros_like(energy_sum)
    endpoint_count = torch.zeros(shape, dtype=torch.long, device=device)
    known_attention = torch.zeros(shape, dtype=dtype, device=device)

    role_tensor = torch.as_tensor(roles, dtype=torch.long, device=device)
    # Only query rows 0..R-2 predict a token stored in this answer.
    for layer in range(layers):
        edges = graph.layer_edges(layer, device)
        if edges.count:
            query = edges.target - graph.response_start
            keep = query < response_count - 1
            edges = edges.select(keep)
            query = query[keep]
            if edges.count:
                token_row = query + 1
                head = edges.head.long()
                source = edges.source.long()
                # Gradient row ``token_row`` is explicitly aligned to the
                # predictor which generated that response token.
                grad = gradients[layer, token_row, head]
                value = values[layer, source, q_to_kv[head]]
                phi = edges.weight.to(dtype=dtype) * (grad * value).sum(dim=-1)
                energy = phi.abs()
                source_role = torch.full_like(source, HISTORY_ROLE)
                prompt = source < graph.response_start
                source_role[prompt] = role_tensor[source[prompt]]
                index = (token_row, torch.full_like(token_row, layer), head, source_role)
                signed_role.index_put_(index, phi, accumulate=True)
                absolute_role.index_put_(index, energy, accumulate=True)
                row = (token_row, torch.full_like(token_row, layer), head)
                energy_sum.index_put_(row, energy, accumulate=True)
                energy_log_sum.index_put_(
                    row,
                    torch.xlogy(energy, energy.clamp_min(EPSILON)),
                    accumulate=True,
                )
                energy_square_sum.index_put_(row, energy.square(), accumulate=True)
                endpoint_count.index_put_(
                    row, torch.ones_like(token_row, dtype=torch.long), accumulate=True
                )
                known_attention.index_put_(
                    row, edges.weight.to(dtype=dtype), accumulate=True
                )

        if response_count > 1:
            query = torch.arange(response_count - 1, device=device)
            token_row = query + 1
            head = torch.arange(heads, device=device)[None, :].expand(
                response_count - 1, heads
            )
            token_grid = token_row[:, None].expand_as(head)
            query_absolute = predictors_device[token_row][:, None].expand_as(head)
            grad = gradients[layer, token_grid, head]
            value = values[layer, query_absolute, q_to_kv[head]]
            diagonal_weight = graph.diagonal[: response_count - 1, layer].to(
                device=device, dtype=dtype
            )
            phi = diagonal_weight * (grad * value).sum(dim=-1)
            energy = phi.abs()
            layer_grid = torch.full_like(head, layer)
            role_grid = torch.full_like(head, HISTORY_ROLE)
            index = (token_grid, layer_grid, head, role_grid)
            signed_role.index_put_(index, phi, accumulate=True)
            absolute_role.index_put_(index, energy, accumulate=True)
            row = (token_grid, layer_grid, head)
            energy_sum.index_put_(row, energy, accumulate=True)
            energy_log_sum.index_put_(
                row,
                torch.xlogy(energy, energy.clamp_min(EPSILON)),
                accumulate=True,
            )
            energy_square_sum.index_put_(row, energy.square(), accumulate=True)
            endpoint_count.index_put_(
                row, torch.ones_like(token_grid, dtype=torch.long), accumulate=True
            )
            known_attention.index_put_(row, diagonal_weight, accumulate=True)

    # The signed probe distribution gives a valid Monte Carlo standard error
    # for linear role contribution sums.  Nonlinear absolute energy, entropy,
    # and cancellation continue to be computed only after averaging the signed
    # Jacobian estimate; per-probe absolute values would be positively biased.
    signed_layer_role_estimator_se = torch.full(
        (response_count, layers, role_count),
        torch.nan,
        dtype=dtype,
        device=device,
    )
    if gradient_probes is not None and gradient_probes.shape[0] > 1:
        probe_count = int(gradient_probes.shape[0])
        probe_signed = torch.zeros(
            (probe_count, response_count, layers, role_count),
            dtype=dtype,
            device=device,
        )
        for probe in range(probe_count):
            for layer in range(layers):
                edges = graph.layer_edges(layer, device)
                if edges.count:
                    query = edges.target - graph.response_start
                    keep = query < response_count - 1
                    edges = edges.select(keep)
                    query = query[keep]
                    if edges.count:
                        token_row = query + 1
                        head = edges.head.long()
                        source = edges.source.long()
                        grad = gradient_probes[probe, layer, token_row, head]
                        value = values[layer, source, q_to_kv[head]]
                        phi = edges.weight.to(dtype=dtype) * (grad * value).sum(
                            dim=-1
                        )
                        source_role = torch.full_like(source, HISTORY_ROLE)
                        prompt = source < graph.response_start
                        source_role[prompt] = role_tensor[source[prompt]]
                        probe_signed[probe].index_put_(
                            (
                                token_row,
                                torch.full_like(token_row, layer),
                                source_role,
                            ),
                            phi,
                            accumulate=True,
                        )
                if response_count > 1:
                    query = torch.arange(response_count - 1, device=device)
                    token_row = query + 1
                    head = torch.arange(heads, device=device)[None, :].expand(
                        response_count - 1, heads
                    )
                    token_grid = token_row[:, None].expand_as(head)
                    query_absolute = predictors_device[token_row][:, None].expand_as(
                        head
                    )
                    grad = gradient_probes[
                        probe, layer, token_grid, head
                    ]
                    value = values[layer, query_absolute, q_to_kv[head]]
                    diagonal_weight = graph.diagonal[
                        : response_count - 1, layer
                    ].to(device=device, dtype=dtype)
                    phi = diagonal_weight * (grad * value).sum(dim=-1)
                    probe_signed[
                        probe,
                        token_row,
                        layer,
                        HISTORY_ROLE,
                    ] += phi.sum(dim=1)
        probe_signed[:, 0] = torch.nan
        signed_layer_role_estimator_se = (
            probe_signed.std(dim=0, correction=1) / np.sqrt(probe_count)
        )

    entropy, hhi = _entropy_and_hhi(
        energy_sum, energy_log_sum, energy_square_sum, endpoint_count
    )
    signed_layer_role = signed_role.sum(dim=2)
    absolute_layer_role = absolute_role.sum(dim=2)
    total_signed = signed_layer_role.sum(dim=-1)
    total_energy = absolute_layer_role.sum(dim=-1)
    cancellation = torch.where(
        total_energy > 0,
        1.0 - total_signed.abs() / total_energy.clamp_min(EPSILON),
        torch.full_like(total_energy, torch.nan),
    ).clamp(0.0, 1.0)
    role_cancellation = torch.where(
        absolute_layer_role > 0,
        1.0 - signed_layer_role.abs() / absolute_layer_role.clamp_min(EPSILON),
        torch.full_like(absolute_layer_role, torch.nan),
    ).clamp(0.0, 1.0)
    head_role_js = _head_role_js(absolute_role)

    if response_count > 1:
        expected_coverage = 1.0 - graph.unresolved[: response_count - 1].to(
            device=device, dtype=dtype
        )
        if not torch.allclose(
            known_attention[1:],
            expected_coverage,
            atol=5e-4,
            rtol=0.0,
        ):
            raise ValueError("functional endpoints do not conserve known attention mass")

    available = torch.ones((response_count,), dtype=torch.bool, device=device)
    if response_count:
        available[0] = False
        for value in (
            signed_role,
            absolute_role,
            signed_layer_role,
            absolute_layer_role,
            total_signed,
            total_energy,
            entropy,
            hhi,
            known_attention,
            cancellation,
            role_cancellation,
            head_role_js,
        ):
            value[0] = torch.nan

    return {
        "functional_available": available.cpu().numpy(),
        "functional_role_names": np.asarray(FUNCTIONAL_ROLE_NAMES),
        "functional_signed_role": signed_role.cpu().numpy(),
        "functional_absolute_role": absolute_role.cpu().numpy(),
        "functional_signed_layer_role": signed_layer_role.cpu().numpy(),
        "functional_signed_layer_role_estimator_se": (
            signed_layer_role_estimator_se.cpu().numpy()
        ),
        "functional_absolute_layer_role": absolute_layer_role.cpu().numpy(),
        "functional_total_signed": total_signed.cpu().numpy(),
        "functional_total_absolute": total_energy.cpu().numpy(),
        "functional_cancellation": cancellation.cpu().numpy(),
        "functional_role_cancellation": role_cancellation.cpu().numpy(),
        "functional_entropy_observed": entropy.cpu().numpy(),
        "functional_hhi_observed": hhi.cpu().numpy(),
        "functional_head_role_js": head_role_js.cpu().numpy(),
        "functional_known_attention_coverage": known_attention.cpu().numpy(),
        "functional_observed_endpoint_count": endpoint_count.cpu().numpy(),
        "functional_cached_query_index": np.concatenate(
            (np.asarray([-1], dtype=np.int64), np.arange(response_count - 1))
        ),
        "functional_predictor_position": predictors.cpu().numpy(),
        "functional_token_index": np.arange(response_count, dtype=np.int64),
    }


__all__ = [
    "FUNCTIONAL_ROLE_NAMES",
    "HISTORY_ROLE",
    "PROMPT_ROLE_NAMES",
    "functional_flow",
    "prompt_role_array",
]
