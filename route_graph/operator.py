"""Read endpoint alignment beyond role mass using ordered attention paths."""

from __future__ import annotations

import numpy as np


class PathEncoder:
    """No learned parameters; attention indexes receiver first, source second.

    Roles are source=0, instruction=1, history=2, special=3. Signals have
    shape [layer input, node, candidate contrast]; attention is [L,H,N,N].
    Earlier heads are averaged; the last head remains an explicit channel.
    """

    def __init__(self, depths: tuple[int, ...] = (1, 2)) -> None:
        if not depths or tuple(sorted(set(depths))) != depths or min(depths) < 1:
            raise ValueError("depths must be increasing distinct positive integers")
        self.depths = depths

    def encode(
        self,
        attention: np.ndarray,
        signals: np.ndarray,
        roles: np.ndarray,
        *,
        query: int,
        end_layers: tuple[int, ...],
        source_units: np.ndarray | None = None,
        layer_offset: int = 0,
    ) -> dict:
        if attention.ndim != 4 or attention.shape[-2:] != (len(roles), len(roles)):
            raise ValueError("attention must have shape [layers, heads, nodes, nodes]")
        if signals.ndim != 3 or signals.shape[:2] != (len(attention), len(roles)):
            raise ValueError("signals must have shape [layers, nodes, contrasts]")
        if not 0 <= query < len(roles) or not np.isin(roles, range(4)).all():
            raise ValueError("invalid query or role")
        if not end_layers or tuple(sorted(set(end_layers))) != end_layers:
            raise ValueError("end_layers must be increasing and distinct")
        if min(end_layers) < max(self.depths) - 1 or max(end_layers) >= len(attention):
            raise ValueError("end_layers must have the requested depth ancestry")
        a = np.asarray(attention[:, :, : query + 1, : query + 1])
        x = np.asarray(signals[:, : query + 1], dtype=np.float64)
        roles = roles[: query + 1]
        if source_units is None:
            source_units = np.zeros(len(roles), dtype=int)
        source_units = np.asarray(source_units)[: query + 1]
        if source_units.shape != roles.shape or (source_units[roles == 0] < 0).any():
            raise ValueError(
                "source units must align with nodes and be nonnegative for source tokens"
            )
        groups = np.where(roles == 0, 4 + source_units, roles)
        if not np.isfinite(a).all() or not np.isfinite(x).all() or (a < 0).any():
            raise ValueError(
                "attention and signals must be finite; attention nonnegative"
            )
        if np.triu(a, k=1).any() or not np.allclose(a.sum(-1), 1, atol=1e-5):
            raise ValueError("attention must be causal and row stochastic")
        result = {
            key: [] for key in ("observed", "null", "residual", "signal", "names")
        }
        masks = {"source": roles == 0, "history": roles == 2}
        relay_mask = (roles == 2).copy()
        relay_mask[query] = False
        for layer in end_layers:
            for depth in self.depths:
                start = layer - depth + 1
                for role, mask in masks.items():
                    seed = x[start] * mask[:, None]
                    observed, null = seed.copy(), seed.copy()
                    for previous in range(start, layer):
                        weights = a[previous].mean(axis=0)
                        observed = (observed + weights @ observed) / 2
                        null = (
                            null
                            + _null_action(weights, null, groups, np.arange(len(roles)))
                        ) / 2
                    paths = [(role, observed, null)]
                    if role == "source" and depth > 1:
                        paths.append(
                            (
                                "source_via_history",
                                observed * relay_mask[:, None],
                                null * relay_mask[:, None],
                            )
                        )
                    mean_signal = (
                        x[start, mask].mean(0) if mask.any() else np.zeros(x.shape[-1])
                    )
                    for path, path_observed, path_null in paths:
                        for head, weights in enumerate(a[layer]):
                            obs = (
                                path_observed[query] + weights[query] @ path_observed
                            ) / 2
                            null_read = _null_action(
                                weights[query : query + 1],
                                path_null,
                                groups,
                                np.array([query]),
                            )[0]
                            ref = (path_null[query] + null_read) / 2
                            result["observed"].extend(obs.tolist())
                            result["null"].extend(ref.tolist())
                            result["residual"].extend((obs - ref).tolist())
                            result["signal"].extend(mean_signal.tolist())
                            result["names"].extend(
                                f"l{layer + layer_offset}/d{depth}/{path}/h{head}/c{candidate + 1}"
                                for candidate in range(x.shape[-1])
                            )
        return result


def _null_action(
    weights: np.ndarray, signal: np.ndarray, groups: np.ndarray, receivers: np.ndarray
) -> np.ndarray:
    """Mean edge permutation within role × source unit × floor(log2(lag)).

    Self edges and singleton cells remain fixed. Prefix sums apply the null
    without constructing another dense adjacency matrix. `receivers` allows
    the last step to read only q, while earlier steps propagate to all nodes.
    """
    rows = np.arange(len(receivers))
    result = weights[rows, receivers, None] * signal[receivers]
    for group in np.unique(groups):
        mask = groups == group
        weight_sum = np.pad(
            np.cumsum(weights * mask, axis=1, dtype=np.float64), ((0, 0), (1, 0))
        )
        signal_sum = np.vstack(
            (np.zeros((1, signal.shape[-1])), np.cumsum(signal * mask[:, None], axis=0))
        )
        count = np.r_[0, np.cumsum(mask)]
        for power in range(len(groups).bit_length()):
            left = np.maximum(0, receivers - (2 ** (power + 1) - 1))
            right = np.maximum(0, receivers - 2**power + 1)
            mass = weight_sum[rows, right] - weight_sum[rows, left]
            mean = (signal_sum[right] - signal_sum[left]) / np.maximum(
                count[right] - count[left], 1
            )[:, None]
            result += mass[:, None] * mean
    return result
