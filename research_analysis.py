"""Reusable structural analysis and visualization for canonical attention research data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from descriptors import temporal_summary
from research_dataset import ResearchDataset


GRAPH_FEATURE_NAMES = (
    "incoming_mass",
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_degree",
    "prompt_degree",
    "history_degree",
    "in_density",
    "prompt_density",
    "history_density",
    "history_edge_share",
    "channel_edge_density",
)

DEFAULT_PLOT_FEATURES = (
    "prompt_mass_share",
    "history_edge_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "channel_edge_density",
)


def raw_attention_graph_features(attention, edges) -> torch.Tensor:
    """Return [response_tokens, 12] structural features from canonical sparse attention."""
    response_idx = attention.response_idx
    response_count = attention.num_response_tokens
    num_nodes = attention.num_tokens
    num_channels = attention.num_channels

    features = torch.zeros((response_count, len(GRAPH_FEATURE_NAMES)), dtype=torch.float32)
    if edges["weight"].numel() == 0:
        return features

    source = edges["source"].long().cpu()
    target = edges["target"].long().cpu()
    weight = edges["weight"].float().cpu()
    rows = target - response_idx

    if bool(((rows < 0) | (rows >= response_count)).any()):
        raise ValueError("decoded attention edge targets must be response tokens")
    if bool(((source < 0) | (source >= target)).any()):
        raise ValueError("decoded attention edges must point to earlier tokens")

    # Merge the same source->target relation across layer/head channels.
    pair_key = rows * num_nodes + source
    unique_key, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)
    pair_weight = torch.zeros(unique_key.numel(), dtype=torch.float32)
    pair_weight.index_add_(0, inverse, weight)
    pair_weight /= float(num_channels)

    pair_rows = torch.div(unique_key, num_nodes, rounding_mode="floor")
    pair_source = unique_key.remainder(num_nodes)
    pair_prompt = pair_source < response_idx
    pair_history = ~pair_prompt

    total_mass = torch.zeros(response_count)
    total_mass.index_add_(0, pair_rows, pair_weight)

    prompt_mass = torch.zeros(response_count)
    prompt_mass.index_add_(0, pair_rows[pair_prompt], pair_weight[pair_prompt])
    nonempty = total_mass > 0
    prompt_share = torch.zeros(response_count)
    prompt_share[nonempty] = prompt_mass[nonempty] / total_mass[nonempty]

    in_degree = torch.bincount(pair_rows, minlength=response_count).float()
    prompt_degree = torch.bincount(pair_rows[pair_prompt], minlength=response_count).float()
    history_degree = torch.bincount(pair_rows[pair_history], minlength=response_count).float()

    probabilities = pair_weight / total_mass[pair_rows]
    entropy = torch.zeros(response_count)
    entropy.index_add_(0, pair_rows, -probabilities * probabilities.log())
    normalized_entropy = torch.zeros(response_count)
    multiple = in_degree > 1
    normalized_entropy[multiple] = entropy[multiple] / in_degree[multiple].log()

    history_mass = torch.zeros(response_count)
    history_mass.index_add_(0, pair_rows[pair_history], pair_weight[pair_history])
    history_lag_mass = torch.zeros(response_count)
    if bool(pair_history.any()):
        history_target = response_idx + pair_rows[pair_history]
        lag = (history_target - pair_source[pair_history]).float()
        history_lag_mass.index_add_(
            0,
            pair_rows[pair_history],
            pair_weight[pair_history] * lag / max(response_count - 1, 1),
        )
    history_lag = torch.zeros(response_count)
    has_history_mass = history_mass > 0
    history_lag[has_history_mass] = (
        history_lag_mass[has_history_mass] / history_mass[has_history_mass]
    )

    response_position = torch.arange(response_count, dtype=torch.float32)
    absolute_target = response_idx + response_position
    in_density = in_degree / absolute_target.clamp_min(1.0)
    prompt_density = prompt_degree / float(max(response_idx, 1))
    history_density = torch.zeros(response_count)
    has_history = response_position > 0
    history_density[has_history] = history_degree[has_history] / response_position[has_history]

    history_edge_share = torch.zeros(response_count)
    has_edges = in_degree > 0
    history_edge_share[has_edges] = history_degree[has_edges] / in_degree[has_edges]

    # Retained channel-level entries among all possible earlier-token channel entries.
    channel_degree = torch.bincount(rows, minlength=response_count).float()
    channel_edge_density = channel_degree / (
        float(num_channels) * absolute_target.clamp_min(1.0)
    )

    features = torch.stack(
        (
            total_mass,
            prompt_share,
            normalized_entropy,
            history_lag,
            in_degree,
            prompt_degree,
            history_degree,
            in_density,
            prompt_density,
            history_density,
            history_edge_share,
            channel_edge_density,
        ),
        dim=1,
    )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("raw graph features must be finite")
    return features


def sample_graph_descriptor(sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-level structural features and one mean/std/slope descriptor."""
    attention = sample.attention()
    edges = sample.attention_edges()
    token_features = raw_attention_graph_features(attention, edges)
    return token_features, temporal_summary(token_features)


class SampleBehaviorVisualizer:
    """Inspect and compare individual correct/hallucinated samples by sample_id."""

    def __init__(self, split_root, *, device="cpu", verify_hashes=False):
        self.dataset = ResearchDataset(split_root, device=device, verify_hashes=verify_hashes)
        self.labels = self.dataset.labels()
        self._response_length_cache = {}

    def is_hallucinated(self, sample_id) -> bool:
        return bool(self.labels.positive_runs(str(sample_id)))

    @property
    def error_sample_ids(self) -> list[str]:
        return [sid for sid in self.dataset.sample_ids if self.is_hallucinated(sid)]

    @property
    def correct_sample_ids(self) -> list[str]:
        return [sid for sid in self.dataset.sample_ids if not self.is_hallucinated(sid)]

    def _response_length(self, sample_id) -> int:
        sample_id = str(sample_id)
        if sample_id not in self._response_length_cache:
            self._response_length_cache[sample_id] = self.dataset[sample_id].attention().num_response_tokens
        return self._response_length_cache[sample_id]

    def list_errors(self, limit=20) -> list[dict]:
        """List hallucinated samples without needing to know a sample_id in advance."""
        rows = []
        for sample_id in self.error_sample_ids[:limit]:
            sample = self.dataset[sample_id]
            rows.append(
                {
                    **sample.metadata,
                    "response_tokens": self._response_length(sample_id),
                    "positive_runs": self.labels.positive_runs(sample_id),
                }
            )
        return rows

    def analyze(self, sample_id) -> dict:
        """Load one sample and return labels, token IDs, graph features, and descriptor."""
        sample_id = str(sample_id)
        sample = self.dataset[sample_id]
        attention = sample.attention()
        edges = sample.attention_edges()
        features = raw_attention_graph_features(attention, edges)
        response_labels = self.labels.response_labels(sample)
        return {
            "sample_id": sample_id,
            "source_id": sample.source_id,
            "metadata": sample.metadata,
            "attention": attention,
            "edges": edges,
            "feature_names": GRAPH_FEATURE_NAMES,
            "features": features,
            "descriptor": temporal_summary(features),
            "response_token_ids": attention.token_ids[attention.response_idx:].detach().cpu(),
            "positive_runs": self.labels.positive_runs(sample_id),
            "response_labels": response_labels.cpu(),
        }

    def match_correct(self, error_sample_id, *, max_candidates=128) -> str:
        """Choose a fully correct control, preferring same source/task and similar length."""
        error_sample_id = str(error_sample_id)
        if not self.is_hallucinated(error_sample_id):
            raise ValueError(f"{error_sample_id} is not labeled hallucinated")
        correct_ids = self.correct_sample_ids
        if not correct_ids:
            raise ValueError("no fully correct sample is available for comparison")

        error_sample = self.dataset[error_sample_id]
        error_length = self._response_length(error_sample_id)
        groups = [
            [sid for sid in correct_ids if self.dataset[sid].source_id == error_sample.source_id]
        ]
        if error_sample.task_type is not None and error_sample.data_source is not None:
            groups.append([
                sid for sid in correct_ids
                if self.dataset[sid].task_type == error_sample.task_type
                and self.dataset[sid].data_source == error_sample.data_source
            ])
        if error_sample.task_type is not None:
            groups.append([
                sid for sid in correct_ids if self.dataset[sid].task_type == error_sample.task_type
            ])
        groups.append(correct_ids)
        candidates = next(group for group in groups if group)[:max_candidates]
        return min(candidates, key=lambda sid: (abs(self._response_length(sid) - error_length), sid))

    def span_summary(self, sample_id, pre_window=8, post_window=8) -> list[dict]:
        """Summarize pre/error/post means for every hallucination span."""
        result = self.analyze(sample_id)
        values = result["features"]
        output = []
        for run_index, (start, end) in enumerate(result["positive_runs"]):
            segments = {
                "pre": values[max(0, start - pre_window):start],
                "error": values[start:end],
                "post": values[end:min(len(values), end + post_window)],
            }
            means = {
                name: segment.mean(dim=0) if len(segment) else torch.full((values.shape[1],), float("nan"))
                for name, segment in segments.items()
            }
            for feature_index, feature_name in enumerate(GRAPH_FEATURE_NAMES):
                output.append(
                    {
                        "run": run_index,
                        "start": start,
                        "end": end,
                        "feature": feature_name,
                        "pre": float(means["pre"][feature_index]),
                        "error": float(means["error"][feature_index]),
                        "post": float(means["post"][feature_index]),
                        "error_minus_pre": float(means["error"][feature_index] - means["pre"][feature_index]),
                    }
                )
        return output

    @staticmethod
    def _feature_indices(features):
        names = tuple(features) if features is not None else DEFAULT_PLOT_FEATURES
        unknown = set(names).difference(GRAPH_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        return names, [GRAPH_FEATURE_NAMES.index(name) for name in names]

    def plot(self, sample_id, *, features=None, save_path=None):
        """Plot one sample's token-level structural trajectories and hallucination spans."""
        import matplotlib.pyplot as plt

        result = self.analyze(sample_id)
        names, indices = self._feature_indices(features)
        columns = 2
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        x = np.arange(len(result["features"]))

        for axis, name, index in zip(axes, names, indices):
            axis.plot(x, result["features"][:, index].numpy())
            for start, end in result["positive_runs"]:
                axis.axvspan(start, max(start, end - 1), alpha=0.16)
            axis.set(title=name, xlabel="Response token position")
        for axis in axes[len(names):]:
            axis.set_visible(False)
        figure.suptitle(f"sample={result['sample_id']}  hallucinated={bool(result['positive_runs'])}")
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure

    def compare(self, error_sample_id, correct_sample_id=None, *, features=None, save_path=None):
        """Overlay one hallucinated sample with a manual or automatically matched correct control."""
        import matplotlib.pyplot as plt

        error_sample_id = str(error_sample_id)
        if not self.is_hallucinated(error_sample_id):
            raise ValueError(f"{error_sample_id} is not labeled hallucinated")
        correct_sample_id = self.match_correct(error_sample_id) if correct_sample_id is None else str(correct_sample_id)
        if self.is_hallucinated(correct_sample_id):
            raise ValueError(f"{correct_sample_id} is not a fully correct sample")

        error = self.analyze(error_sample_id)
        control = self.analyze(correct_sample_id)
        names, indices = self._feature_indices(features)
        columns = 2
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)

        error_x = np.linspace(0.0, 1.0, len(error["features"]))
        control_x = np.linspace(0.0, 1.0, len(control["features"]))
        for axis, name, index in zip(axes, names, indices):
            axis.plot(error_x, error["features"][:, index].numpy(), label="Hallucinated")
            axis.plot(control_x, control["features"][:, index].numpy(), label="Correct")
            for start, end in error["positive_runs"]:
                denominator = max(len(error["features"]) - 1, 1)
                axis.axvspan(start / denominator, (end - 1) / denominator, alpha=0.16)
            axis.set(title=name, xlabel="Normalized response position")
            axis.legend()
        for axis in axes[len(names):]:
            axis.set_visible(False)
        figure.suptitle(f"error={error_sample_id} vs correct={correct_sample_id}")
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
        return figure, correct_sample_id
