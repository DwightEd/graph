"""Label-free packed input and answer-local graph operations."""

import numpy as np
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler
from state_audit.storage import read_json

from experiments.probabilistic_detection.readout import design_matrix


INPUTS = ("context", "observations", "source_index", "answer_index", "target",
          "token_id", "unit_index", "development")


def load_inputs(packs, task, split):
    # np.load is lazy: never access the training archive's label arrays here.
    with np.load(packs / f"{task}_{split}.npz", allow_pickle=False) as saved:
        pack = {name: saved[name] for name in INPUTS}
    metadata = read_json(packs / f"{task}_{split}.json")
    generators = sorted({row["generator"] for row in metadata["records"]})
    pack["generator"] = np.empty(len(pack["target"]), dtype=np.int32)
    for row in metadata["records"]:
        region = slice(row["packed_start"], row["packed_stop"])
        pack["generator"][region] = generators.index(row["generator"])
    return pack, metadata


def runs(answer_index):
    cuts = np.r_[0, np.flatnonzero(np.diff(answer_index)) + 1, len(answer_index)]
    return zip(cuts[:-1], cuts[1:])


def neighbor_mean(matrix, answer_index, radius=4, shuffled=False):
    """Undirected temporal graph without self edges, on valid-token order.

    The shuffled control relabels the graph within each answer. It preserves
    topology and the degree multiset, not the degree of each specific token.
    Singleton answers use a zero context and are retained in evaluation.
    """
    result = np.zeros_like(matrix)
    rng = np.random.default_rng(42)
    for start, stop in runs(answer_index):
        count = stop - start
        order = rng.permutation(count) if shuffled else np.arange(count)
        values = matrix[start:stop][order]
        prefix = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)))
        position = np.arange(count)
        left = np.maximum(0, position - radius)
        right = np.minimum(count, position + radius + 1)
        degree = right - left - 1
        average = (prefix[right] - prefix[left] - values) / np.maximum(degree, 1)[:, None]
        result[start + order] = average
    return result


class NodeFeatures:
    """Refit the existing 123-dimensional expansion without a truth-label API."""

    def fit(self, context, observations):
        self.context_scaler_ = StandardScaler().fit(context)
        scaled = self.context_scaler_.transform(context)
        self.context_spline_ = SplineTransformer(n_knots=4, degree=2, include_bias=False).fit(scaled)
        self.observation_transform_ = QuantileTransformer(
            n_quantiles=min(1000, len(observations)), output_distribution="normal",
            random_state=42).fit(observations)
        matrix = design_matrix(context, observations, self, "conditioned")
        self.scaler_ = StandardScaler().fit(matrix)
        return self

    def _context_design(self, context):
        scaled = self.context_scaler_.transform(context)
        return np.column_stack((np.ones(len(context)), self.context_spline_.transform(scaled)))

    def transform(self, context, observations):
        matrix = design_matrix(context, observations, self, "conditioned")
        return np.clip(self.scaler_.transform(matrix), -8, 8).astype(np.float32)


def strata(pack):
    position_bin = np.minimum((pack["context"][:, 2] * 8).astype(int), 7)
    return pack["generator"] * 8 + position_bin


def matched_donors(pack, rng):
    """Match generator/position bin, and exclude the recipient's entire source."""
    groups = strata(pack)
    donors = np.empty(len(groups), dtype=np.int64)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        sources = pack["source_index"][members]
        if len(np.unique(sources)) < 2:
            raise ValueError("Contrast stratum requires at least two distinct sources")
        chosen = rng.choice(members, len(members))
        invalid = pack["source_index"][chosen] == sources
        while invalid.any():
            chosen[invalid] = rng.choice(members, invalid.sum())
            invalid = pack["source_index"][chosen] == sources
        donors[members] = chosen
    return donors
