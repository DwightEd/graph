"""Sparse, post-observation routing-conditioned residual; no hallucination labels.

Attention is a routing proxy. This module does not estimate semantic mutual
information, causal influence, or a posterior probability of hallucination.
"""

from dataclasses import dataclass
import math
import hashlib
import json

import numpy as np
from scipy import sparse


@dataclass
class FlowGraph:
    response_id: str
    source_id: str
    x: np.ndarray
    adjacency: sparse.csr_matrix  # rows = physical query, columns = earlier key
    start: int
    diagonal: np.ndarray
    attribute: str
    schema: str = ""

    def routing(self):
        a = self.adjacency
        prompt = np.asarray(a[:, :self.start].sum(axis=1)).ravel()
        history = np.asarray(a[:, self.start:].sum(axis=1)).ravel()
        unknown = np.maximum(0., 1. - prompt - history - self.diagonal)
        return np.column_stack((prompt, history, self.diagonal, unknown))

    def routing_entropy(self):
        """Entropy of retained endpoints, self and one unknown bucket, in nats."""
        a = self.adjacency.copy()
        a.data = -a.data * np.log(a.data)
        entropy = np.asarray(a.sum(axis=1)).ravel()
        for mass in (self.diagonal, self.routing()[:, -1]):
            keep = mass > 0
            entropy[keep] -= mass[keep] * np.log(mass[keep])
        return entropy


def build_flow_graph(record, floor=0., attribute="hidden"):
    """Read full-token dense or formal response-CSR without an N x N allocation.

    Retain a channel edge iff weight > floor, then average over ALL channels.
    Rectangular prediction-query caches lack complete physical node attributes.
    """
    if not np.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("floor must be in [0, 1)")
    if attribute not in ("diagonal", "hidden"):
        raise ValueError("attribute must be diagonal or hidden")
    start = record.prompt_length if record.prompt_length is not None else record.response_idx
    if record.prompt_length is not None and record.response_idx not in (0, start):
        raise ValueError("prompt_length and response_idx disagree")
    rows, cols, weights = [], [], []
    if record.attention is not None:
        attn = np.asarray(record.attention)
        if attn.ndim != 4 or attn.shape[-2] != attn.shape[-1]:
            raise ValueError("need full square physical-token attention; prediction-query rectangular cache is unsupported")
        layers, heads, n, _ = attn.shape
        diag = attn.diagonal(axis1=-2, axis2=-1).reshape(layers * heads, n).T
        if not np.isfinite(attn).all() or (attn < 0).any():
            raise ValueError("attention must be finite and nonnegative")
        if (attn.sum(axis=-1) > 1.002).any():
            raise ValueError("attention row mass exceeds one")
        for channel in attn.reshape(-1, n, n):
            r, c = np.nonzero(channel > 0)
            if (c > r).any():
                raise ValueError("future attention edge")
            keep = (c < r) & (channel[r, c] > floor)
            rows.append(r[keep]); cols.append(c[keep]); weights.append(channel[r[keep], c[keep]])
    else:
        if record.sparse is None:
            raise ValueError("no attention data")
        raw = record.sparse
        diagonal = np.asarray(raw["attention_diagonal"])
        if diagonal.ndim != 3:
            raise ValueError("attention_diagonal must have shape [L,H,N]")
        layers, heads, n = diagonal.shape
        if not 0 < start < n:
            raise ValueError("need nonempty prompt and response")
        diag = diagonal.reshape(layers * heads, n).T
        ptr = np.asarray(raw["response_row_ptr"])
        col = np.asarray(raw["response_column_indices"])
        val = np.asarray(raw["response_values"], dtype=float)
        expected = layers * heads * (n - start)
        if (ptr.ndim != 1 or len(ptr) != expected + 1 or ptr.dtype.kind not in "iu"
                or ptr[0] != 0 or np.any(np.diff(ptr.astype(np.int64)) < 0)
                or col.ndim != 1 or val.ndim != 1 or len(col) != len(val)
                or ptr[-1] != len(val) or col.dtype.kind not in "iu"):
            raise ValueError("invalid response CSR pointers/columns")
        if not np.isfinite(val).all() or (val < 0).any() or (col < 0).any() or (col >= n).any():
            raise ValueError("invalid CSR values/indices")
        for row in range(expected):
            q = start + row % (n - start)
            c, v = col[ptr[row]:ptr[row + 1]], val[ptr[row]:ptr[row + 1]]
            if (c > q).any() or len(np.unique(c)) != len(c):
                raise ValueError("future or duplicate CSR edge")
            self_value = diag[q, row // (n - start)]
            if (c == q).any() and not np.allclose(v[c == q], self_value, atol=0.002):
                raise ValueError("CSR self edge disagrees with diagonal")
            if v[c < q].sum() + self_value > 1.002:
                raise ValueError("retained CSR mass plus self exceeds one")
            keep = (c < q) & (v > floor)
            rows.append(np.full(keep.sum(), q)); cols.append(c[keep]); weights.append(v[keep])
    if not layers or not heads or not 0 < start < n:
        raise ValueError("need channels, nonempty prompt and nonempty response")
    if not np.isfinite(diag).all() or (diag < 0).any() or (diag > 1).any():
        raise ValueError("invalid attention diagonal")
    adjacency = sparse.coo_matrix((np.concatenate(weights) / (layers * heads),
                                   (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    adjacency.eliminate_zeros()
    mean_diagonal = diag.mean(axis=1)
    total = np.asarray(adjacency.sum(axis=1)).ravel() + mean_diagonal
    # Correct only FP16 rounding overflow; missing retained mass stays missing.
    rounding_scale = 1. / np.maximum(total, 1.)
    adjacency.data *= np.repeat(rounding_scale, np.diff(adjacency.indptr))
    mean_diagonal = mean_diagonal * rounding_scale
    if attribute == "hidden":
        if record.hidden is None:
            raise ValueError("hidden attributes requested but absent")
        raw_schema = (record.metadata or {}).get("hidden_schema")
        if raw_schema is None:
            raise ValueError("hidden_schema provenance is required for hidden attributes")
        schema_data = json.loads(str(np.asarray(raw_schema).item()))
        required = ("observer_model", "observer_revision", "tokenizer", "tokenizer_revision", "layer", "state_location")
        if not isinstance(schema_data, dict) or any(not isinstance(schema_data.get(k), str) or not schema_data[k] for k in required):
            raise ValueError("hidden_schema needs observer/tokenizer revisions, layer and state_location")
        schema = hashlib.sha256(json.dumps(schema_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        x = np.asarray(record.hidden, dtype=np.float32)
    else:
        x = diag.astype(np.float32)
        schema = f"diagonal:{layers}:{heads}"
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("node attributes must be finite [N,D]")
    return FlowGraph(record.response_id, record.source_id, x, adjacency, start,
                     mean_diagonal, attribute, schema)


def source_weights(graphs):
    """Each source has equal weight, then each answer, then each response token."""
    if not graphs:
        raise ValueError("empty reference roster")
    counts = {}
    for g in graphs:
        counts[g.source_id] = counts.get(g.source_id, 0) + 1
    return np.concatenate([np.full(len(g.x) - g.start,
                           1. / (len(counts) * counts[g.source_id] * (len(g.x) - g.start))) for g in graphs])


class RoutingResidual:
    """Post-observation ridge residual; query weights already depend on target.

    No direct target-state input, but NOT a prefix-measurable innovation.
    """

    def __init__(self, projection_dim=64, ridge=0.01, variance_floor=0.05,
                 seed=20260914, mode="graph"):
        if projection_dim < 1 or ridge <= 0 or variance_floor <= 0:
            raise ValueError("positive dimension, ridge and variance_floor required")
        if mode not in ("graph", "no_neighbors", "permuted_weights", "mass_matched_uniform"):
            raise ValueError("unknown control mode")
        self.projection_dim = projection_dim
        self.ridge, self.variance_floor, self.seed, self.mode = ridge, variance_floor, seed, mode

    def _adjacency(self, g):
        if self.mode != "permuted_weights":
            return g.adjacency
        # Independent per query: appending future nodes cannot change past RNG.
        a = g.adjacency.copy()
        for q in range(g.start, len(g.x)):
            begin, end = a.indptr[q:q + 2]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, q]))
            for prompt in (True, False):
                ids = np.arange(begin, end)[(a.indices[begin:end] < g.start) == prompt]
                a.data[ids] = rng.permutation(a.data[ids])
        return a

    def _design(self, g):
        if g.attribute != self.attribute or g.schema != self.schema or g.x.shape[1] != self.projection.shape[0]:
            raise ValueError("node attribute schema differs from fitted model")
        z = (g.x @ self.projection - self.center) / self.scale
        a = self._adjacency(g)
        p = np.asarray(a[:, :g.start].sum(axis=1)).ravel()
        h = np.asarray(a[:, g.start:].sum(axis=1)).ravel()
        index = np.arange(g.start, len(g.x))
        context = np.column_stack((np.ones(len(index)), np.full(len(index), np.log1p(g.start)),
                                   np.log1p(index - g.start), p[index], h[index]))
        if self.mode == "no_neighbors":
            return z, context, a
        if self.mode == "mass_matched_uniform":
            # Same input dimension/mass as graph; uniform over ALL earlier keys.
            mp = p[index, None] * z[:g.start].mean(axis=0)
            past = np.vstack((np.zeros((1, z.shape[1])), np.cumsum(z[g.start:-1], axis=0)))
            mh = h[index, None] * past / np.maximum(index - g.start, 1)[:, None]
        else:
            mp = (a[:, :g.start] @ z[:g.start])[index]
            mh = (a[:, g.start:] @ z[g.start:])[index]
        return z, np.column_stack((context, mp, mh)), a

    def fit(self, graphs):
        w = source_weights(graphs)
        self.attribute = graphs[0].attribute
        self.schema = graphs[0].schema
        if not self.schema:
            raise ValueError("explicit node attribute schema is required")
        d = graphs[0].x.shape[1]
        if any(g.x.shape[1] != d or g.attribute != self.attribute or g.schema != self.schema for g in graphs):
            raise ValueError("mixed node attribute schemas")
        self.projection = (np.eye(d) if d <= self.projection_dim else
                           np.random.default_rng(self.seed).normal(size=(d, self.projection_dim)) / np.sqrt(self.projection_dim))
        target = np.concatenate([g.x[g.start:] @ self.projection for g in graphs])
        self.center = w @ target
        self.scale = np.maximum(np.sqrt(w @ ((target - self.center) ** 2)), 1e-6)
        designs = [self._design(g) for g in graphs]
        u = np.concatenate([v[1] for v in designs])
        y = np.concatenate([v[0][g.start:] for v, g in zip(designs, graphs)])
        penalty = np.eye(u.shape[1]) * self.ridge
        penalty[0, 0] = 0
        self.beta = np.linalg.solve(u.T @ (w[:, None] * u) + penalty, u.T @ (w[:, None] * y))
        residual = y - u @ self.beta
        self.variance = np.maximum(w @ (residual ** 2), self.variance_floor)
        return self

    def score(self, g):
        z, u, a = self._design(g)
        residual = z[g.start:] - u @ self.beta
        score = np.mean(residual ** 2 / self.variance, axis=1)
        # Sparse identity for sum_j A_ij ||z_i-z_j||^2, without E x D storage.
        mass = np.asarray(a.sum(axis=1)).ravel()
        energy = (mass * np.sum(z * z, axis=1) - 2 * np.sum(z * (a @ z), axis=1)
                  + a @ np.sum(z * z, axis=1)) / z.shape[1]
        return {"score": score, "coding_nats_per_dimension": 0.5 * (score + np.log(2 * np.pi * self.variance).mean()),
                "dirichlet_energy": np.maximum(energy[g.start:], 0.),
                "routing_entropy": g.routing_entropy()[g.start:], "routing_mass": g.routing()[g.start:]}

    def save(self, path):
        np.savez_compressed(path, projection=self.projection, center=self.center,
                            scale=self.scale, beta=self.beta, variance=self.variance,
                            attribute=self.attribute, schema=self.schema, mode=self.mode, seed=self.seed,
                            ridge=self.ridge, variance_floor=self.variance_floor,
                            projection_dim=self.projection_dim)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(projection_dim=int(data["projection_dim"]), ridge=float(data["ridge"]),
                        variance_floor=float(data["variance_floor"]), seed=int(data["seed"]), mode=str(data["mode"]))
            for name in ("projection", "center", "scale", "beta", "variance"):
                setattr(model, name, data[name].copy())
            model.attribute = str(data["attribute"])
            model.schema = str(data["schema"])
        return model


class SourceCalibration:
    """Conservative source-block tail ranks; not normal-only false-positive rates."""

    def __init__(self, predictions, source_ids, alpha=0.05):
        if not 0 < alpha < 1 or not predictions or len(predictions) != len(source_ids):
            raise ValueError("need nonempty paired calibration and alpha in (0,1)")
        maxima = {}
        for values, source in zip(predictions, source_ids):
            values = np.asarray(values)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError("invalid calibration scores")
            maxima[source] = max(maxima.get(source, -np.inf), float(values.max()))
        self.maxima = np.sort(list(maxima.values()))
        self.alpha = alpha

    def p_values(self, score):
        score = np.asarray(score)
        if not np.isfinite(score).all():
            raise ValueError("nonfinite test score")
        return (1 + len(self.maxima) - np.searchsorted(self.maxima, score, side="left")) / (len(self.maxima) + 1)

    def summary(self):
        k = math.ceil((len(self.maxima) + 1) * (1 - self.alpha))
        return {"alpha": self.alpha, "sources": len(self.maxima), "rank": k,
                "threshold": float(self.maxima[k - 1]) if k <= len(self.maxima) else None,
                "threshold_is_infinite": k > len(self.maxima), "comparison": "score > threshold",
                "source_maxima": self.maxima.tolist(), "scope": "exchangeable unlabeled source-block any-alarm"}
