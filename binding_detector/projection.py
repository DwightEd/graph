"""Exact KL projection onto explicitly supplied evidence relations.

No LLM, training, or hallucination labels. Candidate IDs denote occurrences,
not surface strings. Unlisted tuples are UNKNOWN unless closed_world=True.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Factor:
    variables: tuple[str, ...]
    allowed: frozenset[tuple[str, ...]] = frozenset()
    forbidden: frozenset[tuple[str, ...]] = frozenset()
    closed_world: bool = False
    provenance: str = ""


def _normalize(log_weights):
    x = np.asarray(log_weights, dtype=float)
    if x.ndim != 1 or not len(x) or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("log weights must be a nonempty vector; -inf is allowed")
    if not np.isfinite(x).any():
        raise ValueError("all candidates have zero probability")
    return x - logsumexp(x)


def cosine_logits(query, candidates, temperature=0.1):
    """Optional unfitted matcher, NOT a calibrated posterior or causal effect.

    Call separately for each physical layer/head. No channel averaging here.
    Inputs must be from an explicitly aligned representation space.
    """
    q, k = np.asarray(query, float), np.asarray(candidates, float)
    if (q.ndim != 1 or k.ndim != 2 or k.shape[1] != len(q)
            or not np.isfinite(q).all() or not np.isfinite(k).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("invalid aligned query/candidate vectors or temperature")
    norms = np.linalg.norm(k, axis=1) * np.linalg.norm(q)
    if (norms == 0).any():
        raise ValueError("zero-norm representations do not define a match")
    return (k @ q) / norms / temperature


def project(domains, log_weights, factors, *, joint_log_weights=None, max_states=200_000):
    """Return -log legal mass and bounds when some relations are unknown.

    With no joint table, Q(z)=prod_i softmax(log_weights[i])[z_i] is an EXPLICIT
    independence assumption. Supply a full joint log table to avoid it. No
    partial enumeration, epsilon smoothing, or invented relation completes Q.
    """
    names = list(domains)
    if not names or set(log_weights) != set(names):
        raise ValueError("domains and log_weights must have the same variables")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be positive")
    for name, ids in domains.items():
        if (not isinstance(name, str) or not name or not isinstance(ids, (list, tuple)) or not len(ids)
                or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("unique nonempty occurrence IDs required per variable")
    shape = tuple(len(domains[v]) for v in names)
    total = math.prod(shape)
    if total > max_states:
        raise ValueError(f"{total} assignments exceed max_states={max_states}; no truncated score")
    lp = {v: _normalize(log_weights[v]) for v in names}
    if any(len(lp[v]) != len(domains[v]) for v in names):
        raise ValueError("candidate/log-weight length mismatch")
    if joint_log_weights is None:
        indices = np.array(list(product(*(range(n) for n in shape))))
        logq = sum(lp[v][indices[:, j]] for j, v in enumerate(names))
    else:
        joint = np.asarray(joint_log_weights, float)
        if joint.shape != shape:
            raise ValueError("joint table axes must follow domain insertion order")
        logq = _normalize(joint.ravel())
    # Candidate product order agrees with C-order joint.ravel().
    assignments = list(product(*(domains[v] for v in names)))
    position = {v: j for j, v in enumerate(names)}
    supported = np.ones(total, dtype=bool) if factors else np.zeros(total, dtype=bool)
    possible = np.ones(total, dtype=bool)
    for f in factors:
        if (not f.variables or len(set(f.variables)) != len(f.variables)
                or not set(f.variables) <= set(names) or not f.provenance
                or type(f.closed_world) is not bool):
            raise ValueError("factor requires unique known variables and provenance")
        if f.allowed & f.forbidden:
            raise ValueError("the same tuple cannot be allowed and forbidden")
        for row in f.allowed | f.forbidden:
            if len(row) != len(f.variables) or any(x not in domains[v] for v, x in zip(f.variables, row)):
                raise ValueError("factor tuple outside candidate domains")
        tuples = [tuple(z[position[v]] for v in f.variables) for z in assignments]
        allowed = np.array([z in f.allowed for z in tuples])
        denied = np.array([z in f.forbidden for z in tuples])
        supported &= allowed
        possible &= allowed if f.closed_world else ~denied
    supported &= possible
    unknown = possible & ~supported

    def logmass(mask):
        return float(logsumexp(logq[mask])) if mask.any() else -math.inf

    log_supported, log_possible = logmass(supported), logmass(possible)
    lo, hi = max(0., -log_possible), max(0., -log_supported)
    posterior = np.zeros(total)
    if math.isfinite(log_supported):
        posterior[supported] = np.exp(logq[supported] - log_supported)
    q = np.exp(logq)
    before, after = {}, {}
    for j, v in enumerate(names):
        before[v] = [float(q[[z[j] == c for z in assignments]].sum()) for c in domains[v]]
        after[v] = [float(posterior[[z[j] == c for z in assignments]].sum()) for c in domains[v]]
    independent = joint_log_weights is None
    status = ("no_constraints" if not factors else "inconsistent_constraints" if not possible.any()
              else "zero_possible_probability" if not math.isfinite(log_possible)
              else "partial_relations" if unknown.any() else "closed_relations")
    uniform_possible = float(possible.mean())
    return dict(status=status, assignments=total, factor_count=len(factors),
                q_assumption="independent_marginals" if independent else "supplied_joint",
                lower_nats=lo, upper_nats=hi,
                supported_mass=float(q[supported].sum()), unknown_mass=float(q[unknown].sum()),
                forbidden_mass=float(q[~possible].sum()),
                uniform_lower_nats=-math.log(uniform_possible) if uniform_possible else math.inf,
                node_entropy=float(np.mean([-sum(p * math.log(p) for p in marginal if p > 0) for marginal in before.values()])),
                node_confidence=float(np.mean([max(marginal) for marginal in before.values()])),
                node_marginals=before, projected_marginals=after,
                projection_defined=math.isfinite(log_supported),
                semantic_validity="conditional_on_supplied_relations_not_certified")


def permute_relations(domains, factors, seed, groups=None):
    """Global per-variable identity permutation, shared across every factor.

    Preserves factor topology and table cardinality. Optional groups map each
    variable's IDs to exchangeability/type groups. Not a hallucination null.
    """
    rng = np.random.default_rng(seed)
    maps = {}
    for v, ids in domains.items():
        labels = ["all"] * len(ids) if groups is None else groups[v]
        if len(labels) != len(ids):
            raise ValueError("permutation groups must align with candidate IDs")
        mapping = {}
        for label in sorted(set(labels), key=str):
            bucket = [c for c, g in zip(ids, labels) if g == label]
            mapping.update(zip(bucket, rng.permutation(bucket).tolist()))
        maps[v] = mapping
    def changed(rows, variables):
        return frozenset(tuple(maps[v][c] for v, c in zip(variables, row)) for row in rows)
    shuffled = [Factor(f.variables, changed(f.allowed, f.variables), changed(f.forbidden, f.variables),
                       f.closed_world, f.provenance + ";identity_permuted") for f in factors]
    moved = sum(c != d for m in maps.values() for c, d in m.items())
    return shuffled, dict(maps=maps, moved_fraction=moved / sum(map(len, maps.values())))
