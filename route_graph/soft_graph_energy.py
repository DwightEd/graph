"""Fixed, label-free graph inference. Scores are predictions, not certificates."""

import math

EPS = 1e-6
VARIANTS = ("qwen_fullsource_no_graph", "qwen_matcher_no_native", "graph_no_history", "full_graph")


def distribution(value, labels):
    if set(value) != set(labels):
        raise ValueError("finite distribution labels differ")
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in value.values()):
        raise ValueError("invalid finite distribution")
    total = sum(value.values())
    if not math.isclose(total, 1, abs_tol=1e-5):
        raise ValueError("finite distribution does not sum to one")
    return {k: float(value[k]) / total for k in labels}


def positive_dependence(native):
    """Require actual matched-control and numerical checks, never fill failures."""
    controls = native.get("control_deltas", [])
    identities = native.get("control_ids", [])
    valid = (native.get("sham_exact") is True and native.get("prefix_exact") is True
             and native.get("controls_valid") is True and len(controls) >= 2
             and len(identities) == len(controls) and len(set(identities)) == len(identities)
             and native.get("selected_id") not in identities
             and native.get("origin_donor_sham_exact") is True
             and native.get("origin_recipient_sham_exact") is True
             and native.get("origin_artifacts_verified") is True
             and len(native.get("origin_measurement_ids", [])) == len(controls) + 1
             and len(set(native.get("origin_measurement_ids", []))) == len(controls) + 1
             and native.get("measurement_scope") == "raw_origin_mediated_target_dependence"
             and native.get("repeat_valid") is True)
    if not valid:
        return 0.0
    values = [native["delta"], *controls]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite native effect")
    delta = values[0] - max(0.0, *controls)
    return max(0.0, math.tanh(delta / 0.5))


def _clip(value):
    return min(2.0, max(-2.0, value))


def _score(logit, confidence):
    exp = math.exp(-abs(logit))
    sigmoid = 1 / (1 + exp) if logit >= 0 else exp / (1 + exp)
    return 0.5 + confidence * (sigmoid - 0.5)


def infer_claim(claim, previous):
    """Infer four synchronized predictions; previous is an ordered claim registry.

    Keys are immutable role/source/event occurrences. Duplicate assignment paths
    must have been marginalized once by the matcher. Future edges are rejected.
    A low-confidence previous prediction cannot transmit its unshrunk logit.
    """
    q = distribution(claim["q_global"], "SCNU")
    global_logit = math.log((q["C"] + q["N"] + EPS) / (q["S"] + EPS))
    global_confidence = 1 - q["U"]
    source, seen, role_masses = [], set(), {}
    for item in claim.get("source_terms", []):
        key = tuple(item["key"])
        if key in seen:
            raise ValueError("duplicate role/source/event source term")
        seen.add(key)
        pi, quality = item["pi"], item["pool_quality"]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (pi, quality)):
            raise ValueError("candidate weights must be finite in [0,1]")
        role_masses[key[0]] = role_masses.get(key[0], 0.) + pi
        if role_masses[key[0]] > 1 + 1e-6:
            raise ValueError("response role candidate marginal mass exceeds one")
        local = distribution(item["q_local"], "SCU")
        w = positive_dependence(item.get("native", {}))
        alpha_without_native = pi * quality * (local["S"] + local["C"])
        source.append({**item, "w_native_positive_specific": w,
                       "alpha": alpha_without_native * w,
                       "alpha_without_native": alpha_without_native,
                       "signed_relation_logit": _clip(math.log((local["C"] + EPS) / (local["S"] + EPS)))})
    sums = {}
    for field in ("alpha", "alpha_without_native"):
        weight = sum(s[field] for s in source)
        sums[field] = (sum(s[field] * s["signed_relation_logit"] for s in source) / max(1., weight), min(1., weight))
    history, prior_ids = [], set()
    for edge in claim.get("history_terms", []):
        pid = edge["previous_claim_id"]
        if pid not in previous or previous[pid]["span"][1] > claim["span"][0]:
            raise ValueError("history must refer to a frozen, earlier nonoverlapping claim")
        if pid in prior_ids:
            raise ValueError("duplicate history edge")
        prior_ids.add(pid)
        probs = distribution(edge["q_relation"], ("reuse", "correction", "quote", "new_topic", "unknown"))
        label = max(probs, key=probs.get)
        relation_known = probs[label] >= .8
        if not relation_known:
            label = "unknown"
        native = positive_dependence(edge.get("native", {}))
        scope = {"mapped_previous_slot": 1., "mapped_previous_event": .5, "unmapped": 0.}[edge["origin_scope"]]
        beta = native * probs["reuse"] * scope if label == "reuse" else 0.
        prior = previous[pid]["variants"]["full_graph"]
        # This additional confidence factor prevents unknown-but-large raw energy
        # from becoming confident downstream evidence.
        beta *= prior["confidence"]
        history.append({**edge, "edge_label": label, "eligible": label == "reuse" and beta > 0,
                        "relation_known": relation_known,
                        "beta": beta, "previous_logit_clipped": _clip(prior["Lraw"])})
    history_weight = sum(e["beta"] for e in history)
    lh = sum(e["beta"] * e["previous_logit_clipped"] for e in history) / max(1., history_weight)
    kh = min(1., history_weight)
    variants = {}
    for name in VARIANTS:
        ls, ks = (0., 0.) if name == VARIANTS[0] else sums["alpha_without_native" if name == VARIANTS[1] else "alpha"]
        lhist, khist = (lh, kh) if name == "full_graph" else (0., 0.)
        raw = global_logit + ls + lhist
        conf = 1 - (1 - global_confidence) * (1 - ks) * (1 - khist)
        variants[name] = {"Lglobal": global_logit, "Lsource": ls, "Lhistory": lhist,
                          "Lraw": raw, "confidence": conf, "unknown_mass": 1 - conf,
                          "confidence_meaning": "evidence_coverage_not_probability_of_correct_prediction",
                          "directional_confidence": 2 * abs(_score(raw, conf) - .5),
                          "risk_score": _score(raw, conf)}
    return {"claim_id": claim["claim_id"], "span": claim["span"], "q_global": q,
            "extraction_kind": claim.get("extraction_kind", "unspecified"),
            "source_terms": source, "history_terms": history, "variants": variants,
            "scope": "soft_detection; not_calibrated_posterior_or_causal_error_certificate",
            "energy_version": "fixed-energy-origin@20260913", "certificate_ids": []}


def infer_response(claims):
    previous = {}
    for claim in sorted(claims, key=lambda c: (c["span"][0], c["span"][1], c["claim_id"])):
        if claim["claim_id"] in previous:
            raise ValueError("duplicate claim ID")
        previous[claim["claim_id"]] = infer_claim(claim, previous)
    return list(previous.values())
