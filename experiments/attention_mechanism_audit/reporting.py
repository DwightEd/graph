"""Concise, presentation-ready rendering of mechanism audit reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    label: str
    key: str
    scale: float = 1.0
    unit: str = ""


GROUPS = (
    (
        "1. RESPONSE-DIRECTED FUNCTIONAL ROUTING",
        (
            Metric(
                "routing imbalance [primary]",
                "message_routing_drift_mean",
                100.0,
                "%/pp",
            ),
        ),
    ),
    (
        "2. ROUTE DISPERSION / CONTRACTION",
        (
            Metric(
                "source dispersion [primary]",
                "message_source_dispersion_mean",
                100.0,
                "%/pp",
            ),
            Metric(
                "head-role JS [secondary]",
                "head_role_disagreement_mean",
                1.0,
                "JS",
            ),
        ),
    ),
    (
        "3. OBSERVED-TOKEN EVIDENCE SUPPORT AND CAPTURE CANDIDATE",
        (
            Metric(
                "evidence effect [primary]",
                "evidence_message_effect",
                1.0,
                "logp",
            ),
            Metric(
                "E/R-independent [secondary]",
                "message_independent_capture_signature",
                100.0,
                "%/pp",
            ),
        ),
    ),
)

ONSET = (
    Metric("response-evidence imbalance", "message_routing_drift_mean", 100.0, "pp"),
    Metric("source dispersion", "message_source_dispersion_mean", 100.0, "pp"),
    Metric("evidence causal effect", "evidence_message_effect", 1.0, "logp"),
)


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _interval(interval: list[float | None], scale: float) -> str:
    if interval[0] is None:
        return "[n/a,n/a]"
    return f"[{interval[0] * scale:+.4f},{interval[1] * scale:+.4f}]"


def _status(interval: list[float | None]) -> str:
    low, high = interval
    if low is None:
        return "unavailable"
    if low > 0:
        return "CI>0"
    if high < 0:
        return "CI<0"
    return "CI crosses 0"


def _direction(report: dict, key: str) -> str:
    low, high = report["summaries"][key]["ci95"]
    if low is None:
        return "unavailable"
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "unresolved"


def _group_result(report: dict, index: int) -> str:
    if index == 0:
        imbalance = _direction(report, "message_routing_drift_mean")
        if imbalance == "positive":
            return (
                "GROUP RESULT nominal support for greater response-directed functional "
                "routing; this alone is not temporal drift."
            )
        return "GROUP RESULT response-directed functional routing is not resolved."
    if index == 1:
        dispersion = _direction(report, "message_source_dispersion_mean")
        disagreement = _direction(report, "head_role_disagreement_mean")
        if dispersion == "positive":
            return "GROUP RESULT nominal support for broader source dispersion."
        if disagreement == "negative":
            return (
                "GROUP RESULT broad source dispersion is not supported; secondary evidence "
                "shows greater attention-head role agreement."
            )
        return "GROUP RESULT neither broad dispersion nor contraction is resolved."
    evidence = _direction(report, "evidence_message_effect")
    capture = _direction(report, "message_independent_capture_signature")
    if evidence == "negative" and capture == "positive":
        return (
            "GROUP RESULT nominal support for a weaker observed-token evidence effect plus a "
            "minority E/R-message-independent candidate; not pure parameter knowledge."
        )
    if evidence == "negative":
        return "GROUP RESULT nominal support for a weaker observed-token evidence effect only."
    return "GROUP RESULT evidence-support/capture differences are not resolved."


def _summary_line(report: dict, metric: Metric) -> str:
    result = report["summaries"][metric.key]
    delta = result["position_matched_source_equal_difference"]
    correct = result["correct_mean"]
    hallucinated = result["hallucinated_mean"]
    return (
        f"{metric.label:31s} "
        f"raw(C/H)={_number(None if correct is None else correct * metric.scale)}"
        f"/{_number(None if hallucinated is None else hallucinated * metric.scale)} "
        f"matched_delta={_number(None if delta is None else delta * metric.scale, signed=True)} "
        f"CI={_interval(result['ci95'], metric.scale)} "
        f"S={result['sources']} cells={result['matched_cells']} "
        f"{metric.unit} {_status(result['ci95'])}"
    )


def _onset_line(report: dict, metric: Metric) -> str:
    result = report["matched_onset"][metric.key]
    index = result["offset"].index(0)
    value = result["difference_in_difference"][index]
    low = result["ci95_low"][index]
    high = result["ci95_high"][index]
    interval = [low, high]
    return (
        f"{metric.label:31s} "
        f"DiD@0={_number(None if value is None else value * metric.scale, signed=True)} "
        f"CI={_interval(interval, metric.scale)} "
        f"events={result['events'][index]} S={result['sources'][index]} "
        f"{metric.unit} {_status(interval)}"
    )


def render_report(report: dict, *, all_metrics: bool = False) -> str:
    """Return the key audit results and their exact interpretation boundary."""

    tokens = report["tokens"]
    positives = report["hallucinated_tokens"]
    lines = [
        "=== KEY THREE-MECHANISM AUDIT ===",
        (
            f"DATA samples={report['samples']} tokens={tokens} "
            f"hallucinated={positives} prevalence={positives / tokens:.4%}"
        ),
        "SCOPE post-hoc mechanism audit; this is not a trained or unsupervised detector.",
        "DETECTION NOT EVALUATED: no label-free score, threshold, AUROC, or AUPRC.",
        (
            "TEST matched_delta = hallucinated - correct within source + absolute-position "
            "+ relative-position cells; sources receive equal weight."
        ),
        (
            "UNCERTAINTY 95% source-bootstrap CI; nominal only, with no multiple-testing "
            "correction. raw(C/H) is descriptive and is not the test statistic."
        ),
        (
            "PRIMARY ENDPOINTS routing imbalance, overall source dispersion, and observed-token "
            "evidence effect; head-role JS and E/R-independent capture are secondary."
        ),
    ]
    for index, (title, metrics) in enumerate(GROUPS):
        lines.extend(("", f"[{title}]"))
        lines.extend(_summary_line(report, metric) for metric in metrics)
        lines.append(_group_result(report, index))

    lines.extend(
        (
            "",
            "[4. FIRST-HALLUCINATION ONSET]",
            (
                "DiD@0 = change from token tau-1 to the first hallucinated token tau, "
                "minus the matched correct pseudo-onset change."
            ),
        )
    )
    lines.extend(_onset_line(report, metric) for metric in ONSET)
    lines.extend(
        (
            "",
            "CALCULATION DEFINITIONS",
            "message edge magnitude = attention * ||W_O(head) V(source)||_2",
            "evidence = every token in the QA passage block, not annotated key-evidence tokens",
            "response share = earlier response-history messages + predictor self-route message",
            "response-evidence imbalance = response message share - evidence message share",
            "source dispersion = normalized entropy over source-token message magnitudes",
            "head-role disagreement = JS divergence among attention-head role distributions",
            "evidence causal effect = observed-token logp(full) - logp(no evidence messages)",
            "response causal effect = observed-token logp(full) - logp(no response messages)",
            (
                "message-independent capture = full margin>0 AND no(evidence,response) "
                "margin>0 AND evidence effect<=0"
            ),
            "",
            "INTERPRETATION BOUNDARY",
            (
                "Routing shares are observational allocation. Temporal drift requires the onset "
                "DiD, and greater response allocation does not imply greater causal dependence."
            ),
            (
                "Positive dispersion/JS would support diffusion; significant negative values "
                "support contraction instead."
            ),
            (
                "Evidence deletion tests support for the observed token; it does not by itself "
                "prove global evidence-utilization failure or hallucination formation."
            ),
            (
                "The no(evidence,response) branch still retains question/constraint messages, "
                "predictor residual/embedding, and MLP dynamics; it is not parameter-only."
            ),
            (
                "Formation claims additionally require observer=generator and evidence that the "
                "observer itself prefers the observed token."
            ),
            "Layer early/late means first/last one-third of Transformer layers, not token time.",
        )
    )

    if all_metrics:
        lines.extend(("", "=== ALL SAVED METRICS ==="))
        for name, result in report["summaries"].items():
            lines.append(
                f"{name:42s} "
                f"raw(C/H)={_number(result['correct_mean'])}/{_number(result['hallucinated_mean'])} "
                f"matched_delta={_number(result['position_matched_source_equal_difference'], signed=True)} "
                f"CI={_interval(result['ci95'], 1.0)} "
                f"S={result['sources']} cells={result['matched_cells']}"
            )
    return "\n".join(lines)


__all__ = ["render_report"]
