"""Compact rendering of mechanism-audit results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    label: str
    key: str
    scale: float = 1.0
    delta_unit: str = ""


KEY_METRICS = (
    Metric("Routing imbalance (%)", "message_routing_drift_mean", 100.0, "pp"),
    Metric("Source dispersion (%)", "message_source_dispersion_mean", 100.0, "pp"),
    Metric("Head-role JS", "head_role_disagreement_mean"),
    Metric("Evidence effect (log p)", "evidence_message_effect"),
    Metric(
        "E/R-independent candidate (%)",
        "message_independent_capture_signature",
        100.0,
        "pp",
    ),
)

ONSET_METRICS = (
    Metric("Routing imbalance", "message_routing_drift_mean", 100.0, "pp"),
    Metric("Source dispersion", "message_source_dispersion_mean", 100.0, "pp"),
    Metric("Evidence effect", "evidence_message_effect"),
)


def _value(value: float | None, scale: float, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value *= scale
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _ci(interval: list[float | None], scale: float) -> str:
    if interval[0] is None:
        return "[n/a, n/a]"
    return f"[{interval[0] * scale:+.4f}, {interval[1] * scale:+.4f}]"


def _direction(report: dict, key: str) -> str:
    low, high = report["summaries"][key]["ci95"]
    if low is None or low <= 0 <= high:
        return "unresolved"
    return "higher" if low > 0 else "lower"


def _metric_row(report: dict, metric: Metric) -> str:
    result = report["summaries"][metric.key]
    delta = result["position_matched_source_equal_difference"]
    delta_text = _value(delta, metric.scale, signed=True)
    if delta is not None and metric.delta_unit:
        delta_text += f" {metric.delta_unit}"
    return (
        f"{metric.label:33s} "
        f"{_value(result['correct_mean'], metric.scale):>10s} "
        f"{_value(result['hallucinated_mean'], metric.scale):>10s} "
        f"{delta_text:>13s} "
        f"{_ci(result['ci95'], metric.scale)}"
    )


def _findings(report: dict) -> list[str]:
    routing = _direction(report, "message_routing_drift_mean")
    dispersion = _direction(report, "message_source_dispersion_mean")
    head_js = _direction(report, "head_role_disagreement_mean")
    evidence = _direction(report, "evidence_message_effect")
    capture = _direction(report, "message_independent_capture_signature")

    routing_text = (
        "Routing: hallucinated tokens have a higher response-evidence message imbalance."
        if routing == "higher"
        else "Routing: the matched response-evidence imbalance is not resolved."
    )
    if dispersion == "unresolved" and head_js == "lower":
        dispersion_text = (
            "Dispersion: overall source entropy is unchanged; head role distributions are "
            "more similar for hallucinated tokens."
        )
    elif dispersion == "higher":
        dispersion_text = "Dispersion: source-message entropy is higher for hallucinated tokens."
    elif dispersion == "lower":
        dispersion_text = "Dispersion: source-message entropy is lower for hallucinated tokens."
    else:
        dispersion_text = "Dispersion: neither source spread nor head-role contraction is resolved."

    evidence_text = (
        "Evidence: passage messages contribute less to the observed hallucinated token."
        if evidence == "lower"
        else "Evidence: the observed-token evidence-effect difference is not resolved."
    )
    if capture == "higher":
        rate = report["summaries"]["message_independent_capture_signature"][
            "hallucinated_mean"
        ]
        evidence_text += f" The E/R-independent candidate covers {rate:.2%} of hallucinated tokens."
    return [routing_text, dispersion_text, evidence_text]


def _onset(report: dict) -> list[str]:
    first = report["matched_onset"][ONSET_METRICS[0].key]
    index = first["offset"].index(0)
    if first["events"][index] == 0:
        return ["Onset: unavailable (no matched first-hallucination events)."]

    lines = [
        "",
        "First hallucinated token (difference-in-differences at offset 0)",
        f"{'Metric':33s} {'DiD':>13s} {'95% CI':>20s} {'events':>8s} {'sources':>8s}",
    ]
    for metric in ONSET_METRICS:
        result = report["matched_onset"][metric.key]
        value = result["difference_in_difference"][index]
        value_text = _value(value, metric.scale, signed=True)
        if value is not None and metric.delta_unit:
            value_text += f" {metric.delta_unit}"
        lines.append(
            f"{metric.label:33s} {value_text:>13s} "
            f"{_ci([result['ci95_low'][index], result['ci95_high'][index]], metric.scale):>20s} "
            f"{result['events'][index]:8d} {result['sources'][index]:8d}"
        )
    return lines


def _explanation() -> list[str]:
    return [
        "",
        "Definitions",
        "edge mass = attention * ||W_O(head) V(source)||_2",
        "routing imbalance = response message share - passage message share",
        "source dispersion = normalized entropy over source-token edge mass",
        "head-role JS = JS divergence among attention-head role distributions",
        "evidence effect = observed-token logp(full) - logp(no passage messages)",
        (
            "E/R-independent candidate = full margin>0, no(E,R) margin>0, "
            "and evidence effect<=0"
        ),
        "",
        "Limits",
        "Passage means the full QA passage block, not annotated key evidence.",
        "Response share combines earlier response history and the predictor self route.",
        "Routing mass is observational; only the deletion effects are model interventions.",
        "The evidence effect concerns the observed token, not global evidence utilization.",
        "The no(E,R) branch still retains other prompt messages, residual state, and MLP updates.",
    ]


def render_report(
    report: dict,
    *,
    all_metrics: bool = False,
    explain: bool = False,
) -> str:
    """Return the report-ready audit table."""

    tokens = report["tokens"]
    positives = report["hallucinated_tokens"]
    reference = report["summaries"][KEY_METRICS[0].key]
    lines = [
        "=== Mechanism audit: QA ===",
        (
            f"{report['samples']} responses | {tokens} tokens | {positives} hallucinated "
            f"({positives / tokens:.2%})"
        ),
        (
            f"Matched analysis: {reference['sources']} sources, "
            f"{reference['matched_cells']} source-position cells; "
            "delta = hallucinated - correct; 95% source-bootstrap CI"
        ),
        "",
        f"{'Metric':33s} {'Correct':>10s} {'Halluc.':>10s} {'Delta':>13s} 95% CI",
    ]
    lines.extend(_metric_row(report, metric) for metric in KEY_METRICS)
    lines.extend(("", "Findings", *[f"- {line}" for line in _findings(report)]))
    lines.extend(_onset(report))
    lines.extend(
        (
            "",
            "Status: post-hoc labeled mechanism comparison; detector score and threshold not evaluated.",
        )
    )

    if explain:
        lines.extend(_explanation())
    if all_metrics:
        lines.extend(("", "All saved metrics"))
        for name, result in report["summaries"].items():
            lines.append(
                f"{name:42s} "
                f"C={_value(result['correct_mean'], 1.0)} "
                f"H={_value(result['hallucinated_mean'], 1.0)} "
                f"delta={_value(result['position_matched_source_equal_difference'], 1.0, signed=True)} "
                f"CI={_ci(result['ci95'], 1.0)}"
            )
    return "\n".join(lines)


__all__ = ["KEY_METRICS", "ONSET_METRICS", "render_report"]
