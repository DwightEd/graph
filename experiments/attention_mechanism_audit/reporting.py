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


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _count(value: int | None) -> str:
    return "?" if value is None else str(value)


def _estimate(report: dict, metric: Metric) -> str:
    result = report["summaries"][metric.key]
    delta = result["position_matched_source_equal_difference"]
    delta_text = _value(delta, metric.scale, signed=True)
    if delta is not None and metric.delta_unit:
        delta_text += f" {metric.delta_unit}"
    return f"{delta_text} {_ci(result['ci95'], metric.scale)}"


def _p_value(report: dict, metric: Metric) -> str:
    result = report["summaries"][metric.key]
    adjusted = result.get("holm_p_value")
    value = adjusted if adjusted is not None else result.get("p_value")
    if value is None:
        return "n/a"
    prefix = "Holm " if adjusted is not None else "raw "
    return f"{prefix}{value:.4g}"


def _onset(report: dict) -> list[str]:
    first = report["matched_onset"][ONSET_METRICS[0].key]
    index = first["offset"].index(0)
    if first["events"][index] == 0:
        return ["Onset: unavailable (no matched first-hallucination events)."]

    lines = [
        "",
        "Hallucination-span onset (within-response difference-in-differences at offset 0)",
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
    coverage = report.get("coverage", {})
    lines = [
        "=== QA mechanism audit ===",
        (
            f"{report['samples']} responses | {tokens} tokens | {positives} hallucinated "
            f"({positives / tokens:.2%})"
        ),
        (
            f"Coverage: evaluated {coverage.get('evaluated', report['samples'])}/"
            f"{_count(coverage.get('eligible_qa', report['samples']))} eligible QA; "
            f"matched {reference.get('matched_samples', 0)} mixed responses, "
            f"{reference['sources']} sources, {reference['matched_cells']} position cells; "
            f"{reference.get('hallucinated_token_coverage', 0.0):.1%} of hallucinated tokens"
        ),
    ]
    if "by_split" in report:
        split_names = [name for name in ("train", "test") if name in report["by_split"]]
        split_names.extend(name for name in report["by_split"] if name not in split_names)
        lines.append(
            "Splits: "
            + ", ".join(
                f"{name} {report['by_split'][name].get('coverage', {}).get('evaluated', report['by_split'][name]['samples'])}/"
                f"{_count(report['by_split'][name].get('coverage', {}).get('eligible_qa', report['by_split'][name]['samples']))}"
                for name in split_names
            )
        )
        lines.append(
            f"Train/test source overlap: {len(report.get('source_overlap_between_splits', []))}"
        )
        lines.extend(("", "Within-response matched effects (hallucinated - correct)"))
        header = f"{'Metric':30s}" + "".join(f" {name:>31s}" for name in split_names) + f" {'all':>31s}"
        lines.append(header)
        for metric in KEY_METRICS:
            lines.append(
                f"{metric.label:30s}"
                + "".join(
                    f" {_estimate(report['by_split'][name], metric):>31s}"
                    for name in split_names
                )
                + f" {_estimate(report, metric):>31s}"
            )
        lines.append("")
        lines.append("Primary endpoint sign-flip p-values (Holm-adjusted within each report)")
        primary = set(report["statistical_design"]["primary_endpoints"])
        for metric in (metric for metric in KEY_METRICS if metric.key in primary):
            values = ", ".join(
                f"{name}={_p_value(report['by_split'][name], metric)}"
                for name in split_names
            )
            lines.append(f"{metric.label}: {values}, all={_p_value(report, metric)}")
    else:
        lines.extend(
            (
                "",
                "Within-response matched effects (hallucinated - correct)",
                f"{'Metric':33s} {'Delta [95% CI]':>32s} {'p':>12s}",
            )
        )
        for metric in KEY_METRICS:
            lines.append(
                f"{metric.label:33s} {_estimate(report, metric):>32s} "
                f"{_p_value(report, metric):>12s}"
            )
    lines.extend(_onset(report))
    observer = report.get("observer_readout")
    if observer:
        lines.extend(
            (
                "",
                (
                    "Observer target preference (correct/hallucinated): "
                    f"{_percent(observer['target_preferred_correct'])}/"
                    f"{_percent(observer['target_preferred_hallucinated'])}"
                ),
                (
                    "E/R-independent capture conditional on target preference "
                    "(correct/hallucinated): "
                    f"{_percent(observer['capture_given_preferred_correct'])}/"
                    f"{_percent(observer['capture_given_preferred_hallucinated'])}"
                ),
            )
        )
    lines.extend(
        (
            "",
            f"Per-sample audits: {report.get('sample_audits', {}).get('count', 0)} JSON files and figures.",
            "Scope: post-hoc mechanism audit; no detector score or threshold is evaluated.",
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


def render_sample(record: dict) -> str:
    """Print the token/onset evidence for one response without population prose."""

    lines = [
        f"=== Sample {record['sample_id']} ({record.get('split', 'unknown')}) ===",
        (
            f"{len(record['label'])} response tokens | "
            f"{record['hallucinated_tokens']} hallucinated "
            f"({record['hallucinated_fraction']:.2%}) | "
            f"{len(record['onsets'])} spans"
        ),
        "",
        "Sample means",
    ]
    for label, key in (
        ("routing imbalance", "message_routing_drift_mean"),
        ("source dispersion", "message_source_dispersion_mean"),
        ("head-role JS", "head_role_disagreement_mean"),
        ("evidence effect", "evidence_message_effect"),
        ("response effect", "response_message_effect"),
    ):
        value = record["summary"][key]
        lines.append(
            f"{label:20s} all={_value(value['all'], 1.0)} "
            f"correct={_value(value['correct'], 1.0)} "
            f"hallucinated={_value(value['hallucinated'], 1.0)}"
        )
    for onset in record["onsets"]:
        changes = onset["changes_from_previous_token"]
        lines.extend(
            (
                "",
                (
                    f"Onset token {onset['start']}: {onset['token']!r} | "
                    f"span={onset['span_text']!r}"
                ),
                (
                    "change from previous token: "
                    f"routing={_value(changes['message_routing_drift_mean'], 1.0, signed=True)}, "
                    f"dispersion={_value(changes['message_source_dispersion_mean'], 1.0, signed=True)}, "
                    f"evidence effect={_value(changes['evidence_message_effect'], 1.0, signed=True)}"
                ),
                (
                    f"at onset: evidence effect={onset['evidence_effect']:+.4f}, "
                    f"response effect={onset['response_effect']:+.4f}, "
                    f"full margin={onset['full_margin']:+.4f}"
                ),
                "top late-layer source routes:",
            )
        )
        for source in onset["top_late_sources"][:5]:
            lines.append(
                f"  s={source['source_index']:4d} {source['role']:12s} "
                f"{source['token']!r} retained/total="
                f"{source['late_retained_mass_over_total']:.2%}"
            )
        lines.append("top late-layer head-role routes:")
        for route in onset["top_late_head_routes"][:5]:
            lines.append(
                f"  L{route['layer']:02d} H{route['head']:02d} "
                f"{route['role']:12s} magnitude={route['edge_magnitude']:.4f}"
            )
    return "\n".join(lines)


__all__ = ["KEY_METRICS", "ONSET_METRICS", "render_report", "render_sample"]
