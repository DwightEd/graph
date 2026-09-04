"""Population summaries for the registered re-anchor mechanism stages."""

from __future__ import annotations

import numpy as np

from .events import match_events, match_onsets
from .stats import curve_ci, mean_ci, source_means

BROAD_SIGNALS = (
    "prompt_delta",
    "nonlocal_delta",
    "evidence_delta",
    "route_change",
    "predictor_reuse",
    "emitted_token_anchor",
)
MECHANISM_SIGNALS = (
    "evidence_effect",
    "other_prompt_effect",
    "evidence_prompt_interaction",
    "history_effect",
    "evidence_readout_gain",
    "evidence_late_control_loss",
    "context_distribution_js",
    "context_target_logprob_gain",
    "context_adoption_margin",
    "context_target_log_rank",
)


def interval_direction(summary: dict, expected: str) -> str:
    """Classify a preregistered sign without consulting its point estimate."""

    low, high = summary.get("ci95", (None, None))
    if low is None or high is None:
        return "inconclusive"
    if expected == "positive":
        if low > 0:
            return "supported"
        if high < 0:
            return "contradicted"
    elif expected == "negative":
        if high < 0:
            return "supported"
        if low > 0:
            return "contradicted"
    else:
        raise ValueError("expected direction must be positive or negative")
    return "inconclusive"


def matched_onsets(row: dict) -> list[tuple[int, int]]:
    prediction = np.asarray(row["prediction_position"], dtype=np.int64)
    boundary = np.isin(
        prediction,
        np.asarray(row["sentence_boundary_position"], dtype=np.int64),
    )
    relative_position = np.linspace(0.0, 1.0, len(prediction))
    return match_onsets(
        row["label"],
        row["target_token_id"],
        covariates={
            "relative_position": relative_position,
            "entropy": np.asarray(row["baseline_entropy"], dtype=np.float64),
            "target_logprob": np.asarray(
                row["baseline_target_logprob"], dtype=np.float64
            ),
        },
        boundary=boundary,
    )


def matched_transition_events(row: dict) -> list[tuple[int, int]]:
    prediction = np.asarray(row["prediction_position"], dtype=np.int64)
    boundary = np.isin(
        prediction,
        np.asarray(row["sentence_boundary_position"], dtype=np.int64),
    )
    return match_events(
        row["transition_peak"],
        row["target_token_id"],
        covariates={
            "relative_position": np.linspace(0.0, 1.0, len(prediction)),
            "entropy": np.asarray(row["baseline_entropy"], dtype=np.float64),
            "target_logprob": np.asarray(
                row["baseline_target_logprob"], dtype=np.float64
            ),
        },
        boundary=boundary,
    )


def matching_balance(
    rows: list[dict],
    *,
    bootstrap: int,
    seed: int,
    transition: bool = False,
) -> dict:
    fields = {
        "relative_position_gap": ([], []),
        "entropy_gap": ([], []),
        "target_logprob_gap": ([], []),
    }
    boundary_match, token_match, pairs = [], [], 0
    match_sources: list[str] = []
    for row in rows:
        source = row["source_id"]
        prediction = np.asarray(row["prediction_position"], dtype=np.int64)
        relative = np.linspace(0.0, 1.0, len(prediction))
        entropy = np.asarray(row["baseline_entropy"], dtype=np.float64)
        logprob = np.asarray(row["baseline_target_logprob"], dtype=np.float64)
        token = np.asarray(row["target_token_id"])
        boundary = np.isin(
            prediction,
            np.asarray(row["sentence_boundary_position"], dtype=np.int64),
        )
        pairs_for_row = (
            matched_transition_events(row) if transition else matched_onsets(row)
        )
        for onset, clean in pairs_for_row:
            pairs += 1
            match_sources.append(source)
            for name, values in (
                ("relative_position_gap", relative),
                ("entropy_gap", entropy),
                ("target_logprob_gap", logprob),
            ):
                fields[name][0].append(abs(float(values[onset] - values[clean])))
                fields[name][1].append(source)
            boundary_match.append(float(boundary[onset] == boundary[clean]))
            token_match.append(float(token[onset] == token[clean]))
    return {
        "pairs": pairs,
        "sources": len(set(match_sources)),
        **{
            f"mean_absolute_{name}": mean_ci(
                values,
                sources,
                repeats=bootstrap,
                seed=seed + index,
            )
            for index, (name, (values, sources)) in enumerate(fields.items())
        },
        "boundary_match_fraction": mean_ci(
            boundary_match,
            match_sources,
            repeats=bootstrap,
            seed=seed + 10,
        ),
        "token_match_fraction": mean_ci(
            token_match,
            match_sources,
            repeats=bootstrap,
            seed=seed + 11,
        ),
    }


def centered(values, center: int, radius: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    if center - radius < 0 or center + radius >= len(values):
        return None
    return values[center - radius : center + radius + 1]


def slope(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return float("nan")
    position = np.linspace(0.0, 1.0, len(values))
    return float(np.polyfit(position[finite], values[finite], 1)[0])


def circular_event_lift(signal, event) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if len(signal) < 3 or not event.any():
        return float("nan")
    actual = np.nanmean(signal[event])
    null = []
    for shift in range(1, len(event)):
        selected = np.roll(event, shift)
        value = np.nanmean(signal[selected])
        if np.isfinite(value):
            null.append(value)
    return float(actual - np.mean(null)) if null and np.isfinite(actual) else float("nan")


def coupling_population(rows: list[dict], kind: str, *, bootstrap: int, seed: int) -> dict:
    peak_name = f"{kind}_peak"
    paired_name = f"{kind}_paired_anchor"
    rate_name = f"{kind}_coupling_rate"
    null_name = f"{kind}_coupling_null_rate"
    lag_name = f"{kind}_median_anchor_lag"
    peaks = coupled = 0
    weighted_null = 0.0
    null_weight = 0
    sample_lift, sample_source, lags = [], [], []
    for row in rows:
        peak = np.asarray(row[peak_name], dtype=bool)
        paired = np.asarray(row[paired_name], dtype=np.int64)
        count = int(peak.sum())
        peaks += count
        coupled += int((paired >= 0).sum())
        null = float(row[null_name])
        rate = float(row[rate_name])
        if count and np.isfinite(null):
            weighted_null += count * null
            null_weight += count
        if np.isfinite(rate) and np.isfinite(null):
            sample_lift.append(rate - null)
            sample_source.append(row["source_id"])
        lag = float(row[lag_name])
        if np.isfinite(lag):
            lags.append(lag)
    pooled = coupled / peaks if peaks else None
    null = weighted_null / null_weight if null_weight else None
    grouped = source_means(sample_lift, sample_source)
    return {
        "event_peaks": peaks,
        "coupled_events": coupled,
        "pooled_rate": pooled,
        "pooled_null": null,
        "pooled_lift": None if pooled is None or null is None else pooled - null,
        "sample_lift": mean_ci(sample_lift, sample_source, repeats=bootstrap, seed=seed),
        "positive_source_fraction": float(np.mean(grouped > 0)) if len(grouped) else None,
        "median_anchor_lag": float(np.median(lags)) if lags else None,
    }


def normal_summary(rows: list[dict], *, bootstrap: int, seed: int) -> dict:
    correct = [row for row in rows if not np.asarray(row["label"], dtype=bool).any()]
    slopes = {
        name: []
        for name in (
            "prompt_lift",
            "history_lift",
            "conditional_prompt_history_log_odds",
            "raw_prompt_share",
            "raw_history_share",
        )
    }
    slope_sources = {name: [] for name in slopes}
    transition = {
        name: []
        for name in (
            "prompt_delta",
            "evidence_delta",
            "predictor_reuse",
            "emitted_token_anchor",
        )
    }
    transition_sources = {name: [] for name in transition}
    transition_peaks = 0
    transition_pairs = 0
    for row in correct:
        source = row["source_id"]
        candidates = {
            "prompt_lift": row["prompt_lift"],
            "history_lift": row["history_lift"],
            "conditional_prompt_history_log_odds": (
                np.asarray(row["prompt_lift"], dtype=np.float64)
                - np.asarray(row["history_lift"], dtype=np.float64)
            ),
            "raw_prompt_share": row["prompt_share"],
            "raw_history_share": row["history_share"],
        }
        for name, values in candidates.items():
            value = slope(values)
            if np.isfinite(value):
                slopes[name].append(value)
                slope_sources[name].append(source)
        peak = np.asarray(row["transition_peak"], dtype=bool)
        transition_peaks += int(peak.sum())
        pairs = matched_transition_events(row)
        transition_pairs += len(pairs)
        for event, clean in pairs:
            for name in transition:
                series = np.asarray(row[name], dtype=np.float64)
                value = series[event] - series[clean]
                if np.isfinite(value):
                    transition[name].append(float(value))
                    transition_sources[name].append(source)
    direct_route_shift = {
        f"{name}_slope": mean_ci(
            values,
            slope_sources[name],
            repeats=bootstrap,
            seed=seed + index,
        )
        for index, (name, values) in enumerate(slopes.items())
    }
    return {
        "correct_samples": len(correct),
        "direct_route_shift": direct_route_shift,
        "internal_transition": {
            "peaks": transition_peaks,
            "matched_pairs": transition_pairs,
            **{
                name: mean_ci(
                    transition[name],
                    transition_sources[name],
                    repeats=bootstrap,
                    seed=seed + 10 + index,
                )
                for index, name in enumerate(transition)
            },
        },
    }


def mechanism_summary(rows: list[dict], *, bootstrap: int, seed: int) -> dict:
    rows = [row for row in rows if row.get("mechanism")]
    effects = {name: [] for name in ("evidence_entry", *MECHANISM_SIGNALS)}
    sources = {name: [] for name in effects}
    curves = {
        "entry_onset": [],
        "entry_clean": [],
        "presence_onset": [],
        "presence_clean": [],
        "control_onset": [],
        "control_clean": [],
    }
    curve_sources = {name: [] for name in curves}
    pairs = 0
    pair_sources: set[str] = set()
    layer_count = 0
    for row in rows:
        source = row["source_id"]
        label = np.asarray(row["label"], dtype=bool)
        entry = np.nanmean(np.asarray(row["evidence_share_layer"], dtype=float), axis=0)
        presence = np.asarray(row["evidence_state_presence"], dtype=float)
        control = np.asarray(row["evidence_state_control"], dtype=float)
        layer_count = max(layer_count, presence.shape[0])
        for onset, clean in matched_onsets(row):
            pairs += 1
            pair_sources.add(source)
            values = {"evidence_entry": entry}
            values.update(
                {
                    name: np.asarray(row[name], dtype=float)
                    for name in MECHANISM_SIGNALS
                    if name in row
                }
            )
            for name, series in values.items():
                difference = series[onset] - series[clean]
                if np.isfinite(difference):
                    effects[name].append(float(difference))
                    sources[name].append(source)
            candidate = {
                "entry_onset": np.asarray(row["evidence_share_layer"], dtype=float)[:, onset],
                "entry_clean": np.asarray(row["evidence_share_layer"], dtype=float)[:, clean],
                "presence_onset": presence[:, onset],
                "presence_clean": presence[:, clean],
                "control_onset": control[:, onset],
                "control_clean": control[:, clean],
            }
            for name, curve in candidate.items():
                if np.isfinite(curve).all():
                    curves[name].append(curve)
                    curve_sources[name].append(source)
    return {
        "samples": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "onset_pairs": pairs,
        "onset_pair_sources": len(pair_sources),
        "onset_minus_clean": {
            name: mean_ci(
                effects[name], sources[name], repeats=bootstrap, seed=seed + index
            )
            for index, name in enumerate(effects)
        },
        "layer": list(range(layer_count)),
        "layer_curves": {
            name: curve_ci(
                curves[name], curve_sources[name], repeats=bootstrap, seed=seed + 30 + index
            )
            for index, name in enumerate(curves)
        },
    }


def task_summary(rows: list[dict], *, bootstrap: int, seed: int, radius: int) -> dict:
    tokens = sum(len(row["label"]) for row in rows)
    positives = sum(int(np.asarray(row["label"], dtype=bool).sum()) for row in rows)
    broad_effect = {name: [] for name in BROAD_SIGNALS}
    broad_source = {name: [] for name in BROAD_SIGNALS}
    onset_pairs = 0
    onset_curves, clean_curves, curve_sources = [], [], []
    for row in rows:
        source = row["source_id"]
        label = np.asarray(row["label"], dtype=bool)
        pairs = matched_onsets(row)
        onset_pairs += len(pairs)
        for onset, clean in pairs:
            for name in BROAD_SIGNALS:
                series = np.asarray(row[name], dtype=float)
                difference = series[onset] - series[clean]
                if np.isfinite(difference):
                    broad_effect[name].append(float(difference))
                    broad_source[name].append(source)
            positive = centered(row["evidence_delta"], onset, radius)
            negative = centered(row["evidence_delta"], clean, radius)
            if positive is not None and negative is not None:
                if np.isfinite(positive).all() and np.isfinite(negative).all():
                    onset_curves.append(positive)
                    clean_curves.append(negative)
                    curve_sources.append(source)
    offset = np.arange(-radius, radius + 1)
    correct_rows = [
        row for row in rows if not np.asarray(row["label"], dtype=bool).any()
    ]
    normal = normal_summary(rows, bootstrap=bootstrap, seed=seed)
    shift = normal["direct_route_shift"]
    transition = normal["internal_transition"]
    onset_summary = {
        name: mean_ci(
            broad_effect[name], broad_source[name], repeats=bootstrap, seed=seed + 60 + index
        )
        for index, name in enumerate(BROAD_SIGNALS)
    }
    decisions = {
        "H0_prompt_lift_decreases": interval_direction(
            shift["prompt_lift_slope"], "negative"
        ),
        "H0_history_lift_increases": interval_direction(
            shift["history_lift_slope"], "positive"
        ),
        "H1_generic_prompt_reentry": interval_direction(
            transition["prompt_delta"], "positive"
        ),
        "H1_evidence_reentry": interval_direction(
            transition["evidence_delta"], "positive"
        ),
        "H1_predictor_state_reuse": interval_direction(
            transition["predictor_reuse"], "positive"
        ),
        "H1_emitted_token_future_coupling": interval_direction(
            transition["emitted_token_anchor"], "positive"
        ),
        "H2_missed_evidence_entry": interval_direction(
            onset_summary["evidence_delta"], "negative"
        ),
        "H2_predictor_state_reuse": interval_direction(
            onset_summary["predictor_reuse"], "positive"
        ),
        "H2_emitted_token_anchor_association": interval_direction(
            onset_summary["emitted_token_anchor"], "positive"
        ),
    }
    return {
        "samples": len(rows),
        "tokens": tokens,
        "positive_tokens": positives,
        "prevalence": positives / tokens if tokens else None,
        "onset_pairs": onset_pairs,
        "normal": normal,
        "prompt_to_anchor": coupling_population(rows, "prompt", bootstrap=bootstrap, seed=seed + 50),
        "nonlocal_to_anchor": coupling_population(rows, "review", bootstrap=bootstrap, seed=seed + 51),
        "onset_minus_matched_clean": onset_summary,
        "matching_balance": matching_balance(
            rows, bootstrap=bootstrap, seed=seed + 70
        ),
        "transition_matching_balance": matching_balance(
            correct_rows,
            bootstrap=bootstrap,
            seed=seed + 71,
            transition=True,
        ),
        "registered_decisions": decisions,
        "onset_evidence_curve": {
            "offset": offset.tolist(),
            "hallucination": curve_ci(
                onset_curves, curve_sources, repeats=bootstrap, seed=seed + 80
            ),
            "matched_clean": curve_ci(
                clean_curves, curve_sources, repeats=bootstrap, seed=seed + 81
            ),
        },
        "mechanism": mechanism_summary(rows, bootstrap=bootstrap, seed=seed + 100),
    }
