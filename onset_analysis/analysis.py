"""Audit factual-choice instability and pre-onset lookback jointly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from onset_analysis.statistics import summarize_onsets
from onset_analysis.traces import (
    load_audit_manifest,
    load_labels,
    load_trace,
    trace_path,
)

PAIR_SCHEMA = "onset-analysis/pair@1"


@dataclass(frozen=True)
class OnsetChoiceConfig:
    audit_root: Path
    output_dir: Path
    pre_window: int = 3
    match_window: int = 64
    bootstrap: int = 200
    seed: int = 20260910

    def __post_init__(self) -> None:
        if min(self.pre_window, self.match_window) < 1 or self.bootstrap < 0:
            raise ValueError(
                "windows must be positive and bootstrap must be non-negative"
            )


class OnsetChoiceAudit:
    """Compare each content-bearing hallucination onset with a local normal token."""

    def __init__(self, config: OnsetChoiceConfig, *, progress: bool = True) -> None:
        self.config = config
        self.progress = progress

    def run(self) -> dict:
        root = self.config.audit_root.resolve()
        entries = load_audit_manifest(root)
        output = self.config.output_dir
        if output.exists():
            raise FileExistsError(f"onset-choice output already exists: {output}")
        output.mkdir(parents=True)

        pairs = []
        processed = skipped = spans = content_onsets = 0
        coverage = {
            "first_error_spans": 0,
            "first_errors_without_content": 0,
            "first_errors_without_prior_predictor": 0,
            "unknown_label_tokens": 0,
        }
        hallucination_tokens = 0
        iterator = tqdm(
            entries,
            desc="audit factual onsets",
            unit="sample",
            disable=not self.progress,
        )
        for entry in iterator:
            sample_path = trace_path(root, entry["path"])
            label_path = sample_path.with_suffix(".labels.npz")
            if not sample_path.is_file():
                skipped += 1
                continue
            if not label_path.is_file():
                raise FileNotFoundError(f"label sidecar does not exist: {label_path}")
            trace = load_trace(sample_path, entry)
            labels = load_labels(label_path, entry["response_tokens"])
            candidates, sample_spans = _content_onsets(
                labels, trace["token_text"], trace["special_targets"]
            )
            spans += sample_spans
            content_onsets += len(candidates)
            hallucination_tokens += int((labels == 1).sum())
            coverage["unknown_label_tokens"] += int((labels == -1).sum())
            if sample_spans:
                first_error = int(np.flatnonzero(labels == 1)[0])
                coverage["first_error_spans"] += 1
                first_content = next(
                    (index for start, index, _ in candidates if start == first_error),
                    None,
                )
                coverage["first_errors_without_content"] += first_content is None
                coverage["first_errors_without_prior_predictor"] += first_content == 0
            pairs.extend(self._match_sample(entry, trace, labels, candidates))
            processed += 1
            iterator.set_postfix(onsets=content_onsets, pairs=len(pairs))

        if not pairs:
            raise ValueError(
                "no measurable hallucination onset has a matched normal control"
            )
        event_path = output / "events.jsonl"
        event_path.write_text(
            "".join(
                json.dumps(pair, sort_keys=True, allow_nan=False) + "\n"
                for pair in pairs
            ),
            encoding="utf-8",
        )
        report = summarize_onsets(
            pairs,
            processed=processed,
            skipped=skipped,
            spans=spans,
            content_onsets=content_onsets,
            pre_window=self.config.pre_window,
            match_window=self.config.match_window,
            bootstrap=self.config.bootstrap,
            seed=self.config.seed,
            event_path=event_path,
            coverage=coverage,
            hallucination_tokens=hallucination_tokens,
        )
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    def _match_sample(
        self,
        entry: dict,
        trace: dict,
        labels: np.ndarray,
        onsets: list[tuple[int, int, int]],
    ) -> list[dict]:
        used_controls: set[int] = set()
        pairs = []
        first_error = (
            int(np.flatnonzero(labels == 1)[0]) if np.any(labels == 1) else None
        )
        for span_start, response_index, span_length in onsets:
            onset = _measure(trace, response_index, self.config.pre_window)
            if onset is None:
                continue
            token_class = onset["token_class"]
            candidates = [
                index
                for index, label in enumerate(labels)
                if label == 0
                and index not in used_controls
                and 0 < index
                and abs(index - response_index) <= self.config.match_window
                and np.all(
                    # The preceding predictor supplies the baseline for the first delta.
                    labels[max(0, index - self.config.pre_window) : index + 1] == 0
                )
                and _token_class(trace["token_text"][index]) == token_class
                and not trace["special_targets"][index]
            ]
            candidates.sort(key=lambda index: (abs(index - response_index), index))
            control = None
            control_index = -1
            for candidate in candidates:
                control = _measure(trace, candidate, self.config.pre_window)
                if control is not None:
                    control_index = candidate
                    break
            if control is None:
                continue
            used_controls.add(control_index)
            onset["span_length"] = span_length
            pairs.append(
                {
                    "schema": PAIR_SCHEMA,
                    "pair_id": (
                        f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}"
                        f":{response_index}"
                    ),
                    "sample_id": entry["sample_id"],
                    "source_id": entry["source_id"],
                    "split": entry["split"],
                    "task_type": entry["task_type"],
                    "span_start": span_start,
                    "is_first_error": span_start == first_error,
                    "match_distance": abs(control_index - response_index),
                    "onset": onset,
                    "control": control,
                }
            )
        return pairs


def _content_onsets(
    labels: np.ndarray, token_text: np.ndarray, special: np.ndarray
) -> tuple[list[tuple[int, int, int]], int]:
    events = []
    spans = 0
    index = 0
    while index < len(labels):
        if labels[index] != 1:
            index += 1
            continue
        stop = index + 1
        while stop < len(labels) and labels[stop] == 1:
            stop += 1
        spans += 1
        content = next(
            (
                position
                for position in range(index, stop)
                if not special[position]
                and _token_class(token_text[position]) is not None
            ),
            None,
        )
        if content is not None:
            events.append((index, content, stop - index))
        index = stop
    return events, spans


def _token_class(value: str) -> str | None:
    text = str(value).strip()
    if not text or not any(character.isalnum() for character in text):
        return None
    has_letter = any(character.isalpha() for character in text)
    has_digit = any(character.isdigit() for character in text)
    if has_letter and has_digit:
        return "alphanumeric"
    return "word" if has_letter else "number"


def _measure(trace: dict, response_index: int, pre_window: int) -> dict | None:
    if response_index < 1:
        return None
    margin = float(trace["observed_margin"][response_index])
    predictor_entropy = float(trace["predictor_entropy"][response_index])
    if not np.isfinite(margin) or not np.isfinite(predictor_entropy):
        return None
    groups = trace["group_index"]
    share = trace["share"]
    first = max(1, response_index - pre_window + 1)
    best = None
    for query in range(first, response_index + 1):
        evidence = (
            share[..., query, groups["evidence"]]
            - share[..., query - 1, groups["evidence"]]
        )
        far = (
            share[..., query, groups["history_far"]]
            - share[..., query - 1, groups["history_far"]]
        )
        remote = evidence + far
        if not np.isfinite(remote).any():
            continue
        candidate = (float(np.nanmean(remote)), query, evidence, far, remote)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    remote_gain, query, evidence, far, remote = best
    local_names = ("history_local", "self")
    local_loss = sum(
        share[..., query - 1, groups[name]] - share[..., query, groups[name]]
        for name in local_names
    )
    entropy_gain = (
        trace["attention_entropy"][..., query]
        - trace["attention_entropy"][..., query - 1]
    )
    concentration_gain = trace["top1"][..., query] - trace["top1"][..., query - 1]
    evidence_gain = float(np.nanmean(evidence))
    far_gain = float(np.nanmean(far))
    dispersion_gain = float(np.nanmean(entropy_gain - concentration_gain))
    if remote_gain <= 0:
        mode = "local_persistence"
    elif dispersion_gain > 0:
        mode = "diffuse_global"
    elif evidence_gain >= far_gain:
        mode = "focused_evidence"
    else:
        mode = "focused_far_history"
    return {
        "response_index": response_index,
        "token_id": int(trace["token_ids"][response_index]),
        "token_text": str(trace["token_text"][response_index]),
        "token_class": _token_class(trace["token_text"][response_index]),
        "instability": -margin,
        "predictor_entropy": predictor_entropy,
        "lookback_mode": mode,
        "peak_offset": query - response_index,
        "remote_gain": remote_gain,
        "evidence_gain": evidence_gain,
        "far_history_gain": far_gain,
        "local_loss": float(np.nanmean(local_loss)),
        "dispersion_gain": dispersion_gain,
        "head_remote_fraction": float(np.nanmean(remote > 0)),
    }
