"""Join labels only after observations are saved. This module does not select heads."""

import numpy as np


def token_spans(answer: dict) -> list[tuple[int, int]] | None:
    if answer["labels"] is None or answer["response_offsets"] is None:
        return None
    offsets = np.asarray(answer["response_offsets"])
    ordinary = ~np.isin(answer["response_ids"], answer["special_token_ids"])
    spans = []
    for label in answer["labels"]:
        hits = np.flatnonzero(
            (offsets[:, 0] < label["end"])
            & (offsets[:, 1] > label["start"])
            & (offsets[:, 1] > offsets[:, 0])
            & ordinary
        )
        if len(hits):
            for part in np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1):
                spans.append((int(part[0]), int(part[-1] + 1)))
    merged = []
    for start, end in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def repetition(ids: list[int]) -> float:
    return 1 - len(set(ids)) / len(ids)


def surface_type(answer: dict, token: int) -> str:
    start, end = answer["response_offsets"][token]
    text = answer["response"][start:end].strip()
    if text and text[0].isalpha():
        return "letter"
    if text and text[0].isdigit():
        return "number"
    return "other"


def match_normal(answer: dict, spans: list[tuple[int, int]], window: int) -> list:
    ids = answer["response_ids"]
    error = np.zeros(len(ids), dtype=bool)
    for start, end in spans:
        error[start:end] = True
    available = ~np.isin(ids, answer["special_token_ids"])
    matches = []
    for start, end in spans:
        if start < 2 * window:
            matches.append(None)
            continue
        length = end - start
        candidates = []
        for left in range(2 * window, len(ids) - length + 1):
            right = left + length
            if error[left - window : min(len(ids), right + window)].any():
                continue
            if not available[left:right].all() or abs(left - start) / len(ids) > 0.25:
                continue
            if surface_type(answer, left) != surface_type(answer, start):
                continue
            if bool(error[:left].any()) != bool(error[:start].any()):
                continue
            if abs(repetition(ids[left:right]) - repetition(ids[start:end])) <= 0.15:
                candidates.append((abs(left - start), left, right))
        match = min(candidates)[1:] if candidates else None
        matches.append(match)
        if match is not None:
            available[match[0] : match[1]] = False
    return matches


def span_row(answer: dict, nodes: list[dict], span: tuple, kind: str, pair: int, window: int):
    start, end = span
    targets = {node["target"] for node in nodes}
    before = [token for token in targets if max(0, start - window) <= token < start]
    inside = [token for token in targets if start <= token < end]
    offsets = answer["response_offsets"]
    observed = max(0, start - max(window, start - window)) if answer["evidence"] else 0
    return dict(
        pair=pair,
        kind=kind,
        start=start,
        end=end,
        length=end - start,
        text=answer["response"][offsets[start][0] : offsets[end - 1][1]],
        pre_onset_observable=observed == window,
        pre_onset_queries_observed=observed,
        onset_observable=start >= window and bool(answer["evidence"]),
        nodes_before=len(before),
        node_at_onset=start in targets,
        nodes_inside=len(inside),
        first_inside_delay=min(inside) - start if inside else None,
    )


def label_report(answer: dict, nodes: list[dict], window: int) -> tuple[list, dict]:
    spans = token_spans(answer)
    if spans is None:
        return [], dict(label_status="unavailable", error_tokens=None, continuation_tokens=None)
    rows, matched = [], 0
    controls = match_normal(answer, spans, window)
    for index, (span, control) in enumerate(zip(spans, controls)):
        rows.append(span_row(answer, nodes, span, "hallucination", index, window))
        if control is not None:
            rows.append(span_row(answer, nodes, control, "normal_control", index, window))
            matched += 1
    count = sum(end - start for start, end in spans)
    summary = dict(
        label_status="available",
        error_tokens=count,
        spans=len(spans),
        continuation_tokens=count - len(spans),
        matched_spans=matched,
        continuation_fraction=(count - len(spans)) / count if count else None,
    )
    return rows, summary
