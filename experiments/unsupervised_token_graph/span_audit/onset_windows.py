"""All annotation onsets stay in the denominator, including unmatched onsets."""

import numpy as np

from .matching import token_kind


def normal_positions(answer, onset, window, position_gap):
    """Same answer, surface class, position band and prior-error status.

    Both the normal window and its W-token margins must have no gold error.
    This is a label-assisted reference distribution, not detector calibration.
    """
    length = len(answer.response_ids)
    positions = []
    prior_error = bool(answer.error_mask[:onset].any())
    for position in range(window + 1, length):
        if abs(position - onset) / length > position_gap:
            continue
        if answer.error_mask[max(0, position - window - 1):position + window + 1].any():
            continue
        if bool(answer.error_mask[:position].any()) != prior_error:
            continue
        if token_kind(answer, position) == token_kind(answer, onset):
            positions.append(position)
    return np.asarray(positions, dtype=int)


def onset_metadata(answer, number, span, window, references):
    start, end = answer.offsets[span.start, 0], answer.offsets[span.end - 1, 1]
    return dict(id=answer.response_id, source_id=answer.source_id, task=answer.task,
                generator=answer.generator, split=answer.split, span_index=number,
                onset=span.start, end=span.end, text=answer.text[start:end],
                answer_first=not bool(answer.error_mask[:span.start].any()),
                run_onset=span.start == 0 or not bool(answer.error_mask[span.start - 1]),
                contaminated_before=bool(answer.error_mask[max(0, span.start-window-1):span.start].any()),
                candidate_normal_windows=len(references))


def compare_window(values, onset, references, minimum_controls, quantile, minimum_gain):
    """A held-out nearest normal anchor and error anchor use identical references.

    Overlapping normal windows are correlated; percentiles are descriptive.
    An all-head maximum is compared to all-head normal maxima, never to a
    single-head threshold. The nearest anchor is selected without its score.
    """
    row = dict(gain=float(values[onset]), normal_onset=-1, normal_gain=np.nan,
               reference_windows=0, threshold=np.nan, percentile=np.nan,
               event=None, normal_event=None, status='missing_attention_rows')
    if not np.isfinite(values[onset]):
        return row
    row['status'] = 'insufficient_normal_windows'
    if not len(references):
        return row
    control = min(references, key=lambda position: (abs(position-onset), position))
    row.update(normal_onset=int(control), normal_gain=float(values[control]))
    pool = values[references[references != control]]
    pool = pool[np.isfinite(pool)]
    row['reference_windows'] = len(pool)
    if len(pool) < minimum_controls:
        return row
    threshold = max(float(np.quantile(pool, quantile, method='higher')), minimum_gain)
    row.update(threshold=threshold, percentile=float(np.mean(pool < values[onset])),
               event=bool(values[onset] > threshold), status='measured')
    if np.isfinite(values[control]):
        row['normal_event'] = bool(values[control] > threshold)
    return row


def peak_location(gain, endpoint, endpoint_gain, onset, phase, window):
    positions = np.arange(onset-window, onset) if phase == 'strict_before' else np.array([onset])
    valid = positions[(positions >= 0) & (positions < len(gain))]
    if not len(valid) or not np.isfinite(gain[valid]).all():
        return dict(peak_target=-1, peak_source=-1, peak_source_gain=np.nan)
    target = int(valid[np.argmax(gain[valid])])
    return dict(peak_target=target, peak_source=int(endpoint[target]),
                peak_source_gain=float(endpoint_gain[target]))
