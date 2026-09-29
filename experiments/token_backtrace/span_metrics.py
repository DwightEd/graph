"""Character-union and annotated-span coverage; no span-average prediction."""

import numpy as np

from experiments.decision_risk_flow.data import read_json
from experiments.native_support.ragtruth_benchmark.data import annotations


def character_counts(text, spans, offsets, alarm):
    gold = np.zeros(len(text), dtype=bool)
    prediction = np.zeros(len(text), dtype=bool)
    for span in spans:
        gold[span['start']:span['end']] = True
    for (start, stop), selected in zip(offsets, alarm):
        if selected:
            prediction[start:stop] = True
    return np.array([np.sum(gold & prediction), np.sum(~gold & prediction),
        np.sum(gold & ~prediction), len(spans),
        sum(prediction[span['start']:span['end']].any() for span in spans),
        sum(prediction[span['start']:span['end']].all() for span in spans)], dtype=np.int64)


def span_metrics(root, metadata, pack, scores, thresholds):
    truth = annotations(root, read_json(root / 'manifest.json'), metadata['records'])
    totals = {name: np.zeros(6, dtype=np.int64) for name in scores}
    for record in metadata['records']:
        response = read_json(root / record['directory'] / 'response.json')
        region = slice(record['packed_start'], record['packed_stop'])
        offsets = np.asarray(response['offsets'])[pack['target'][region]]
        annotation = truth[record['id']]
        for name, values in scores.items():
            totals[name] += character_counts(response['text'], annotation['character_spans'],
                                            offsets, values[region] > thresholds[name])
    result = {}
    for name, count in totals.items():
        tp, fp, fn, spans, hit, covered = map(int, count)
        result[name] = dict(character_tp=tp, character_fp=fp, character_fn=fn,
            character_precision=tp / max(1, tp + fp), character_recall=tp / max(1, tp + fn),
            character_f1=2 * tp / max(1, 2 * tp + fp + fn), annotated_spans=spans,
            any_overlap_span_recall=hit / max(1, spans),
            full_character_coverage_span_recall=covered / max(1, spans),
            scope='custom character-union metrics, not official benchmark scorer; spaces included')
    return result
