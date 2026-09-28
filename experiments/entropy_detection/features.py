"""Label-free temporal uncertainty features, with explicit answer/unit boundaries."""
import numpy as np


TEMPORAL_NAMES = ('entropy', 'surprisal', 'surprisal_minus_entropy', 'previous_entropy',
    'entropy_innovation', 'entropy_z', 'past_entropy_peak4', 'past_entropy_peak8',
    'future_entropy_peak4', 'past_surprisal_peak4', 'entropy_mean4',
    'entropy_drop_from_peak', 'entropy_peak_times_surprisal', 'relative_position')


def uncertainty_features(entropy, surprisal, units, positions):
    """No gold boundaries. Gaps and natural text units stop temporal propagation."""
    entropy = np.asarray(entropy, dtype=np.float64)
    surprisal = np.asarray(surprisal, dtype=np.float64)
    result = np.empty((len(entropy), len(TEMPORAL_NAMES)), dtype=np.float32)
    start = 0
    for index in range(len(entropy)):
        if index and (units[index] != units[index-1] or positions[index] != positions[index-1]+1):
            start = index
        previous = entropy[max(start, index-8):index]
        mean = float(previous.mean()) if len(previous) else entropy[index]
        deviation = max(float(previous.std()), .25) if len(previous) else .25
        past4 = entropy[max(start, index-3):index+1]
        past8 = entropy[max(start, index-7):index+1]
        stop = index+1
        while stop < min(len(entropy), index+4):
            if units[stop] != units[index] or positions[stop] != positions[stop-1]+1:
                break
            stop += 1
        peak = float(past4.max())
        result[index] = (entropy[index], surprisal[index], surprisal[index]-entropy[index], mean,
            entropy[index]-mean, np.clip((entropy[index]-mean)/deviation, -20, 20),
            peak, float(past8.max()), float(entropy[index:stop].max()),
            float(surprisal[max(start, index-3):index+1].max()), float(past4.mean()),
            peak-entropy[index], peak*surprisal[index], positions[index]/max(1, positions[-1]))
    return result


def matrices(pack, records):
    temporal = np.empty((len(pack['target']), len(TEMPORAL_NAMES)), dtype=np.float32)
    for record in records:
        region = slice(record['packed_start'], record['packed_stop'])
        temporal[region] = uncertainty_features(pack['observations'][region, 6],
            -pack['observations'][region, 2], pack['unit_index'][region], pack['target'][region])
    static = np.column_stack((pack['context'], pack['observations'])).astype(np.float32)
    return dict(uncertainty=temporal, static=static,
                joint=np.column_stack((static, temporal)))


def feature_names(kind):
    from experiments.probabilistic_detection.data import CONTEXT, OBSERVATIONS
    names = tuple(CONTEXT) + tuple(OBSERVATIONS)
    return list(TEMPORAL_NAMES if kind == 'uncertainty' else names if kind == 'static' else names + TEMPORAL_NAMES)
