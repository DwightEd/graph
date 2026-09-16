"""整段评分、混合校准、非重叠区间动态规划；不把边际分数称为真假概率。"""

import numpy as np
import torch

from .data import DetectionResult


def score_spans(model, graph, spans, batch_size=256):
    model.eval()
    with torch.no_grad():
        encoded = model.encode_graph(graph)
        batches = [(-model.score_encoded(encoded, spans[start:start + batch_size])).cpu().numpy()
                   for start in range(0, len(spans), batch_size)]
    return np.concatenate(batches) if batches else np.empty(0, np.float32)


def fit_unlabeled_reference(records, tail_quantile=.95):
    """每个 source/长度最多32个等距样本；校准集混有未知错误，不筛正确样本。"""
    pools = {}
    for group, source, lengths, scores in records:
        for length in np.unique(lengths):
            values = scores[lengths == length]
            values = values[np.linspace(0, len(values) - 1, min(32, len(values)), dtype=int)]
            pools.setdefault((group, int(length)), {}).setdefault(source, []).append(values)
    reference = {}
    for (group, length), sources in pools.items():
        values, weights = [], []
        for arrays in sources.values():
            scores = np.concatenate(arrays)
            values.extend(scores.tolist())
            weights.extend([1 / len(scores)] * len(scores))
        order = np.argsort(values)
        values = np.asarray(values)[order]
        cumulative = np.cumsum(np.asarray(weights)[order])
        cumulative /= cumulative[-1]
        quantiles = values[np.searchsorted(cumulative, [.25, .75, tail_quantile])]
        reference.setdefault(group, {})[str(length)] = dict(
            center=float(quantiles[2]), scale=max(float(quantiles[1] - quantiles[0]), 1e-3),
            sources=len(sources), observations=len(values))
    if not reference:
        raise ValueError('No covered calibration spans')
    return dict(groups=reference, quantile=tail_quantile,
                meaning='unlabeled mixed-source anomaly budget; not a false-positive guarantee')


def interval_inference(length, bounds, potentials):
    """Positive runs cannot touch: each labeling has one decomposition, not many segmentations."""
    ending = [[] for _ in range(length + 1)]
    starting = [[] for _ in range(length)]
    for index, (start, end) in enumerate(bounds):
        ending[int(end)].append(index)
        starting[int(start)].append(index)

    forward = np.full(length + 1, -np.inf)
    best = np.zeros(length + 1)
    choice = np.full(length + 1, -1, int)
    forward[0] = 0.
    for end in range(1, length + 1):
        forward[end] = forward[end - 1]
        best[end] = best[end - 1]
        for index in ending[end]:
            start = int(bounds[index, 0])
            previous = max(0, start - 1)
            value = forward[previous] + potentials[index]
            forward[end] = np.logaddexp(forward[end], value)
            value = best[previous] + potentials[index]
            if value > best[end]:
                best[end], choice[end] = value, index

    backward = np.full(length + 1, -np.inf)
    backward[length] = 0.
    for start in range(length - 1, -1, -1):
        backward[start] = backward[start + 1]
        for index in starting[start]:
            end = int(bounds[index, 1])
            following = min(length, end + 1)
            backward[start] = np.logaddexp(backward[start], potentials[index] + backward[following])

    marginals = np.zeros(length)
    interval_marginals = np.zeros(len(bounds))
    for index, (start, end) in enumerate(bounds):
        left = forward[max(0, int(start) - 1)]
        right = backward[min(length, int(end) + 1)]
        probability = np.exp(min(0., left + potentials[index] + right - forward[length]))
        interval_marginals[index] = probability
        marginals[start:end] += probability

    selected, end = [], length
    while end:
        index = choice[end]
        if index < 0:
            end -= 1
        else:
            selected.append(index)
            end = max(0, int(bounds[index, 0]) - 1)
    return np.clip(marginals, 0, 1), np.asarray(selected[::-1], int), interval_marginals


def decode_spans(graph, spans, scores, reference, boundary_penalty=2.):
    group = graph.sample.task + '|' + graph.sample.generator
    table = reference['groups'][group]
    lengths_available = np.asarray(sorted(map(int, table)))
    bounds = np.asarray([(span.start, span.end) for span in spans], int).reshape(-1, 2)
    potentials = np.empty(len(spans))
    for index, span in enumerate(spans):
        length = span.end - span.start
        nearest = lengths_available[np.argmin(abs(lengths_available - length))]
        calibration = table[str(nearest)]
        standardized = (scores[index] - calibration['center']) / calibration['scale']
        potentials[index] = length * standardized - boundary_penalty
    marginal, selected, _ = interval_inference(graph.sample.response_length, bounds, potentials)
    marginal[~graph.coverage] = np.nan
    result = DetectionResult(graph.sample.response_id, marginal, bounds[selected], scores[selected], graph.coverage)
    return result, potentials
