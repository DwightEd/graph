"""无标签source参照与近邻距离；没有优化器、正确样本筛选或测试集调参。"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import pairwise_distances

from .operators import VARIANTS


def split_sources(records, seed, fraction=.2):
    by_task = defaultdict(set)
    official = {}
    for row in records:
        source = row['source_id']
        previous = official.setdefault(source, row['split'])
        if previous != row['split']:
            raise ValueError(f'overlapping train/test source: {source}')
        if row['split'] == 'train':
            by_task[row['task']].add(source)
    roles = {}
    random = np.random.default_rng(seed)
    for task, sources in sorted(by_task.items()):
        sources = sorted(sources)
        if len(sources) < 2:
            raise ValueError(f'{task}: need two train sources for independent calibration')
        random.shuffle(sources)
        count = min(len(sources) - 1, max(1, round(len(sources) * fraction)))
        roles.update({source: 'calibration' for source in sources[:count]})
        roles.update({source: 'reference' for source in sources[count:]})
    return roles


def group_records(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['task'], row['generator'])].append(row)
    return groups


def sample_reference_rows(root, records, budget, per_source, seed):
    """每个来源最多per_source个token；长回答和多回答来源不能支配参照。"""
    sources = defaultdict(list)
    for row in records:
        sources[row['source_id']].append(row)
    random = np.random.default_rng(seed)
    ordered_sources = sorted(sources)
    random.shuffle(ordered_sources)
    selected = []
    for source in ordered_sources[:max(1, budget // per_source)]:
        candidates = []
        for row in sources[source]:
            with np.load(root / row['file'], allow_pickle=False) as archive:
                positions = np.flatnonzero(archive['coverage'])
            candidates.extend((row['file'], int(position), source) for position in positions)
        count = min(per_source, len(candidates), budget - len(selected))
        indices = random.choice(len(candidates), count, replace=False)
        selected.extend(candidates[index] for index in indices)
    if not selected:
        raise ValueError('No covered tokens in the reference partition')
    return selected


def gather_embeddings(root, selected):
    by_file = defaultdict(list)
    for row_number, (file, position, source) in enumerate(selected):
        by_file[file].append((row_number, position))
    arrays = {name: [None] * len(selected) for name in VARIANTS}
    for file, coordinates in by_file.items():
        with np.load(root / file, allow_pickle=False) as archive:
            for name in VARIANTS:
                values = archive[name]
                for row_number, position in coordinates:
                    arrays[name][row_number] = values[position]
    return {name: np.asarray(rows, np.float32) for name, rows in arrays.items()}


def fit_reference(values, neighbors):
    """中位数/IQR只用参考来源估计；保留整个混合样本，不剔除疑似错误。"""
    center = np.median(values, axis=0)
    quartiles = np.quantile(values, [.25, .75], axis=0)
    scale = np.maximum(quartiles[1] - quartiles[0], .02)
    bank = ((values - center) / scale).astype(np.float32)
    return dict(center=center, scale=scale, bank=bank,
                neighbors=min(neighbors, len(bank)))


def novelty_distance(values, reference, batch_size=256):
    normalized = (values - reference['center']) / reference['scale']
    count = int(reference['neighbors'])
    scores = np.empty(len(values), np.float64)
    for start in range(0, len(values), batch_size):
        batch = normalized[start:start + batch_size]
        distances = pairwise_distances(batch, reference['bank'], metric='euclidean')
        nearest = np.partition(distances, count - 1, axis=1)[:, :count]
        scores[start:start + len(batch)] = nearest.mean(axis=1) / np.sqrt(values.shape[1])
    return scores


def source_quantiles(values, source_ids, probabilities):
    """不同来源总权重相同，阈值不按长回答的token数加权。"""
    _, inverse, counts = np.unique(source_ids, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    positions = np.searchsorted(cumulative, probabilities, side='left')
    return np.asarray(values)[order[np.minimum(positions, len(order) - 1)]]


def calibrate(values, source_ids, quantile):
    q25, median, q75, threshold = source_quantiles(values, source_ids, [.25, .5, .75, quantile])
    scale = max(float(q75 - q25), .001)
    return dict(center=float(median), scale=scale, threshold=float((threshold - median) / scale))


def apply_calibration(values, calibration):
    return (values - calibration['center']) / calibration['scale']


def smooth_scores(values, window):
    """纯时间平滑对照：不看attention端点，不使用金标边界。"""
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.)
    total = np.r_[0., np.cumsum(filled)]
    count = np.r_[0, np.cumsum(valid)]
    positions = np.arange(len(values))
    left = np.maximum(0, positions - window)
    right = np.minimum(len(values), positions + window + 1)
    width = count[right] - count[left]
    scores = np.divide(total[right] - total[left], width,
                       out=np.full(len(values), np.nan), where=width > 0)
    scores[~valid] = np.nan
    return scores


def binary_spans(alarm):
    """连续报警成段，单词与长段都允许；没有1～32窗口枚举。"""
    change = np.diff(np.r_[False, alarm, False].astype(int))
    return np.column_stack((np.flatnonzero(change == 1), np.flatnonzero(change == -1)))
