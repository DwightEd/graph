"""先求同source的配对差，再对source重采样；head不是独立样本。"""

import numpy as np


def source_effects(source_ids, effects):
    """effects: [pair, channel, metric]，数值为错误减正常。"""
    rows = []
    for source in sorted(set(source_ids)):
        selected = np.asarray(source_ids) == source
        values = effects[selected]
        count = np.isfinite(values).sum(axis=0)
        total = np.nansum(values, axis=0)
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        rows.append(mean)
    return np.stack(rows)


def source_bootstrap(values, draws=200, seed=17):
    """一个source只投一票；置信区间是探索性描述，不自动宣布机制成立。"""
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    lower = np.full_like(mean, np.nan)
    upper = np.full_like(mean, np.nan)
    if len(values) < 2 or draws == 0:
        return mean, lower, upper, count

    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        selected = random.integers(len(values), size=len(values))
        sampled = values[selected]
        denominator = np.isfinite(sampled).sum(axis=0)
        numerator = np.nansum(sampled, axis=0)
        estimates.append(np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan), where=denominator > 0))

    estimates = np.asarray(estimates)
    for coordinate in np.ndindex(mean.shape):
        distribution = estimates[(slice(None),) + coordinate]
        distribution = distribution[np.isfinite(distribution)]
        if count[coordinate] >= 2 and len(distribution):
            lower[coordinate], upper[coordinate] = np.quantile(distribution, [.025, .975])
    return mean, lower, upper, count
