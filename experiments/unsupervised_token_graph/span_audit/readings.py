"""单head的真实读取与条件随机端点基线；不做同层伪多跳。"""

import numpy as np


class PredictionRows:
    """缓存query q预测q+1；缺行返回None，不补零。"""

    def __init__(self, channel, prompt_length):
        self.channel = channel
        self.row_by_token = {
            int(query + 1 - prompt_length): row
            for row, query in enumerate(channel.queries)
        }

    def read(self, response_position):
        row = self.row_by_token.get(response_position)
        if row is None:
            return None
        return self.channel.row(row)


def source_statistics(keys, weights, prompt_length):
    """只称prompt来源，不把整个prompt自动当作正确证据。"""
    prompt_weights = weights[keys < prompt_length]
    prompt_mass = float(prompt_weights.sum())
    if prompt_mass <= 0:
        return np.array([0., np.nan, np.nan, weights.sum()])

    distribution = prompt_weights / prompt_mass
    positive = distribution[distribution > 0]
    entropy = -np.sum(positive * np.log(positive))
    return np.array([prompt_mass, np.exp(entropy), distribution.max(), weights.sum()])


def lag_groups(lags):
    """lag1独立；其余为2、3-4、5-8、9-16等距离组。"""
    return np.ceil(np.log2(lags)).astype(int)


def endpoint_expectation(answer, position, keys, weights, span):
    """保持距离组、是否复现当前token及各组总权重，解析计算随机端点期望。"""
    history = keys >= answer.prompt_length
    source_tokens = keys[history] - answer.prompt_length
    history_weights = weights[history]
    previous = np.arange(position)
    target_id = answer.response_ids[position]

    repeated = answer.response_ids[previous] == target_id
    strata = 2 * lag_groups(position - previous) + repeated
    selected = (previous >= span.start) & (previous < span.end)
    totals = np.bincount(strata)
    selected_counts = np.bincount(strata[selected], minlength=len(totals))
    fractions = selected_counts / np.maximum(totals, 1)

    source_strata = strata[source_tokens]
    expected = np.sum(history_weights * fractions[source_strata])
    noncopy = answer.response_ids[source_tokens] != target_id
    expected_noncopy = np.sum(history_weights[noncopy] * fractions[source_strata[noncopy]])
    movable = (fractions[source_strata] > 0) & (fractions[source_strata] < 1)
    return float(expected), float(expected_noncopy), float(history_weights[movable].sum())


def reuse_statistics(answer, position, keys, weights, span):
    """统计对同一已知区间的读取；结束后仍用原区间，绝不在金标终点清零。"""
    if position == 0:
        return np.zeros(5)
    source_tokens = keys - answer.prompt_length
    selected = (source_tokens >= span.start) & (source_tokens < span.end)
    observed = float(weights[selected].sum())

    selected_ids = answer.token_ids[keys[selected]]
    noncopy = selected_ids != answer.response_ids[position]
    observed_noncopy = float(weights[selected][noncopy].sum())
    expected, expected_noncopy, movable_mass = endpoint_expectation(
        answer, position, keys, weights, span)
    return np.array([observed, expected, observed - expected,
                     observed_noncopy - expected_noncopy, movable_mass])
