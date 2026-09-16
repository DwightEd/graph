"""三个问题分别测量：入口变化、段内依赖、退出变化。没有综合风险分。"""

import numpy as np

from .matching import matched_positions
from .readings import PredictionRows, reuse_statistics, source_statistics


METRICS = (
    'onset_prompt_mass', 'onset_effective_sources', 'onset_top_source_share', 'onset_retained_mass',
    'onset_prompt_change', 'onset_source_count_change',
    'inside_observed_mass', 'inside_lag_copy_null', 'inside_endpoint_excess',
    'inside_noncopy_excess', 'inside_movable_mass',
    'after_observed_mass', 'after_endpoint_excess', 'after_noncopy_excess',
    'boundary_excess_change',
)


def paired_readings(rows, positions):
    """只比较两侧都有缓存行的位置，保留覆盖数。"""
    for error_position, control_position in positions:
        error_row = rows.read(error_position)
        control_row = rows.read(control_position)
        if error_row is not None and control_row is not None:
            yield (error_position, control_position), (error_row, control_row)


def average_readings(readings, width):
    if not readings:
        return np.full((2, width), np.nan)
    return np.mean(readings, axis=0)


def measure_entry(answer, pair, rows, window):
    onset = [(pair.error.start, pair.control.start)]
    current = []
    for _, edge_rows in paired_readings(rows, onset):
        current.append([source_statistics(*row, answer.prompt_length) for row in edge_rows])

    previous = []
    positions = matched_positions(answer, pair, 'before', window)
    for _, edge_rows in paired_readings(rows, positions):
        previous.append([source_statistics(*row, answer.prompt_length) for row in edge_rows])
    values = average_readings(current, 4)
    baseline = average_readings(previous, 4)
    return np.column_stack((values, values[:, :2] - baseline[:, :2])), len(current), len(previous)


def measure_phase(answer, pair, rows, phase, window):
    positions = matched_positions(answer, pair, phase, window)
    readings = []
    for targets, edge_rows in paired_readings(rows, positions):
        pair_values = []
        for position, row, span in zip(targets, edge_rows, (pair.error, pair.control)):
            pair_values.append(reuse_statistics(answer, position, *row, span))
        readings.append(pair_values)
    return readings


def measure_pair(answer, pair, rows, window):
    entry, onset_count, before_count = measure_entry(answer, pair, rows, window)
    inside_rows = measure_phase(answer, pair, rows, 'inside', window)
    after_rows = measure_phase(answer, pair, rows, 'after', window)
    inside = average_readings(inside_rows, 5)
    after = average_readings(after_rows, 5)
    tail = average_readings(inside_rows[-window:], 5)
    boundary_change = after[:, 2] - tail[:, 2]

    values = np.column_stack((entry, inside, after[:, [0, 2, 3]], boundary_change))
    counts = [onset_count, before_count, len(inside_rows), len(after_rows)]
    return values, counts


def measure_answer(answer, pairs, channels, window):
    """流式处理每个物理head；只保存最终测量，不常驻1024张CSR。"""
    identities = []
    readings = []
    coverage = []
    for channel in channels:
        identities.append((channel.layer, channel.head))
        rows = PredictionRows(channel, answer.prompt_length)
        results = [measure_pair(answer, pair, rows, window) for pair in pairs]
        readings.append([result[0] for result in results])
        coverage.append([result[1] for result in results])

    shape = (len(identities), len(pairs), 2, len(METRICS))
    values = np.asarray(readings, dtype=np.float64).reshape(shape)
    counts = np.asarray(coverage, dtype=int).reshape(len(identities), len(pairs), 4)
    return np.asarray(identities, dtype=int).reshape(-1, 2), values, counts
