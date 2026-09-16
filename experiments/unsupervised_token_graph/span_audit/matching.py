"""用标签选正常对照，不训练分类器，也不把窗口当成模型发现的片段。"""

import numpy as np

from .units import Span, SpanPair


def token_kind(answer, position):
    """仅匹配首token的表面类别，不冒充词性或事实角色。"""
    start, end = answer.offsets[position]
    text = answer.text[start:end].strip()
    if not text:
        return 'space'
    if text[0].isdigit():
        return 'number'
    if text[0].isalpha():
        return 'word'
    return 'punctuation'


def repetition_rate(answer, span):
    tokens = answer.response_ids[span.start:span.end]
    return 1 - len(np.unique(tokens)) / len(tokens)


def has_text(answer, span):
    offsets = answer.offsets[span.start:span.end]
    return bool(np.all(offsets[:, 1] > offsets[:, 0]))


def overlaps(first, second):
    return first.start < second.end and second.start < first.end


def candidate_controls(answer, error, used, window):
    """只搜索与某个金标片段等长的正常对照，不枚举检测候选。"""
    length = len(answer.response_ids)
    for start in range(length - error.length + 1):
        candidate = Span(start, start + error.length)
        begin = max(0, candidate.start - window)
        end = min(length, candidate.end + window)
        if answer.error_mask[begin:end].any():
            continue
        previous_error = answer.error_mask[:error.start].any()
        if answer.error_mask[:candidate.start].any() != previous_error:
            continue
        if any(overlaps(candidate, previous) for previous in used):
            continue
        if not has_text(answer, candidate):
            continue
        if token_kind(answer, start) == token_kind(answer, error.start):
            yield candidate


def pair_distance(answer, error, control, position_limit, repetition_limit, entropy_limit):
    """同答固定了source/任务/生成器；其余差异按预定容差筛选并保留记录。"""
    position_gap = abs(error.start - control.start) / len(answer.response_ids)
    repetition_gap = abs(repetition_rate(answer, error) - repetition_rate(answer, control))
    entropy_gap = abs(answer.entropy[error.start] - answer.entropy[control.start])

    if position_gap > position_limit or repetition_gap > repetition_limit:
        return None
    if np.isfinite(entropy_gap) and entropy_gap > entropy_limit:
        return None
    return SpanPair(error, control, position_gap, repetition_gap, float(entropy_gap))


def match_controls(answer, position_limit=.25, repetition_limit=.15, entropy_limit=.5, window=8):
    """每个错误区间最多一个不重用的正常对照；不放宽规则强行凑满。"""
    pairs = []
    used = []
    for error in answer.spans:
        candidates = []
        for control in candidate_controls(answer, error, used, window):
            pair = pair_distance(answer, error, control, position_limit,
                                 repetition_limit, entropy_limit)
            if pair is not None:
                candidates.append(pair)
        if not candidates:
            continue
        best = min(candidates, key=lambda pair: (
            pair.position_gap, pair.repetition_gap, pair.control.start))
        pairs.append(best)
        used.append(best.control)
    return pairs


def matched_positions(answer, pair, phase, window):
    """两个区间使用相同相对位置；边界外遇到其他错误或回答结束就截断。"""
    error, control = pair.error, pair.control
    if phase == 'before':
        offsets = range(-1, -window - 1, -1)
        bases = (error.start, control.start)
    elif phase == 'inside':
        offsets = range(1, error.length)
        bases = (error.start, control.start)
    else:
        offsets = range(window)
        bases = (error.end, control.end)

    positions = []
    for offset in offsets:
        left, right = bases[0] + offset, bases[1] + offset
        if min(left, right) < 0 or max(left, right) >= len(answer.response_ids):
            break
        if phase != 'inside' and (answer.error_mask[left] or answer.error_mask[right]):
            break
        positions.append((left, right))
    return positions
