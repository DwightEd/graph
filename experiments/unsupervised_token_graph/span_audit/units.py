"""实验单位：真实标注片段、正常对照与一个回答的原始观测。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start


@dataclass(frozen=True)
class SpanPair:
    error: Span
    control: Span
    position_gap: float
    repetition_gap: float
    entropy_gap: float


@dataclass
class Answer:
    response_id: str
    source_id: str
    task: str
    generator: str
    split: str
    text: str
    token_ids: np.ndarray
    prompt_length: int
    offsets: np.ndarray
    error_mask: np.ndarray
    spans: list[Span]
    entropy: np.ndarray
    cache_paths: list[Path]

    @property
    def response_ids(self):
        return self.token_ids[self.prompt_length:]


def marked_spans(offsets, annotations):
    """金标字符区间映射为token区间；合并重叠，但不合并相邻的独立标注。"""
    spans = []
    for annotation in annotations:
        intersects = offsets[:, 0] < annotation['end']
        intersects &= offsets[:, 1] > annotation['start']
        intersects &= offsets[:, 1] > offsets[:, 0]
        positions = np.flatnonzero(intersects)
        if not len(positions):
            raise ValueError('Gold span has no verified token coverage')
        spans.append(Span(int(positions[0]), int(positions[-1]) + 1))

    merged = []
    for span in sorted(spans, key=lambda item: item.start):
        if merged and span.start < merged[-1].end:
            previous = merged.pop()
            span = Span(previous.start, max(previous.end, span.end))
        merged.append(span)
    return merged


def span_mask(length, spans):
    mask = np.zeros(length, dtype=bool)
    for span in spans:
        mask[span.start:span.end] = True
    return mask
