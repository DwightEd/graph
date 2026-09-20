"""Align saved token text exactly to official responses; never approximate offsets."""

import numpy as np
import pandas as pd

from .positions import merge_spans
from .sentences import map_offsets, sentence_intervals


def read_tokens(path, methods, records):
    table = pd.read_csv(path, dtype={"id": str, "source_id": str}, keep_default_na=False)
    for name in methods:
        table[name] = pd.to_numeric(table[name].replace("", np.nan), errors="raise")
    frames = []
    for identity, frame in table.groupby("id", sort=False):
        frame = frame.sort_values("token").reset_index(drop=True)
        record = records[str(identity)]
        if not np.array_equal(frame.token, np.arange(len(frame))):
            raise ValueError("A complete, unique token stream is required: " + identity)
        if set(frame.source_id) != {str(record["source_id"])}:
            raise ValueError("Source identity mismatch: " + identity)
        if "".join(frame.text) != record["response"]:
            raise ValueError("Token text does not concatenate to original response: " + identity)
        end = frame.text.str.len().cumsum().to_numpy()
        offsets = np.column_stack((np.r_[0, end[:-1]], end))
        frame["char_start"], frame["char_end"] = offsets.T
        sentences = sentence_intervals(record["response"])
        frame["sentence"], frame["crosses_sentence"] = map_offsets(record["response"], offsets, sentences)
        frames.append(annotate_spans(frame, offsets, record["labels"]))
    return pd.concat(frames, ignore_index=True)


def annotate_spans(frame, offsets, labels):
    intervals = []
    for label in labels:
        selected = (offsets[:, 1] > label["start"]) & (offsets[:, 0] < label["end"])
        indices = np.flatnonzero(selected & (offsets[:, 1] > offsets[:, 0]))
        if len(indices):
            intervals.append((indices[0], indices[-1] + 1))
    for name in ("span_index", "span_start", "span_end", "span_length", "offset"):
        frame[name] = -1
    for number, (start, end) in enumerate(merge_spans(intervals)):
        frame.loc[start:end - 1, "span_index"] = number
        frame.loc[start:end - 1, "span_start"] = start
        frame.loc[start:end - 1, "span_end"] = end
        frame.loc[start:end - 1, "span_length"] = end - start
        frame.loc[start:end - 1, "offset"] = np.arange(end - start)
    if not np.array_equal(frame.span_index.ge(0), frame.gold.astype(bool)):
        raise ValueError("Official annotation union differs from saved gold: " + frame.id.iloc[0])
    return frame
