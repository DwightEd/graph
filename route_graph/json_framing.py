"""Lossless boundary parsing of one intact reader-generated JSON object."""

import hashlib
import json
import re


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_framed_json(raw):
    """Parse one intact object; never fix its contents or complete a prefix.

    Some reader outputs append closing punctuation after a complete root object.
    Such framing is recorded explicitly. Prose, second values/objects, duplicate
    keys, non-finite numbers and incomplete objects remain invalid.
    """
    if not isinstance(raw, str):
        raise TypeError("raw reader output must be text")
    left, right = len(raw) - len(raw.lstrip()), len(raw.rstrip())
    fenced = False
    if raw[left:right].startswith("```json\n") and raw[left:right].endswith("```"):
        left, right, fenced = left + 8, right - 3, True
        left += len(raw[left:right]) - len(raw[left:right].lstrip())
    if left >= right or raw[left] != "{":
        raise ValueError("expected JSON object at start")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    prediction, end = decoder.raw_decode(raw, idx=left)
    if end > right:
        raise ValueError("JSON object exceeds framing boundary")
    if not isinstance(prediction, dict):
        raise TypeError("expected JSON object")
    json.dumps(prediction, allow_nan=False)
    trailing = raw[end:right]
    if trailing.strip() and re.fullmatch(r"[\s}\]%]+", trailing) is None:
        raise ValueError("non-framing content follows JSON object")
    return {
        "prediction": prediction,
        "framing": {
            "status": "recovered" if fenced or trailing.strip() else "strict",
            "removed_json_fence": fenced,
            "discarded_suffix": trailing,
            "consumed_characters": end - left,
            "raw_root_span": [left, end],
            "raw_prefix": raw[:left],
            "raw_suffix": raw[end:],
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "parser": "intact-first-root-terminal-punctuation@1",
        },
    }
