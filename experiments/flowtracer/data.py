"""Small, dependency-free sample readers for FlowTracer."""

import json
from pathlib import Path


def iter_samples(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        yield from (value if isinstance(value, list) else [value])


def get_token_regions(sample):
    if "target_tokens" in sample:
        target = sorted({int(i) for i in sample["target_tokens"]})
    elif "target" in sample:
        value = sample["target"]
        start, end = (value["start"], value["end"]) if isinstance(value, dict) else value
        target = list(range(int(start), int(end)))
    elif "target_span" in sample:
        start, end = sample["target_span"]
        target = list(range(int(start), int(end)))
    else:
        raise ValueError("sample requires target_tokens, target, or target_span")
    return target
