"""Shared cache IO; official annotations are accessed only by evaluation."""

import json
from pathlib import Path
import sys

import numpy as np

# Direct ``python -m experiments...`` calls use the existing native adapters.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'teaching/state_audit/src'))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def inputs(record):
    root = Path(record['root'])
    prompt = read_json(root / record['source_file'])['prompt_with_source']
    response = read_json(root / record['directory'] / 'response.json')
    return prompt, response


def labels(records):
    """Load official labels after the caller has frozen detector scores."""
    from experiments.native_support.ragtruth_benchmark.data import annotations

    root = Path(records[0]['root'])
    truth = annotations(root, read_json(root / 'manifest.json'), records)
    return {row['key']: np.asarray(truth[row['id']]['labels'], dtype=np.int64)
            for row in records}
