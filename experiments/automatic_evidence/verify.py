"""Check numerical/coverage contracts against raw arrays, without modifying them."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .model import odds
from .prepare import TRACES
from .score import METHODS


def check_record(row, output):
    directory = output / row['key']
    groups = read_json(directory / 'sources.json')
    keys = [key for group in groups for key in group['keys']]
    expected = np.flatnonzero(row['source']['source_mask']).tolist()
    assert keys == expected, (row['key'], 'source coverage')
    trace_path = next(root/row['key']/'trace.npz' for root in TRACES if (root/row['key']).exists())
    with np.load(directory/'effects.npz') as effect, np.load(directory/'scores.npz') as scores, np.load(trace_path) as trace:
        np.testing.assert_array_equal(effect['token_ids'], row['response']['answer_ids'])
        for name in ('output_js', 'masked_logp', 'hidden_change', 'message_change', 'original_hidden'):
            assert np.isfinite(effect[name]).all(), (row['key'], name)
        assert effect['output_js'].min() >= 0
        assert effect['output_js'].max() <= np.log(2)+1e-5
        assert effect['message_change'].shape == (len(groups), 32, 32, len(effect['token_ids']))
        original_error = float(np.max(np.abs(effect['original_logp']-trace['logp'])))
        assert original_error < 2e-4, (row['key'], original_error)
        selected = scores['selected']
        actual = effect['masked_logp'][np.arange(len(selected)), selected]
        np.testing.assert_allclose(scores['localized_odds'], odds(actual)-odds(effect['original_logp']))
        for name in METHODS:
            assert np.isfinite(scores[name]).all(), (row['key'], name)
    spans = read_json(directory/'output_spans.json')
    assert [target for span in spans for target in range(span['start'], span['stop'])] == list(range(len(selected)))
    return dict(source_tokens=len(keys), source_spans=len(groups), targets=len(selected),
                original_logp_error=original_error, output_spans=len(spans))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_json(args.output/'manifest.json')['records']
    checks = {row['key']: check_record(row, args.output) for row in rows}
    expected = read_json(args.output/'witness_code_sha256.json')
    current = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
               for name in ('model.py', 'prepare.py', 'capture.py', 'score.py')}
    for name, digest in current.items():
        assert digest == expected['files'][name]['sha256'], (name, 'frozen code changed')
    write_json(args.output/'checks.json', dict(status='complete', records=checks,
        current_code_sha256=current, witness_record=expected, labels_read=False))
    print(checks, flush=True)


if __name__ == '__main__':
    main()
