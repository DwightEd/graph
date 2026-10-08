"""Frozen label-free readouts of native, candidate-specific head responses.

These are intervention hypotheses, not probabilities of factual correctness.
Entropy never enters a risk score. All physical heads are kept until the stated
readout; summing layers means simultaneous infinitesimal frozen-direction
interventions and is not a conserved attribution of source information.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE, LOCAL, OTHER_PROMPT = 0, 1, 4
ACTUAL_LOGP, NATIVE_MARGIN = 0, 1
DENOMINATOR_EPSILON = 1e-12


def score_responses(response, projected_norm, actual_logp, margin,
                    random_response=None):
    """Read [T,objective,layer,group,head] with no labels or learned weights."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 5 or response.shape[1:2] != (2,) or response.shape[3] != 5:
        raise ValueError('response must have [token,2,layer,5,head] axes')
    if not np.isfinite(response).all():
        raise ValueError('native response contains unmeasured/nonfinite values')
    source_logp = response[:, ACTUAL_LOGP, :, SOURCE]
    source_margin = response[:, NATIVE_MARGIN, :, SOURCE]
    local_margin = response[:, NATIVE_MARGIN, :, LOCAL]
    opposition = np.maximum(-source_margin, 0) * np.maximum(local_margin, 0)
    reverse = np.maximum(source_margin, 0) * np.maximum(-local_margin, 0)
    numerator = (opposition - reverse).sum(axis=(1, 2))
    denominator = np.sqrt(np.square(source_margin).sum(axis=(1, 2)) *
                          np.square(local_margin).sum(axis=(1, 2)))
    active = denominator > DENOMINATOR_EPSILON
    conflict = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=active)
    scores = {
        'source_candidate_regret': -source_logp.sum(axis=(1, 2)),
        'source_margin_rejection': -source_margin.sum(axis=(1, 2)),
        'source_local_countervote': conflict,
        'native_actual_nll': -np.asarray(actual_logp, dtype=np.float64),
        'native_actual_rejection': -np.asarray(margin, dtype=np.float64),
        'other_prompt_candidate_regret': -response[:, ACTUAL_LOGP, :, OTHER_PROMPT].sum(axis=(1, 2)),
        'absolute_source_response': np.abs(source_logp).sum(axis=(1, 2)),
        'source_message_norm': np.asarray(projected_norm)[:, :, SOURCE].sum(axis=(1, 2)),
    }
    if random_response is not None:
        random_response = np.asarray(random_response, dtype=np.float64)
        if random_response.shape != response.shape:
            raise ValueError('random control must preserve objective and head axes')
        scores['random_source_candidate_regret'] = -random_response[:, ACTUAL_LOGP, :, SOURCE].sum(axis=(1, 2))
        scores['random_source_margin_rejection'] = -random_response[:, NATIVE_MARGIN, :, SOURCE].sum(axis=(1, 2))
    diagnostics = dict(countervote_numerator=numerator, countervote_denominator=denominator,
                       countervote_active=active)
    if any(value.shape != (len(response),) or not np.isfinite(value).all()
           for value in scores.values()):
        raise ValueError('every score must cover every captured token exactly once')
    return scores, diagnostics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_answer(capture, record):
    """Only measurement fields enter scoring, never evaluation labels/text."""
    path = capture / record.get('arrays', f"{record['id']}/arrays.npz")
    with np.load(path, allow_pickle=False) as data:
        random = data['response_random_norm'] if 'response_random_norm' in data else None
        scores, diagnostics = score_responses(data['response'], data['projected_norm'],
            data['actual_logp'], data['margin'], random)
        identity = {key: data[key].copy() for key in ('actual_id', 'rival_id', 'query_position')}
    return scores, diagnostics, identity, path


def read_relay_answer(capture, record):
    path = capture / record['arrays']
    with np.load(path, allow_pickle=False) as data:
        names = ('regret_current', 'regret_past', 'regret_both', 'interaction')
        scores = {name: data[name].astype(np.float64) for name in names}
        scores['native_actual_nll'] = -data['actual_logp'].astype(np.float64)
        scores['native_actual_rejection'] = -data['margin'].astype(np.float64)
        identity = {key: data[key].copy() for key in ('actual_id', 'rival_id', 'query_position')}
    count = len(identity['actual_id'])
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError(f"{record['id']}: finite relay scores do not cover every token")
    return scores, {}, identity, path


def run(capture, output, relay=False):
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = capture / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'DONE' or not manifest.get('full_answer_capture'):
        raise ValueError('detection scoring requires completed whole-answer capture')
    arrays, records, offset, inputs = {}, [], 0, {str(manifest_path): file_hash(manifest_path)}
    for record in manifest['records']:
        reader = read_relay_answer if relay else read_answer
        scores, diagnostics, identity, input_path = reader(capture, record)
        count = len(identity['actual_id'])
        if count != record['token_count'] or count != record['full_token_count']:
            raise ValueError(f"incomplete capture for {record['id']}: {count}/{record['token_count']}")
        arrays.update({f"{record['id']}/{key}": value for key, value in {**scores, **diagnostics, **identity}.items()})
        records.append(dict(id=record['id'], source_id=record['source_id'], start=offset,
                            stop=offset + count, token_count=count, score_names=list(scores)))
        offset += count
        inputs[str(input_path)] = file_hash(input_path)
    np.savez_compressed(output / 'scores.npz', **arrays)
    formulas = ({'primary': 'logp_D(actual)-logp_C(actual)',
                 'current': 'logp_D(actual)-logp_A(actual)',
                 'past': 'logp_D(actual)-logp_B(actual)',
                 'interaction': 'regret_both-regret_current-regret_past'} if relay else
                {'primary': '-sum source d(logp(actual))/d(amplitude)',
                 'margin': '-sum source d(actual_logit-native_rival_logit)/d(amplitude)',
                 'countervote': 'sum positive(-u_source)*positive(u_local)-positive(u_source)*positive(-u_local), divided by product of L2 norms'})
    result = dict(primary='regret_both' if relay else 'source_candidate_regret', records=records, token_count=offset,
        input_hashes=inputs, score_sha256=file_hash(output / 'scores.npz'),
        code_hash=file_hash(__file__), uses_entropy=False, natural_label_fit=False,
        scope='L15 finite current/past/both amplification; teacher-forced native descendants' if relay else
              'fixed native past KV, current query, simultaneous frozen-direction local response',
        formulas=formulas)
    (output / 'frozen_scores.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relay', action='store_true', help='read finite current/past/both source effects')
    args = parser.parse_args()
    result = run(args.capture, args.output, relay=args.relay)
    print(json.dumps(dict(primary=result['primary'], tokens=result['token_count'],
                         score_sha256=result['score_sha256'])), flush=True)


if __name__ == '__main__':
    main()
