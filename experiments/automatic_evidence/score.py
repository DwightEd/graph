"""Freeze token scores and evidence regimes before any correctness labels are read."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .model import adjacent_change, js, normalize, odds, regimes


METHODS = ('localized_odds', 'global_odds', 'max_span_odds',
           'verified_localized_odds', 'state_change', 'profile_disagreement')


def localization_metrics(prior, effects, groups):
    """Finite intervention fidelity, NOT gold semantic evidence accuracy."""
    divergence = effects['output_js']
    rows = np.arange(len(divergence))
    maximum = np.maximum(divergence.max(-1), 1e-12)
    lengths = np.array([len(group['keys']) for group in groups])
    profiles = dict(joint=prior['prior'], roots=prior['root_energy'],
                    messages=prior['message_energy'], attention=prior['attention_profile'])
    result = {}
    for name, profile in profiles.items():
        best = profile.argmax(-1)
        chosen = divergence[rows, best]
        random_expected = []
        for target, selected in enumerate(best):
            # Closest nonselected lengths are an exact local random-control expectation.
            distance = np.abs(lengths - lengths[selected]).astype(float)
            distance[selected] = np.inf
            controls = np.flatnonzero(distance == distance.min())
            random_expected.append(float(divergence[target, controls].mean()))
        random_expected = np.array(random_expected)
        result[name] = dict(top1_effect_ratio=float((chosen/maximum).mean()),
            matched_random_ratio=float((random_expected/maximum).mean()),
            selected_js_mean=float(chosen.mean()), matched_random_js_mean=float(random_expected.mean()),
            exceeds_matched_random=float((chosen > random_expected).mean()),
            effect_argmax_hit=float((best == divergence.argmax(-1)).mean()))
    return result


def score_record(row, output):
    directory = output / row['key']
    with np.load(directory / 'prior.npz') as prior, np.load(directory / 'effects.npz') as effects:
        np.testing.assert_array_equal(prior['token_ids'], effects['token_ids'])
        count = len(effects['token_ids'])
        index = np.arange(count)
        profile = prior['prior']
        verified = normalize(profile * (effects['output_js'] + 1e-12))
        selected = profile.argmax(-1)
        confirmed = verified.argmax(-1)
        contrast = odds(effects['masked_logp']) - odds(effects['original_logp'][:, None])
        hidden_change = adjacent_change(effects['original_hidden'])
        payload_path = Path('outputs/span_maintenance_20260929_v2') / row['key'] / 'payload_changes.npz'
        with np.load(payload_path) as payload:
            # Existing full physical layer/head content changes; first position has no predecessor.
            message_change = np.zeros(count)
            message_change[1:] = np.sqrt(np.nanmean(payload['head_value_change'][:, 0, :, 1:] ** 2, axis=(0, 1)))
        spans, change, cutoffs = regimes(verified, hidden_change, message_change)
        scores = dict(localized_odds=contrast[index, selected],
            global_odds=odds(effects['empty_logp'])-odds(effects['original_logp']),
            max_span_odds=contrast.max(-1), verified_localized_odds=contrast[index, confirmed],
            state_change=change[:, 0],
            profile_disagreement=js(normalize(prior['root_energy']), normalize(prior['message_energy'])))
        assert all(np.isfinite(value).all() for value in scores.values())
        thresholds = {name: float(np.quantile(value, .95)) for name, value in scores.items()}
        np.savez_compressed(directory / 'scores.npz', **scores, selected=selected, confirmed=confirmed,
            verified_profile=verified, state_changes=change, state_thresholds=cutoffs,
            token_ids=effects['token_ids'])
        write_json(directory / 'output_spans.json', spans)
        groups = read_json(directory / 'sources.json')
        fidelity = localization_metrics(prior, effects, groups)
    return thresholds, fidelity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_json(args.output / 'manifest.json')['records']
    thresholds, localization = {}, {}
    for row in records:
        read_json(args.output / row['key'] / 'capture_complete.json')
        thresholds[row['key']], localization[row['key']] = score_record(row, args.output)
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'localization.json', localization)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', labels_read=False,
        created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS,
        answers=len(records), tokens=sum(len(row['response']['answer_ids']) for row in records)))


if __name__ == '__main__':
    main()
