"""Freeze token scores and evidence regimes before any correctness labels are read."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .model import adjacent_change, js, normalize, odds, regimes, relation_scores


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


def score_relations(row, input_path, output):
    """Do not force a parsed relation onto tokens selecting an unparsed source."""
    from .state_path import path_spans

    directory = output / row['key']
    candidates = read_json(directory / 'candidates.json')
    with np.load(input_path / row['key'] / 'prior.npz') as prior:
        selected = prior['prior'].argmax(-1)
    scores = {name: np.zeros(len(selected)) for name in
              ('relation_max_gain', 'relation_gain', 'relation_message_gain', 'uncorrected_flip_gain', 'repeat_set_odds')}
    covered = np.zeros(len(selected), dtype=bool)
    with np.load(directory / 'relation_effects.npz') as effects:
        np.testing.assert_array_equal(effects['token_ids'], row['response']['answer_ids'])
        gain, message_gain, flip_gain, control_gain, specificity = relation_scores(
            effects['original_logp'], effects['flip_logp'], effects['equivalent_logp'],
            effects['flip_message_change'], effects['equivalent_message_change'])
        for source in sorted({candidate['source'] for candidate in candidates['edits']}):
            columns = [index for index, candidate in enumerate(candidates['edits']) if candidate['source'] == source]
            mask = selected == source
            covered[mask] = True
            scores['relation_max_gain'][mask] = gain[mask][:, columns].max(-1)
            for name, values in (('relation_gain', gain), ('relation_message_gain', message_gain),
                                 ('uncorrected_flip_gain', flip_gain)):
                scores[name][mask] = values[mask][:, columns].mean(-1)
        repeat_covered = np.zeros(len(selected), dtype=bool)
        for index, group in enumerate(candidates['repeated_sets']):
            mask = np.isin(selected, group['sources'])
            repeat_covered[mask] = True
            scores['repeat_set_odds'][mask] = (odds(effects['repeat_logp'][:, index]) - odds(effects['original_logp']))[mask]
        np.savez_compressed(directory / 'scores.npz', **scores, selected=selected, confirmed=selected,
            covered=covered, repeat_covered=repeat_covered, candidate_gain=gain, candidate_flip_gain=flip_gain,
            candidate_control_gain=control_gain, candidate_message_specificity=specificity,
            token_ids=effects['token_ids'])
    write_json(directory / 'output_spans.json', path_spans(selected))
    assert all(np.isfinite(values).all() for values in scores.values())
    return {name: float(np.quantile(values, .95)) for name, values in scores.items()}


def score_all_relations(row, output):
    """Exploratory readout of already measured candidates; no source-prior veto."""
    from .state_path import path_spans

    directory = output / row['key']
    edits = read_json(directory/'candidates.json')['edits']
    methods = ('all_relation_logp', 'all_relation_odds', 'all_relation_message_logp')
    count = len(row['response']['answer_ids'])
    values = {name: np.zeros(count) for name in methods}
    with np.load(directory/'scores.npz') as previous:
        selected = previous['selected'].copy()
    with np.load(directory/'relation_effects.npz') as effects:
        if edits:
            gain, _, _, _, specificity = relation_scores(effects['original_logp'],
                effects['flip_logp'], effects['equivalent_logp'],
                effects['flip_message_change'], effects['equivalent_message_change'])
            logp_gain = effects['flip_logp']-effects['original_logp'][:, None]
            logp_gain -= np.abs(effects['equivalent_logp']-effects['original_logp'][:, None])
            values = dict(all_relation_logp=logp_gain.max(-1), all_relation_odds=gain.max(-1),
                          all_relation_message_logp=(logp_gain*specificity).max(-1))
            selected = np.array([edits[index]['source'] for index in logp_gain.argmax(-1)])
        np.savez_compressed(directory/'all_scores.npz', **values, selected=selected, confirmed=selected,
            covered=np.full(count, bool(edits)), repeat_covered=np.zeros(count, bool), token_ids=effects['token_ids'])
    write_json(directory/'all_output_spans.json', path_spans(selected))
    return {name: float(np.quantile(value, .95)) for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readout', choices=('selected', 'all'), default='selected')
    args = parser.parse_args()
    records = read_json(args.output / 'manifest.json')['records']
    protocol = read_json(args.output / 'protocol.json')
    if args.readout == 'all':
        assert protocol.get('mode') == 'relation'
        if (args.output/'all_scores_frozen.json').exists():
            raise FileExistsError('All-candidate readout is already frozen')
        read_json(args.output/'scores_frozen.json')
        thresholds = {row['key']: score_all_relations(row, args.output) for row in records}
        write_json(args.output/'all_thresholds.json', thresholds)
        write_json(args.output/'all_scores_frozen.json', dict(status='complete', labels_read=False,
            created_utc=datetime.now(timezone.utc).isoformat(), primary='all_relation_logp',
            methods=list(next(iter(thresholds.values()))), previous_development_results_seen=True,
            scope='CPU exploratory readout; all measured relation candidates; raw logp primary; zero if no candidates'))
        return
    if protocol.get('mode') == 'relation':
        thresholds = {}
        for row in records:
            read_json(args.output / row['key'] / 'capture_complete.json')
            thresholds[row['key']] = score_relations(row, Path(protocol['input']), args.output)
        write_json(args.output / 'thresholds.json', thresholds)
        write_json(args.output / 'scores_frozen.json', dict(status='complete', labels_read=False,
            created_utc=datetime.now(timezone.utc).isoformat(), methods=protocol['methods'],
            answers=len(records), tokens=sum(len(row['response']['answer_ids']) for row in records)))
        return
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
