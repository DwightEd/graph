"""Freeze automatic candidates using only input text and existing native caches."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from experiments.anchored_flow.edges import layer_arrays
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_source_control.measure import MODEL, centered_effect
from .model import group_roots, joint_profile, normalize, source_spans, relation_edits, repeated_source_sets


KEYS = ('15604', '11907', '12015', '12045', '12219', '219', '9022', '7305')
TRACES = (Path('outputs/token_backtrace_20260930_pilot'),
          Path('outputs/token_backtrace_20260930_controls'))


def message_profiles(row, groups):
    signed, attention = [], []
    for layer in range(32):
        weights, derivative = layer_arrays(row, layer)
        centered = centered_effect(weights, derivative)
        signed.append(np.stack([centered[..., group['keys']].sum(-1) for group in groups], -1))
        attention.append(np.stack([weights[..., group['keys']].sum(-1) for group in groups], -1))
    return np.asarray(signed), np.asarray(attention)


def prepare_record(row, tokenizer, output):
    directory = output / row['key']
    directory.mkdir()
    groups = source_spans(row['prompt'], row['source']['source_mask'], tokenizer)
    write_json(directory / 'sources.json', groups)
    trace_path = next(root / row['key'] / 'trace.npz' for root in TRACES if (root / row['key']).exists())
    with np.load(trace_path) as trace:
        np.testing.assert_array_equal(trace['token_ids'], row['response']['answer_ids'])
        root_signed, root_energy = group_roots(trace, groups)
        prompt = len(row['prompt'])
        history = np.abs(trace['root_effect'][:, prompt:]).sum(-1)
        history_fraction = history / np.maximum(np.abs(trace['root_effect']).sum(-1), 1e-30)
    signed, attention = message_profiles(row, groups)
    message_energy = np.sqrt((signed.astype(float) ** 2).sum((0, 1)))
    profile = joint_profile(root_energy, message_energy)
    np.savez_compressed(directory / 'prior.npz', root_signed=root_signed, root_energy=root_energy,
        message_signed=signed, attention=attention, prior=profile, history_fraction=history_fraction,
        attention_profile=normalize(attention.sum((0, 1))), message_energy=message_energy,
        token_ids=row['response']['answer_ids'])
    print(dict(key=row['key'], source_spans=len(groups), tokens=len(profile)), flush=True)


def prepare_relations(input_path, output, reuse_effects=None):
    """Extend an existing capture; no new trace collection or hand-picked tokens."""
    manifest = read_json(input_path / 'manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    write_json(output / 'manifest.json', manifest)
    protocol = dict(mode='relation', input=str(input_path), primary='relation_max_gain',
        methods=['relation_max_gain', 'relation_gain', 'relation_message_gain', 'uncorrected_flip_gain', 'repeat_set_odds'],
        labels_for_scoring=False, previously_exposed_development=True, new_backward_calls=0,
        candidates='all automatic boolean values, auxiliary negations, bounded durations in existing source spans',
        equivalent_controls='case change, negation contraction, bound paraphrase; semantic equivalence assumed, not certified',
        selection='existing joint top source; abstain if uncovered; main uses max over alternatives in that source, mean is a separate ablation; never pool output tokens',
        primary_score='max over alternative edits in the selected source of flip-minus-original token log odds minus absolute equivalent-minus-original log odds',
        message_ablation='candidate corrected gains times positive excess RMS message change over control divided by flip RMS, then mean within selected source',
        missing_candidate='score zero as explicit abstention; coverage mask saved; not evidence of correctness',
        repeat_sets='same quoted numeric value across fields; lexical redundancy hypothesis, not semantic equivalence',
        thresholds='within-answer unlabeled mixture95, alarm-budget comparison only',
        positions='text edits may change prompt token length; native positions, length changes recorded',
        no_risk_pooling=True, original_answer_preserved=True, original_generator=False)
    if reuse_effects is not None:
        protocol['reuse_effects'] = str(reuse_effects)
    write_json(output / 'protocol.json', protocol)
    for row in manifest['records']:
        directory = output / row['key']
        directory.mkdir()
        groups = read_json(input_path / row['key'] / 'sources.json')
        candidates = []
        for group in groups:
            start, stop = group['keys'][0], group['keys'][-1]+1
            assert group['keys'] == list(range(start, stop))
            text = tokenizer.decode(row['prompt'][start:stop])
            for edit in relation_edits(text):
                item = dict(source=group['index'], keys=[start, stop], original=text, **edit)
                for name in ('flip', 'equivalent'):
                    changed = text[:edit['start']] + edit[name] + text[edit['stop']:]
                    item[name+'_text'] = changed
                    item[name+'_ids'] = tokenizer.encode(changed, add_special_tokens=False)
                    item[name+'_length_change'] = len(item[name+'_ids'])-(stop-start)
                candidates.append(item)
        repeated = repeated_source_sets(groups)
        write_json(directory / 'candidates.json', dict(edits=candidates, repeated_sets=repeated))
        print(dict(key=row['key'], relation_candidates=len(candidates), repeated_sets=len(repeated)), flush=True)
    write_json(output / 'prepared.json', dict(status='complete', labels_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('source', 'relation'), default='source')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reuse-effects', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'relation':
        if args.input is None:
            parser.error('--mode relation requires --input with the existing source capture')
        prepare_relations(args.input, args.output, args.reuse_effects)
        return
    records = read_json('outputs/token_evidence_20260929_v1/manifest.json')['records']
    records = [row for row in records if row['key'] in KEYS]
    write_json(args.output / 'manifest.json', dict(records=records, model=MODEL, labels_used=False,
        created_utc=datetime.now(timezone.utc).isoformat(), exposed_development=True))
    write_json(args.output / 'protocol.json', dict(primary='localized_odds',
        source_candidates='all source-mask tokens partitioned at punctuation; no evidence annotations',
        source_ranking='geometric consensus of root gradient norms and centered current-query message effects',
        full_effect='each source span excluded as keys in all layers, positions and clamped text retained',
        js='full vocabulary, natural logs, independently for every output token',
        verification='all source candidates; gradient screening does not discard candidates',
        primary_score='masked-minus-original actual-token log odds at prior top source span; no route veto',
        ablations=['global_odds', 'max_span_odds', 'verified_localized_odds', 'state_change', 'profile_disagreement'],
        thresholds='label-free within-answer mixture95; alarm budget only, not nominal normal FPR',
        labels_for_scoring=False, labels_seen_in_design=True, native_gradients='reuse independent-target full-history embedding-gradient norms and gradient-dot-embedding summaries',
        message_scope='cached native current-query actual-vs-rival margin, fixed past KV; distinct objective from roots',
        output_spans='robust adjacent evidence JS plus hidden/message change; no risk averaging',
        online=False, attribution_is_not_entailment=True, new_full_test=False))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for row in records:
        prepare_record(row, tokenizer, args.output)
    write_json(args.output / 'prepared.json', dict(status='complete', labels_read=False))


if __name__ == '__main__':
    main()
