"""Freeze automatic candidates using only input text and existing native caches."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from experiments.anchored_flow.edges import layer_arrays
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_source_control.measure import MODEL, centered_effect
from .model import group_roots, joint_profile, normalize, source_spans


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
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
        labels_for_scoring=False, labels_seen_in_design=True, native_gradients='reuse full independent-token root Jacobians',
        message_scope='cached native current-query actual-vs-rival margin, fixed past KV; distinct objective from roots',
        output_spans='robust adjacent evidence JS plus hidden/message change; no risk averaging',
        online=False, attribution_is_not_entailment=True, new_full_test=False))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for row in records:
        prepare_record(row, tokenizer, args.output)
    write_json(args.output / 'prepared.json', dict(status='complete', labels_read=False))


if __name__ == '__main__':
    main()
