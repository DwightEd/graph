"""Post-evaluation error audit, not a scoring rule or a model-selection tool."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.predictions), Path(args.output)
    manifest = json.loads((root / 'manifest.json').read_text())
    evaluation = json.loads(Path(args.evaluation).read_text())
    if not manifest['complete'] or evaluation['prediction_manifest_sha256'] != hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest():
        raise ValueError('complete evaluated predictions required')
    if evaluation['annotation_sha256'] != hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest():
        raise ValueError('annotation mismatch')
    from transformers import AutoTokenizer
    from next_iteration.grounding_contrast import focus_prefix
    from next_iteration.grounding_contrast_evaluate import annotation_targets
    tokenizer = AutoTokenizer.from_pretrained(manifest['settings']['model'], local_files_only=True)
    annotations = {str(r['id']): r for r in map(json.loads, Path(args.annotations).read_text().splitlines())
                   if str(r['id']) in manifest['planned_ids']}
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    indistinguishable = {'pre_token_state_only': {}, 'through_current_token': {}}
    detail = []
    for rid in manifest['planned_ids']:
        path = root / f'response_{rid}.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['output_sha256'][path.name]:
            raise ValueError('changed predictions')
        r, a = json.loads(path.read_text()), annotations[rid]
        if hashlib.sha256(a['response'].encode()).hexdigest() != r['response_sha256']:
            raise ValueError('response mismatch')
        y, onset, through = annotation_targets(r['offsets'], a['labels'])
        prefixes = [tokenizer.decode(r['token_ids'][:i + 1], skip_special_tokens=False) for i in range(len(y))]
        previous_error_in_clause = False
        previous_clause = None
        any_error = False
        for i, prefix in enumerate(prefixes):
            for mode, stop in [('pre_token_state_only', i), ('through_current_token', i + 1)]:
                key = (r['prompt_sha256'], tuple(r['token_ids'][:stop]))
                indistinguishable[mode].setdefault(key, []).append((int(y[i]), int(onset[i])))
            history, current = focus_prefix(prefix)
            clause = len(history)
            if clause != previous_clause:
                previous_error_in_clause = False
                previous_clause = clause
            if y[i]:
                region = 'error_onset' if onset[i] else 'error_continuation'
            elif previous_error_in_clause:
                region = 'correct_after_error_same_clause'
            elif any_error:
                region = 'correct_after_error_other_clause'
            else:
                region = 'correct_before_any_error'
            row = dict(id=rid, source_id=r['source_id'], task=r['task'], token_index=i,
                       start=r['offsets'][i][0], end=r['offsets'][i][1],
                       text=a['response'][r['offsets'][i][0]:r['offsets'][i][1]],
                       label=int(y[i]), onset=int(onset[i]), region=region,
                       source_risk=r['scores']['source_risk'][i], history_risk=r['scores']['history_risk'][i],
                       grounding_contrast=r['scores']['grounding_contrast'][i],
                       native_nll=r['scores']['native_nll'][i], native_entropy=r['scores']['native_entropy'][i],
                       retrospective_lookahead=r['retrospective_lookahead'][i])
            detail.append(row)
            groups.setdefault(region, []).append(row)
            previous_error_in_clause |= bool(y[i])
            any_error |= bool(y[i])
    with (output / 'all_tokens.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader(); writer.writerows(detail)
    summary = dict(purpose='diagnose frozen prediction failures; not a new detector or causal claim',
                   threshold='source_risk > 0 = B more likely than A; ties are not positive',
                   regions={name: dict(tokens=len(rows), source_risk_mean=float(np.mean([r['source_risk'] for r in rows])),
                                       source_B_fraction=float(np.mean([r['source_risk'] > 0 for r in rows])),
                                       history_B_fraction=float(np.mean([r['history_risk'] > 0 for r in rows])))
                            for name, rows in groups.items()})
    summary['identical_visible_contexts'] = {}
    for mode, contexts in indistinguishable.items():
        endpoints = {}
        for column, endpoint in [(0, 'all_token'), (1, 'span_onset')]:
            ptotal = sum(sum(v[column] for v in values) for values in contexts.values())
            ntotal = len(detail) - ptotal
            mixed_contexts = mixed_observations = tied_pairs = 0
            for values in contexts.values():
                pcount = sum(v[column] for v in values)
                ncount = len(values) - pcount
                if pcount and ncount:
                    mixed_contexts += 1; mixed_observations += len(values)
                    tied_pairs += pcount * ncount
            endpoints[endpoint] = dict(mixed_contexts=mixed_contexts, observations_in_mixed_contexts=mixed_observations,
                                       unavoidable_opposite_label_tie_pairs=tied_pairs,
                                       loose_auroc_upper_bound=1 - .5 * tied_pairs / (ptotal * ntotal) if ptotal * ntotal else None)
        summary['identical_visible_contexts'][mode] = endpoints
    summary['context_bound_scope'] = (
        'Applies only to deterministic scores with the specified visible information in this fixed observer dataset. '
        'Pre-token bound applies to entropy/margin, NOT target-token NLL. Through-current-token bound applies to NLL/verifier. '
        'Loose finite-sample necessary bound, not a general impossibility theorem or a fitted detector.'
    )
    (output / 'region_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
