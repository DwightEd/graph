"""Post-hoc localization/confounding audit. These are not registered P2 candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from next_iteration.grounding_contrast_evaluate import annotation_targets
    root = Path(args.predictions)
    manifest = json.loads((root / 'manifest.json').read_text())
    evaluation = json.loads((root / 'evaluation.json').read_text())
    if not manifest['complete'] or manifest.get('error'):
        raise ValueError('incomplete predictions')
    annotation_hash = hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest()
    if annotation_hash != evaluation['annotation_sha256']:
        raise ValueError('annotation hash changed')
    annotations = {str(r['id']): r for r in map(json.loads, Path(args.annotations).read_text().splitlines())
                   if str(r['id']) in manifest['planned_ids']}
    labels, scores, within = [], {}, {}
    mixed = 0
    for rid in manifest['planned_ids']:
        file = root / f'response_{rid}.json'
        if hashlib.sha256(file.read_bytes()).hexdigest() != manifest['output_sha256'][file.name]:
            raise ValueError('prediction hash mismatch')
        r = json.loads(file.read_text())
        a = annotations[rid]
        if hashlib.sha256(a['response'].encode()).hexdigest() != r['response_sha256']:
            raise ValueError('response mismatch')
        y, _, _ = annotation_targets(r['offsets'], a['labels'])
        h, n = np.array(r['scores']['native_entropy']), len(y)
        diagnostics = dict(token_index=np.arange(n), generated_mass=r['generated_evidence_mass'],
                           first_entropy=np.full(n, h[0]), causal_prefix_mean=np.cumsum(h) / np.arange(1, n + 1))
        has_both = bool(y.any() and not y.all())
        mixed += int(has_both)
        for name, values in {**r['scores'], **diagnostics}.items():
            scores.setdefault(name, []).extend(values)
            if has_both:
                within.setdefault(name, []).append((float(roc_auc_score(y, values)), int(y.sum() * (n - y.sum()))))
        labels.extend(y)
    total_pairs = int(sum(labels) * (len(labels) - sum(labels)))
    within_pairs = sum(w for _, w in within['risk_transport'])
    results = dict(scope='post-hoc development diagnostic, not detector selection or a new primary result',
                   responses=len(manifest['planned_ids']), responses_with_both_classes=mixed,
                   tokens=len(labels), positives=int(sum(labels)),
                   all_opposite_class_pairs=total_pairs, within_response_opposite_class_pairs=within_pairs,
                   cross_response_pair_fraction=1 - within_pairs / total_pairs,
                   annotation_sha256=annotation_hash, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   metrics={name: dict(pooled_auroc=float(roc_auc_score(labels, values)),
                                       pooled_ap=float(average_precision_score(labels, values)),
                                       within_response_pair_weighted_auroc=sum(a * w for a, w in within[name]) / within_pairs,
                                       within_response_macro_auroc=float(np.mean([a for a, _ in within[name]])))
                            for name, values in scores.items()})
    with Path(args.output).open('x') as f:
        json.dump(results, f, indent=2); f.write('\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
