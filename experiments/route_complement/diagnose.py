"""Label-assisted local comparisons, retaining every head and source address."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans
from experiments.message_js.measure import normalize, js_rows
from .score import FIELDS


def head_auc(correct, wrong):
    """[L,H,token,field] -> [L,H,field], no independence or significance claim."""
    difference = wrong[:, :, :, None, :] - correct[:, :, None, :, :]
    available = np.isfinite(difference)
    numerator = ((difference > 0) + .5 * (difference == 0)).sum((2, 3))
    denominator = available.sum((2, 3))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def read_side(output, base, operator, record, positions):
    directory = output / record['key']
    with np.load(directory / 'head_features.npz') as saved:
        features = saved['values'][:, :, positions]
    with np.load(directory / 'measurements.npz') as saved:
        source = saved['source_mask']
        prompt = int(saved['prompt_length'])
    attention = np.load(base / record['key'] / 'attention.npy', mmap_mode='r')
    derivative = np.load(base / record['key'] / 'derivative.npy', mmap_mode='r')
    residual = np.load(operator / record['key'] / 'residual.npy', mmap_mode='r')
    ffn = np.load(operator / record['key'] / 'ffn.npy', mmap_mode='r')
    read_addresses, use_addresses, signed, cancellation = [], [], [], []
    for layer in range(32):
        reading = np.asarray(attention[layer])[:, positions, :prompt].copy()
        acting = np.asarray(derivative[layer])[:, positions, :prompt].copy()
        direct = np.asarray(residual[layer])[:, positions, :prompt].copy()
        transformed = np.asarray(ffn[layer])[:, positions, :prompt].copy()
        reading *= source
        acting *= source
        direct *= source
        transformed *= source
        read_addresses.append(normalize(reading).mean(1))
        use_addresses.append(normalize(np.abs(acting)).mean(1))
        signed.append(acting.sum(-1))
        denominator = np.abs(direct).sum(-1) + np.abs(transformed).sum(-1)
        cancellation.append(1 - np.divide(np.abs(acting).sum(-1), denominator,
                            out=np.ones_like(denominator), where=denominator > 0))
    gram = np.load(directory / 'head_gram.npy', mmap_mode='r')[:, 0, positions]
    norms = np.sqrt(np.maximum(np.diagonal(gram, axis1=-2, axis2=-1), 0))
    cosine = gram / np.maximum(norms[..., :, None] * norms[..., None, :], 1e-30)
    return dict(features=features, read_address=np.stack(read_addresses), use_address=np.stack(use_addresses),
                source_signed=np.stack(signed), source_ffn_cancel=np.stack(cancellation),
                head_cosine=cosine.mean(1))


def write_head_table(destination, case, correct, wrong, auroc):
    rows = []
    for layer in range(32):
        for head in range(32):
            for field, name in enumerate(FIELDS):
                rows.append(dict(case=case, layer=layer, head=head, field=name,
                    correct=float(correct['features'][layer, head, :, field].mean()),
                    wrong=float(wrong['features'][layer, head, :, field].mean()),
                    auroc=float(auroc[layer, head, field])))
    with destination.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'evaluation.json')
    manifest = read_json(args.output / 'manifest.json')
    base, operator = Path(manifest['base']), Path(manifest['operator'])
    lookup = {row['trace']: row for row in manifest['records'] if row['role'] == 'natural'}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    destination = args.output / 'diagnostics'
    destination.mkdir(exist_ok=False)
    summaries, maps, browser = [], [], []
    for case, group in local.groupby('case', sort=True):
        sides = {}
        for side, selected in group.groupby('side'):
            record = lookup[selected.trace.iloc[0]]
            positions = selected.position.to_numpy()
            sides[side] = read_side(args.output, base, operator, record, positions)
            with np.load(Path(manifest['samples']) / record['trace']) as trace:
                prompt = int(trace['prompt_length'])
                pieces = trace['token_text'][prompt:][positions].tolist()
                prompt_text = trace['token_text'][:prompt].tolist()
            browser.append(dict(case=case, side=side, key=record['key'], positions=positions.tolist(), text=pieces,
                values=sides[side]['features'].transpose(0, 1, 3, 2).tolist(),
                read=sides[side]['read_address'].tolist(), use=sides[side]['use_address'].tolist(), prompt=prompt_text))
        correct, wrong = sides['supported'], sides['unsupported']
        auroc = head_auc(correct['features'], wrong['features'])
        delta = wrong['features'].mean(2) - correct['features'].mean(2)
        write_head_table(destination / (case + '_all_heads.csv'), case, correct, wrong, auroc)
        saved = dict(head_auc=auroc, feature_delta=delta,
            read_address_delta=wrong['read_address']-correct['read_address'],
            use_address_delta=wrong['use_address']-correct['use_address'],
            head_cosine_delta=wrong['head_cosine']-correct['head_cosine'])
        for side, values in sides.items():
            for name, value in values.items():
                saved[side + '_' + name] = value
        np.savez_compressed(destination / (case + '.npz'), **saved)
        summaries.append(dict(case=case, correct_tokens=correct['features'].shape[2], wrong_tokens=wrong['features'].shape[2],
            fields={name: dict(correct_head_mean=float(correct['features'][..., index].mean()),
                wrong_head_mean=float(wrong['features'][..., index].mean()),
                heads_auc_above_075=int((auroc[..., index] > .75).sum()),
                heads_auc_below_025=int((auroc[..., index] < .25).sum())) for index, name in enumerate(FIELDS)}))
        maps.append(auroc)
    consensus = {name: dict(both_above_075=int(((maps[0][..., index] > .75) & (maps[1][..., index] > .75)).sum()),
        both_below_025=int(((maps[0][..., index] < .25) & (maps[1][..., index] < .25)).sum())) for index, name in enumerate(FIELDS)}
    write_json(destination / 'summary.json', dict(cases=summaries, consistent_heads=consensus,
        labels_for_diagnostics_only=True, head_selection_for_detection=False,
        uncertainty='Two sources only; token-pair AUROCs are descriptive, not independent tests. Prefix, wording and positions differ.'))
    write_json(destination / 'head_browser_data.json', dict(fields=FIELDS, rows=browser))
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, (summary, values) in enumerate(zip(summaries, maps)):
        for column, field in enumerate((0, 2, 5)):
            im = axes[row, column].imshow(values[..., field], vmin=0, vmax=1, cmap='coolwarm', aspect='auto')
            axes[row, column].set(title=summary['case'].split('_', 1)[1]+' / '+FIELDS[field], xlabel='Physical head', ylabel='Layer')
    figure.colorbar(im, ax=axes.ravel().tolist(), label='Local error-vs-correct AUROC (descriptive)')
    figure.savefig(destination / 'all_head_local_auc.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
