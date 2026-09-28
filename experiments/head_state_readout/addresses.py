"""Post-freeze address inspection: all source keys, no evidence/entity spans."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output/'evaluation.json')
    manifest = read_json(args.output/'manifest.json')
    lookup = {row['trace']: row for row in manifest['records'] if row['role']=='natural'}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    summaries = []
    for (case, side, trace), chosen in local.groupby(['case', 'side', 'trace']):
        record = lookup[trace]
        with np.load(Path(manifest['samples'])/trace) as saved:
            prompt = int(saved['prompt_length'])
            pieces = saved['token_text'][:prompt]
        with np.load(Path(manifest['route_base'])/record['key']/'measurements.npz') as saved:
            source = saved['source_mask'].astype(bool)
        original = Path(manifest['base'])/record['key']
        attention = np.load(original/'attention.npy', mmap_mode='r')
        derivative = np.load(original/'derivative.npy', mmap_mode='r')
        positions = chosen.position.to_numpy()
        reading = np.asarray(attention[:, :, positions, :prompt]).mean(2)*source
        acting = np.asarray(derivative[:, :, positions, :prompt]).mean(2)*source
        magnitude = np.abs(np.asarray(derivative[:, :, positions, :prompt])).mean(2)*source
        np.savez_compressed(args.output/(case+'_'+side+'_addresses.npz'),
                            reading=reading, signed=acting, magnitude=magnitude, source_mask=source)
        display = {}
        for name, values in [('reading', reading), ('absolute_native_effect', magnitude)]:
            total = values.sum((0, 1))
            total /= max(total.sum(), 1e-30)
            top = np.argsort(total)[-16:][::-1]
            display[name] = [dict(key=int(index), text=str(pieces[index]), mass=float(total[index]),
                context=''.join(pieces[max(0,index-3):index+4])) for index in top]
        summaries.append(dict(case=case, side=side, heads='all 32x32, retained in NPZ',
            mean_source_attention=float(reading.sum(-1).mean()), top_addresses=display))
    write_json(args.output/'address_audit.json', dict(labels='only select the already-reviewed output region after scoring',
        prompt_annotations=False, role='descriptive address inspection; no detector change', rows=summaries))


if __name__ == '__main__':
    main()
