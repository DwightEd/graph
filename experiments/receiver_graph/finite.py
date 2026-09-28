"""Validate selected actual history messages and their same-head aggregation."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.native import prefill
from experiments.decision_risk_flow.run import load_model
from experiments.route_complement.capture import get_inputs
from experiments.context_response.finite import native_margin


@torch.no_grad()
def check_edge(model, cache, tokens, prompt, answer, edge, saved, attention, derivative):
    receiver = edge['receiver']
    layer, head = edge['layer'], edge['head']
    position = torch.tensor([prompt-1+receiver], device=model.device)
    alternative = int(saved['alternative'][receiver])
    baseline = native_margin(model, cache, tokens, position, answer[receiver], alternative)
    assert abs(baseline-float(saved['confidence'][receiver,2]))<.005
    rows = []
    for label, senders in (('selected', [edge['sender']]), ('matched', [edge['control_sender']]),
                           ('joint', [edge['sender'], edge['control_sender']])):
        delta = torch.zeros(1, len(tokens), device=model.device)
        slope = 0.
        for sender in senders:
            key = prompt+sender
            assert key<int(position[0])
            delta[0,key] = -float(attention[layer,head,receiver,key])
            slope -= float(derivative[layer,head,receiver,key])
        for dose in (0., .01, .25, 1.):
            gate = dict(layer=layer, head=head, attention_delta=delta, dose=dose)
            observed = native_margin(model, cache, tokens, position, answer[receiver], alternative, gate)-baseline
            rows.append(dict(**edge, treatment=label, dose=dose, predicted=dose*slope, observed=observed))
    return rows


def summarize(rows):
    doses = {}
    for dose in (0., .01, .25, 1.):
        selected = [r for r in rows if r['dose']==dose]
        predicted = np.array([r['predicted'] for r in selected])
        observed = np.array([r['observed'] for r in selected])
        doses[str(dose)] = dict(count=len(selected), max_abs_error=float(np.max(abs(predicted-observed))),
            mean_abs_error=float(np.mean(abs(predicted-observed))),
            sign_agreement=float(np.mean(np.sign(predicted)==np.sign(observed))))
    interactions = []
    keys = {(r['key'],r['receiver'],r['layer'],r['head']) for r in rows}
    for key,receiver,layer,head in sorted(keys):
        group = [r for r in rows if (r['key'],r['receiver'],r['layer'],r['head'])==(key,receiver,layer,head)]
        for dose in (.01,.25,1.):
            value = {r['treatment']:r['observed'] for r in group if r['dose']==dose}
            interactions.append(dict(key=key, receiver=receiver, layer=layer, head=head, dose=dose,
                nonadditivity=value['joint']-value['selected']-value['matched']))
    return dict(doses=doses, interactions=interactions,
                scope='selected and matched keys in one head, native downstream at fixed past; not semantic truth or future generation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    edges = read_json(args.output/'interventions.json')['edges']
    root = Path(next(r['root'] for r in manifest['records'] if r['kind']=='observer'))
    lookup = {str(r['source_id']):r['source_file'] for r in read_json(root/'manifest.json')['records']}
    model = load_model(manifest['model'])
    rows = []
    start = perf_counter()
    for row in manifest['records']:
        selected = [e for e in edges if e['key']==row['key']]
        if not selected:
            continue
        prompt, answer, _ = get_inputs(row, Path(manifest['samples']), root, lookup)
        cache, _, _ = prefill(model, prompt, answer, checkpoints=(32,))
        directory = Path(manifest['base'])/row['key']
        attention = np.load(directory/'attention.npy', mmap_mode='r')
        derivative = np.load(directory/'derivative.npy', mmap_mode='r')
        with np.load(directory/'readouts.npz') as saved:
            for edge in selected:
                rows.extend(check_edge(model, cache, prompt+answer[:-1], len(prompt), answer,
                                       edge, saved, attention, derivative))
        write_json(args.output/'finite_progress.json', dict(last=row['key'], rows=rows))
        print('finite', row['key'], len(rows), round(perf_counter()-start,2), flush=True)
    write_json(args.output/'finite_effects.json', dict(status='complete', rows=rows, seconds=perf_counter()-start))
    write_json(args.output/'finite_summary.json', summarize(rows))


if __name__=='__main__':
    main()
