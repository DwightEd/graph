"""Audit request priors and source addresses for every exposed TP/FP/FN token."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from .request_anchor import readout


def verify_priors(output, manifest):
    checked, seen = [], set()
    for row in manifest['records']:
        values = np.load(output/row['key']/'request.npz')['values']
        assert values.shape[:2]==(32, 32) and values.shape[-1]==4 and np.isfinite(values).all()
        identity = row['source_id']
        if identity in seen:
            continue
        seen.add(identity)
        saved = np.load(output/'anchors'/(identity+'.npz'))
        prior, indices = saved['prior'], saved['source_positions']
        np.testing.assert_allclose(prior.sum(-1), 1., atol=1e-12, rtol=0)
        assert np.all((saved['mass']>=0) & (saved['mass']<=1.))
        source = Path(manifest['relation_base'])/'sources'/identity
        queries = np.load(source/'query.npy', mmap_mode='r')
        keys = np.load(source/'key.npy', mmap_mode='r')
        reading = np.load(Path(manifest['base'])/row['key']/'attention.npy', mmap_mode='r')
        errors = []
        for layer in range(32):
            query = np.asarray(queries[layer])[:, -1].astype(np.float64)
            key = np.repeat(np.asarray(keys[layer])[:, indices], 4, axis=0).astype(np.float64)
            conditional = softmax(np.einsum('hd,hkd->hk', query, key)/128**.5, axis=-1)
            original = np.asarray(reading[layer])[:, 0, indices].astype(np.float64)
            original = original/original.sum(-1, keepdims=True)
            errors.append(float(np.max(np.abs(conditional-original))))
        assert max(errors)<1e-4, (identity, max(errors))
        checked.append(dict(source=identity, max_first_query_probability_error=max(errors),
            anchor_queries=saved['queries'].tolist(), fields_finite=True))
    write_json(output/'request_verification.json', dict(status='passed', sources=checked,
        answers=len(manifest['records']), preserved_physical_heads=1024,
        max_probability_error=max(r['max_first_query_probability_error'] for r in checked),
        scope='prompt Q/K first-query reconstruction and probability normalization; not finite intervention validation'))


def head_scores(output, rows):
    raw = {r['key']: readout(np.load(output/r['key']/'request.npz')['values']) for r in rows}
    reference = []
    for row in rows:
        if row['role']=='fit':
            indices = np.flatnonzero(valid_tokens(row, output, OLD))
            chosen = indices[np.linspace(0, len(indices)-1, 64).astype(int)]
            reference.append(raw[row['key']][chosen])
    reference = np.concatenate(reference)
    center = np.median(reference, axis=0)
    scale = np.maximum(np.quantile(reference, .75, axis=0)-np.quantile(reference, .25, axis=0), 1e-3)
    result = {}
    for key, values in raw.items():
        deviation = (values-center)/scale
        result[key] = np.sqrt(np.log1p(np.maximum(deviation[..., 0], 0))*np.log1p(np.abs(deviation[..., 1])))
        expected = np.load(output/key/'scores.npz')['request_joint']
        selected = np.partition(result[key], -8, axis=1)[:, -8:].mean(-1)
        np.testing.assert_allclose(selected, expected, atol=1e-12, rtol=0)
    return result


def addresses(tokenizer, prompt, indices, values, signed=False):
    ranking = np.abs(values) if signed else values
    chosen = np.argsort(ranking)[-5:][::-1]
    return [dict(position=int(indices[index]), value=float(values[index]),
        text=tokenizer.decode(prompt[max(0, int(indices[index])-4):int(indices[index])+5])) for index in chosen]


def audit_tokens(output, manifest):
    with (output/'token_audit.csv').open() as stream:
        tokens = [r for r in csv.DictReader(stream) if r['method']=='request_joint_fused' and r['status']!='TN']
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    scores = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        scores.update(head_scores(output, [r for r in manifest['records'] if r['task']==task]))
    result = []
    for row in manifest['records']:
        if row['role']!='regression':
            continue
        key = row['key']
        anchor = np.load(output/'anchors'/(row['source_id']+'.npz'))
        indices, prior = anchor['source_positions'], anchor['prior']
        prompt = np.load(Path(manifest['relation_base'])/'sources'/row['source_id']/'input.npz')['token_ids'].tolist()
        attention = np.load(Path(manifest['base'])/key/'attention.npy', mmap_mode='r')
        gradient = np.load(Path(manifest['base'])/key/'derivative.npy', mmap_mode='r')
        features = np.load(output/key/'request.npz')['values']
        for token in [r for r in tokens if r['key']==key]:
            position = int(token['position'])
            heads = []
            for flat in np.argsort(scores[key][position])[-8:][::-1]:
                layer, head = divmod(int(flat), 32)
                current = np.asarray(attention[layer, head, position])[indices]
                effect = np.asarray(gradient[layer, head, position])[indices]
                heads.append(dict(layer=layer, head=head, anomaly=float(scores[key][position, flat]),
                    js=float(features[layer, head, position, 0]), response=float(features[layer, head, position, 1]),
                    source_mass=float(features[layer, head, position, 2]),
                    current_read=addresses(tokenizer, prompt, indices, current),
                    request_prior=addresses(tokenizer, prompt, indices, prior[layer, head]),
                    current_gate_effect=addresses(tokenizer, prompt, indices, effect, signed=True)))
            result.append(dict(key=key, token=position, text=token['text'], status=token['status'], heads=heads))
    write_json(output/'request_token_addresses.json', dict(rows=result,
        scope='every exposed primary TP/FP/FN; top8 scoring heads, top5 addresses per view; full raw arrays retained',
        interpretation='reading and actual-vs-alternative choice effects; signs do not label factual support'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    verify_priors(args.output, manifest)
    audit_tokens(args.output, manifest)
    print('verified 36 source priors and saved source addresses for every primary TP/FP/FN')


if __name__=='__main__':
    main()
