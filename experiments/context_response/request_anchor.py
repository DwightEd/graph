"""Request-conditioned native source reading, without annotated prompt entities."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import softmax, xlogy

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.source_relation.refine import fit_rank
from .score import calibrate
from .sparse import sparse_mean

FIELDS = ('request_js', 'request_response', 'source_mass', 'anchor_read_mass')


def matched_endpoint_mean(reading, queries, source_ids, prompt_ids):
    """Preserve source mass in each lag/copy group; average its endpoints."""
    expected = reading.copy()
    for index, query in enumerate(queries):
        lag = query-source_ids
        band = np.ceil(np.log2(np.maximum(lag, 1))).astype(int)
        band[lag==0] = -1
        copied = prompt_ids[source_ids]==prompt_ids[query]
        groups = 2*band+copied.astype(int)
        for group in np.unique(groups):
            selected = groups==group
            expected[:, index, selected] = reading[:, index, selected].mean(-1, keepdims=True)
    np.testing.assert_allclose(expected.sum(-1), reading.sum(-1), atol=1e-14, rtol=1e-12)
    return expected


def anchor_reading(source, matched=False):
    saved = np.load(source/'input.npz')
    source_ids = np.flatnonzero(saved['source_mask'])
    prompt_ids = saved['token_ids']
    # The non-source suffix and final pre-answer query require no entity tags.
    queries = np.flatnonzero((np.arange(len(prompt_ids))>source_ids[-1]) & ~saved['source_mask'])
    queries = np.unique(np.r_[queries, len(prompt_ids)-1])
    q = np.load(source/'query.npy', mmap_mode='r')
    k = np.load(source/'key.npy', mmap_mode='r')
    priors, masses = [], []
    for layer in range(len(q)):
        query = np.asarray(q[layer])[:, queries].astype(np.float64)
        keys = np.repeat(np.asarray(k[layer]), q.shape[1]//k.shape[1], axis=0).astype(np.float64)
        logits = query @ keys.swapaxes(-1, -2)/q.shape[-1]**.5
        visible = np.arange(len(prompt_ids))[None]<=queries[:, None]
        weights = softmax(np.where(visible[None], logits, -np.inf), axis=-1)
        source_reading = weights[..., source_ids]
        if matched:
            source_reading = matched_endpoint_mean(source_reading, queries, source_ids, prompt_ids)
        reading = source_reading.sum(1)
        mass = reading.sum(-1)
        assert np.all(mass>0)
        priors.append(reading/mass[:, None])
        masses.append(mass/len(queries))
    return np.stack(priors), np.stack(masses), queries, source_ids


def measure_record(row, manifest, output, prior, prior_mass, source_ids):
    attention = np.load(Path(manifest['base'])/row['key']/'attention.npy', mmap_mode='r')
    gradient = np.load(Path(manifest['base'])/row['key']/'derivative.npy', mmap_mode='r')
    result = np.empty((*attention.shape[:3], len(FIELDS)), dtype=np.float32)
    for layer in range(len(attention)):
        reading = np.asarray(attention[layer])[..., source_ids].astype(np.float64)
        effect = np.asarray(gradient[layer])[..., source_ids].astype(np.float64)
        assert np.all(reading>0)
        mass = reading.sum(-1, keepdims=True)
        distribution = reading/mass
        anchor = prior[layer, :, None]
        middle = .5*(distribution+anchor)
        divergence = .5*(xlogy(distribution, distribution/middle)+xlogy(anchor, anchor/middle)).sum(-1)
        delta = mass*anchor-reading
        response = (effect/reading*delta).sum(-1)
        assert np.max(np.abs(delta.sum(-1)))<1e-10
        result[layer] = np.stack((divergence, response, mass[..., 0],
            np.broadcast_to(prior_mass[layer, :, None], mass.shape[:2])), axis=-1)
    np.savez(output/row['key']/'request.npz', values=result, fields=FIELDS)


def readout(values):
    values = values.transpose(2, 0, 1, 3).reshape(values.shape[2], 1024, 4)
    return np.stack((values[..., 0], np.sign(values[..., 1])*np.log1p(np.abs(values[..., 1]))), axis=-1)


def score_task(output, previous, rows):
    scores, raw, reference = {}, {}, []
    for row in rows:
        key = row['key']
        raw[key] = readout(np.load(output/key/'request.npz')['values'])
        with np.load(previous/key/'scores.npz') as saved:
            scores[key] = {name: saved[name] for name in ('source_route_fixed', 'original_full_reference_fixed', 'strong_fused')}
        if row['role']=='fit':
            available = np.flatnonzero(valid_tokens(row, output, OLD))
            chosen = available[np.linspace(0, len(available)-1, 64).astype(int)]
            reference.append(raw[key][chosen])
    reference = np.concatenate(reference)
    center = np.median(reference, axis=0)
    scale = np.maximum(np.quantile(reference, .75, axis=0)-np.quantile(reference, .25, axis=0), 1e-3)
    for key, values in raw.items():
        deviation = (values-center)/scale
        distance = np.log1p(np.maximum(deviation[..., 0], 0))
        effect = np.log1p(np.abs(deviation[..., 1]))
        scores[key]['request_distance'] = sparse_mean(distance)
        scores[key]['request_response'] = sparse_mean(effect)
        scores[key]['request_joint'] = sparse_mean(np.sqrt(distance*effect))
    for name in ('request_distance', 'request_response', 'request_joint'):
        rank = fit_rank(scores, name, rows, output)
        for key in scores:
            scores[key][name+'_fused'] = .75*scores[key]['original_full_reference_fixed']+.25*rank[key]
    thresholds = calibrate(output, rows, scores)
    for name in ('source_route_fixed', 'original_full_reference_fixed', 'strong_fused'):
        thresholds[name] = read_json(previous/'thresholds.json')[rows[0]['task']][name]
    return scores, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('outputs/context_response_20260928_v4'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', choices=('native', 'matched-null'), default='native')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    manifest['prior_mode'] = args.prior
    write_json(args.output/'manifest.json', manifest)
    (args.output/'anchors').mkdir()
    anchors = {}
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        (directory/'context.npz').symlink_to((args.previous/row['key']/'context.npz').resolve())
        identity = row['source_id']
        if identity not in anchors:
            anchors[identity] = anchor_reading(Path(manifest['relation_base'])/'sources'/identity, args.prior=='matched-null')
            prior, mass, queries, ids = anchors[identity]
            np.savez(args.output/'anchors'/(identity+'.npz'), prior=prior, mass=mass, queries=queries, source_positions=ids)
        prior, mass, _, ids = anchors[identity]
        measure_record(row, manifest, args.output, prior, mass, ids)
        print('measured', row['key'], flush=True)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        scores, thresholds[task] = score_task(args.output, args.previous, rows)
        for key, values in scores.items():
            np.savez_compressed(args.output/key/'scores.npz', **values)
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', methods=list(next(iter(scores.values()))),
        main='request_joint_fused', labels_used=False, head_selection='all physical heads, within-token top8',
        prior_mode=args.prior,
        anchor='non-source suffix plus pre-answer query; native or exact lag/copy-matched endpoint mean',
        scope='request-conditioned source-reading prior; not verified evidence; first-order margin response'))


if __name__=='__main__':
    main()
