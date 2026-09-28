"""Read native cached tensors without annotation-dependent head selection."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import inputs, read_json, write_json

BLOCKS = ('geometry_head', 'geometry_gram', 'js', 'jacobian_head', 'tangent', 'fisher')


def surface_category(piece):
    word = piece.strip()
    if not word:
        return 0
    if any(character.isdigit() for character in word):
        return 1
    if not any(character.isalnum() for character in word):
        return 2
    if word[0].isupper():
        return 3
    return 4


def token_context(row, manifest, confidence, route):
    if row['kind'] == 'observer':
        prompt_ids, response = inputs(row)
        ids = response['answer_ids']
        pieces = response['token_text']
    else:
        with np.load(Path(manifest['samples']) / row['trace']) as saved:
            prompt = int(saved['prompt_length'])
            prompt_ids = saved['token_ids'][:prompt].tolist()
            ids = saved['token_ids'][prompt:]
            pieces = saved['token_text'][prompt:]
    count = len(ids)
    position = np.arange(count)
    membership = np.isin(ids, prompt_ids).astype(float)
    # Surface categories require neither prompt annotation nor an entity model.
    surface = []
    for piece in pieces:
        surface.append(np.eye(5)[surface_category(piece)])
    numeric = np.stack((confidence[:, 0], confidence[:, 1], np.log1p(position),
        position / max(count - 1, 1), np.full(count, np.log1p(len(prompt_ids))), route, membership), -1)
    return np.concatenate((numeric, np.asarray(surface)), -1)


def geometry_blocks(directory):
    with np.load(directory / 'head_features.npz') as saved:
        heads = saved['values']
    count = heads.shape[2]
    geometry = heads[..., :4].transpose(2, 0, 1, 3).reshape(count, -1)
    gram = np.load(directory / 'head_gram.npy', mmap_mode='r')[:, :2]
    norms = np.sqrt(np.maximum(np.diagonal(gram, axis1=-2, axis2=-1), 0))
    cosine = gram / np.maximum(norms[..., :, None] * norms[..., None, :], 1e-20)
    left, right = np.triu_indices(32, 1)
    pairs = cosine[..., left, right].transpose(2, 0, 1, 3).reshape(count, -1)
    return geometry, pairs, heads[..., 4:6]


def read_features(base, manifest, row):
    directory = base / row['key']
    original = Path(manifest['base']) / row['key']
    operator = Path(manifest['operator']) / row['key']
    geometry, pairs, source_js = geometry_blocks(directory)
    with np.load(original / 'readouts.npz') as data, np.load(operator / 'operator.npz') as native:
        count = len(data['token_ids'])
        raw = data['measured']
        js = np.concatenate((source_js, raw[..., :3]), -1).transpose(2, 0, 1, 3).reshape(count, -1)
        magnitudes = raw[..., 4:8]
        signed_parts = magnitudes / np.maximum(magnitudes.sum(-1, keepdims=True), 1e-20)
        cancel = np.stack((native['prompt_cancel'], native['history_cancel']), -1)
        jacobian = np.concatenate((signed_parts, cancel), -1).transpose(2, 0, 1, 3).reshape(count, -1)
        tangent = native['tangent']
        tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-20)
        gram = native['gram']
        energy = np.maximum(np.diagonal(gram, axis1=-2, axis2=-1), 0)
        cosine = gram / np.maximum(np.sqrt(energy[:, :, None] * energy[:, None, :]), 1e-20)
        left, right = np.triu_indices(3, 1)
        fisher = np.concatenate((np.log1p(energy), cosine[:, left, right], np.arcsinh(native['margin'])), -1)
        with np.load(directory / 'scores.npz') as scores:
            context = token_context(row, manifest, data['confidence'], scores['raw_route'])
        blocks = dict(geometry_head=geometry, geometry_gram=pairs, js=js,
            jacobian_head=jacobian, tangent=tangent.reshape(count, -1), fisher=fisher)
        return blocks, context, data['token_ids'].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('outputs/route_complement_20260928'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.base / 'manifest.json')
    args.output.mkdir(exist_ok=False)
    write_json(args.output / 'manifest.json', dict(manifest, route_base=str(args.base.resolve())))
    for row in manifest['records']:
        directory = args.output / row['key']
        directory.mkdir()
        blocks, context, ids = read_features(args.base, manifest, row)
        for name, values in blocks.items():
            np.save(directory / (name + '.npy'), values.astype(np.float32))
        np.savez_compressed(directory / 'context.npz', values=context, token_ids=ids)
        print(row['key'], len(ids), {name: value.shape[1] for name, value in blocks.items()}, flush=True)
    write_json(args.output / 'features_complete.json', dict(status='complete', labels_read=False,
        blocks=BLOCKS, head_selection=False, reference='original full attention/value/tangent caches preserved',
        operator_scope='Native choice-gradient per head; full hidden JVP in three specified global directions, not full Jacobian',
        missing='undefined relay JS stays NaN until reference-median imputation with missingness feature'))


if __name__ == '__main__':
    main()
