"""Train first-only source address controls using existing source-program captures.

No natural answer annotations are read. Both source worlds stay in one split.
The primary address reader is compared with equally sized group/payload controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.native_support.evaluate import ranking
from .first_address import FirstAddressReader, first_choice_loss


ROOT = Path('outputs/supervised_local_transport_20261008')
VARIANTS = ('group_summed', 'address', 'payload_rewired')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_fields(arrays, position, normalization):
    """Select one pre-choice receiver, retaining all native/difference sites."""
    nodes = arrays['nodes'][:, position].astype(np.float32)
    groups = arrays['group_heads'][:, position].astype(np.float32).reshape(2, 5, -1)
    result = dict(node_fields=np.concatenate([nodes[0], nodes[0] - nodes[1]]),
                  boundary_fields=np.concatenate([groups[0], groups[0] - groups[1]]))
    for name in result:
        result[name] = (result[name] - normalization[name + '_mean']) / normalization[name + '_scale']
    result['query_blocked'] = nodes[1].sum(0)
    result['query_native'] = nodes[0].sum(0)
    # Native source AV retains query-head coordinates until GQA groups are
    # averaged to the matching 8*128 source V basis used by all source maps.
    result['summed_payload'] = groups[0, 0].reshape(8, 4, 128).mean(1).flatten()
    return result


def prepare(args):
    started = time.time()
    records = json.loads((args.choices / 'choices.json').read_text())
    normalization = dict(np.load(args.parent / 'fits/normalization.npz'))
    embedding = np.load(args.parent / 'native_unembedding.npy', mmap_mode='r')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / 'cache'
    cache.mkdir()
    fields, keys, values, positions, token_ids = {}, [], [], [], []
    selected_records, bounds, capture_hashes = [], [0], {}
    for index, record in enumerate(records):
        position = record['first_divergence']
        pair = [j for j, first in enumerate(record['first']) if first]
        path = args.parent / 'capture' / record['id'] / 'arrays.npz'
        with np.load(path) as arrays:
            if arrays['answer_ids'].tolist() != record['answer_ids']:
                raise ValueError('Candidate histories do not match declared correct capture')
            current = selected_fields(arrays, position, normalization)
            source = np.flatnonzero(arrays['source_mask'])
            keys.append(arrays['key'][0, source].reshape(len(source), -1))
            values.append(arrays['value'][0, source].reshape(len(source), -1))
            positions.append(source)
            token_ids.append(arrays['token_ids'][source])
        for name, value in current.items():
            fields.setdefault(name, []).append(value)
        candidates = [record['candidate_ids'][j] for j in pair]
        fields.setdefault('candidate_vectors', []).append(np.asarray(embedding[candidates]))
        bounds.append(bounds[-1] + len(source))
        selected_records.append(dict(id=record['id'], source_id=record['source_id'],
            partition=record['partition'], world=record['world'], candidate_ids=candidates,
            labels=[record['labels'][j] for j in pair], first_divergence=position))
        capture_hashes[record['id']] = file_hash(path)
        if index % 200 == 0:
            print(f'ADDRESS PREPARE graphs={index + 1}/{len(records)} wall={time.time()-started:.1f}', flush=True)
    for name, value in fields.items():
        np.save(cache / (name + '.npy'), np.stack(value).astype(np.float32))
    for name, value in (('source_key', keys), ('source_value', values),
                        ('source_positions', positions), ('source_token_ids', token_ids)):
        np.save(cache / (name + '.npy'), np.concatenate(value))
    np.save(cache / 'source_bounds.npy', np.asarray(bounds))
    write_json(cache / 'records.json', selected_records)
    freeze_prepare(args, cache, capture_hashes, started)


def freeze_prepare(args, cache, capture_hashes, started):
    sources = {split: sorted({row['source_id'] for row in json.loads((cache / 'records.json').read_text())
                              if row['partition'] == split}) for split in ('fit', 'dev')}
    if set(sources['fit']) & set(sources['dev']):
        raise ValueError('Source heldout split overlap')
    snapshot = args.output / 'prepare_code'
    snapshot.mkdir()
    for name in ('run_first_address.py', 'first_address.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    write_json(args.output / 'PROTOCOL.json', dict(primary='address', variants=VARIANTS,
        sources=sources, first_only=True, normalization='parent fit-source full-coordinate normalization',
        query='raw blocked residual + attention_write + MLP_write at prechoice receiver',
        source='native contextual raw pre-RoPE K/V; each physical coordinate retained',
        controls='same parameters; group pooled-key gate with summed AV; payload rewire fixes keys',
        group_capacity_note='same parameter count and maps, different allowed address interactions',
        objective='equal-source first-only BCE + softplus(correct_risk-wrong_risk)',
        natural_labels_used=False, new_observer_forwards=0,
        input_sha256={'choices': file_hash(args.choices / 'choices.json'),
            'normalization': file_hash(args.parent / 'fits/normalization.npz'),
            'unembedding': file_hash(args.parent / 'native_unembedding.npy')},
        capture_sha256=capture_hashes, cache_sha256={path.name: file_hash(path) for path in cache.iterdir()},
        wall_seconds=time.time()-started))
    print(json.dumps({split: len(ids) for split, ids in sources.items()}), flush=True)


def load_cache(output):
    cache = output / 'cache'
    protocol = json.loads((output / 'PROTOCOL.json').read_text())
    if any(file_hash(cache / name) != digest for name, digest in protocol['cache_sha256'].items()):
        raise ValueError('Frozen first-choice inputs changed')
    fields = {path.stem: np.load(path, mmap_mode='r') for path in cache.glob('*.npy')}
    records = json.loads((cache / 'records.json').read_text())
    return fields, records, protocol


def address_batch(fields, selected, device):
    """Only real senders participate in padded source attention."""
    bounds = fields['source_bounds']
    lengths = bounds[selected + 1] - bounds[selected]
    width = int(lengths.max())
    source_dim = fields['source_key'].shape[1]
    keys = np.zeros((len(selected), width, source_dim), np.float32)
    values = np.zeros_like(keys)
    valid = np.arange(width)[None] < lengths[:, None]
    for row, index in enumerate(selected):
        start, stop = bounds[index:index + 2]
        keys[row, :lengths[row]] = fields['source_key'][start:stop]
        values[row, :lengths[row]] = fields['source_value'][start:stop]
    names = ('node_fields', 'boundary_fields', 'summed_payload', 'candidate_vectors')
    batch = {name: torch.from_numpy(np.array(fields[name][selected])).to(device) for name in names}
    batch['query'] = torch.from_numpy(np.array(fields['query_blocked'][selected])).to(device)
    batch.update(source_key=torch.from_numpy(keys).to(device),
        source_value=torch.from_numpy(values).to(device), source_valid=torch.from_numpy(valid).to(device))
    return batch


def metrics(records, scores):
    result = ranking(np.tile([0, 1], len(records)), scores.flatten())
    result['accuracy_at_zero'] = float(np.mean((scores > 0) == np.asarray([0, 1])))
    result['bce'] = float(np.mean(np.logaddexp(0, scores) - scores * np.asarray([0, 1])))
    result['candidate_pair_order'] = float(np.mean(scores[:, 1] > scores[:, 0]))
    swaps = {}
    for record, pair in zip(records, scores):
        for candidate, label, score in zip(record['candidate_ids'], [0, 1], pair):
            swaps.setdefault((record['source_id'], candidate), {})[label] = float(score)
    if any(set(pair) != {0, 1} for pair in swaps.values()):
        raise ValueError('Each candidate needs one compatible and one incompatible source world')
    differences = np.asarray([pair[1] - pair[0] for pair in swaps.values()])
    result['source_swap'] = dict(pairs=len(swaps), wrong_above_correct=float(np.mean(differences > 0)),
        both_signs_correct=float(np.mean([pair[0] <= 0 < pair[1] for pair in swaps.values()])),
        mean_risk_difference=float(differences.mean()))
    return result


@torch.no_grad()
def predictions(model, variant, fields, selected, batch_size, device):
    model.eval()
    result = []
    for start in range(0, len(selected), batch_size):
        batch = address_batch(fields, selected[start:start + batch_size], device)
        result.append(model(**batch, variant=variant).cpu().numpy())
    return np.concatenate(result)


def evaluate_epoch(models, fields, records, args):
    result, scores = {}, {}
    for split in ('fit', 'dev'):
        selected = np.asarray([i for i, row in enumerate(records) if row['partition'] == split])
        selected_records = [records[i] for i in selected]
        result[split], scores[split] = {}, {}
        for variant, model in models.items():
            values = predictions(model, variant, fields, selected, args.batch_graphs, args.device)
            result[split][variant] = metrics(selected_records, values)
            scores[split][variant] = values
    return result, scores


def fit(args):
    started = time.time()
    fields, records, protocol = load_cache(args.output)
    destination = args.output / (args.run_name or ('fits_seed' + str(args.seed)))
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    models, optimizers = {}, {}
    for variant in VARIANTS:
        torch.manual_seed(args.seed)
        models[variant] = FirstAddressReader(dropout=args.dropout,
            address_energy=args.address_energy, address_temperature=args.address_temperature).to(args.device)
        optimizers[variant] = torch.optim.AdamW(models[variant].parameters(), lr=args.learning_rate, weight_decay=.01)
    settings = dict(seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
        dropout=args.dropout, pair_weight=args.pair_weight, batch_graphs=args.batch_graphs,
        address_energy=args.address_energy, address_temperature=args.address_temperature,
        parameter_count={variant: sum(p.numel() for p in model.parameters()) for variant, model in models.items()},
        selection='source-heldout first-only AUROC; strict greater keeps earlier ties',
        natural_label_fits=0, program_fits=len(models), new_observer_forwards=0)
    write_json(destination / 'FIT_PROTOCOL.json', settings)
    for name in ('run_first_address.py', 'first_address.py'):
        shutil.copy2(Path(__file__).with_name(name), destination / name)
    selected = np.asarray([i for i, row in enumerate(records) if row['partition'] == 'fit'])
    history, best = [], {variant: -float('inf') for variant in VARIANTS}
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(selected)
        for start in range(0, len(order), args.batch_graphs):
            batch = address_batch(fields, order[start:start + args.batch_graphs], args.device)
            for variant, model in models.items():
                model.train()
                torch.manual_seed(args.seed + epoch * 100000 + start)
                optimizers[variant].zero_grad(set_to_none=True)
                loss = first_choice_loss(model(**batch, variant=variant), args.pair_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizers[variant].step()
        results, scores = evaluate_epoch(models, fields, records, args)
        history.append(dict(epoch=epoch + 1, metrics=results, wall_seconds=time.time()-started))
        for variant, model in models.items():
            if results['dev'][variant]['auroc'] > best[variant]:
                best[variant] = results['dev'][variant]['auroc']
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch + 1,
                    fit_metrics=results['fit'][variant], dev_metrics=results['dev'][variant],
                    settings=settings), destination / (variant + '.pt'))
                np.savez(destination / (variant + '_scores.npz'), **{split: scores[split][variant] for split in scores})
        write_json(destination / 'HISTORY.json', history)
        print(f'ADDRESS EPOCH {epoch + 1}/{args.epochs} ' + json.dumps(results), flush=True)
    write_json(destination / 'COMPLETE.json', dict(settings, best_dev_auroc=best,
        wall_seconds=time.time()-started, cache_protocol_sha256=file_hash(args.output / 'PROTOCOL.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'fit'))
    parser.add_argument('--parent', type=Path, default=ROOT / 'source_selfsup')
    parser.add_argument('--choices', type=Path, default=ROOT / 'source_choices_v2')
    parser.add_argument('--output', type=Path, default=Path('outputs/first_address_validation_20261008'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--batch-graphs', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--pair-weight', type=float, default=1.)
    parser.add_argument('--address-energy', choices=('dot', 'cosine'), default='dot')
    parser.add_argument('--address-temperature', type=float, default=8.)
    parser.add_argument('--run-name')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    dict(prepare=prepare, fit=fit)[args.phase](args)


if __name__ == '__main__':
    main()
