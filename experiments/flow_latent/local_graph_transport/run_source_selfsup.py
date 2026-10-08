"""Learn source-program compatibility, then freeze full-QA natural token scores.

Program targets come from literal source ownership, never natural hallucination
annotations. A current candidate is supplied explicitly as its native unembedding
vector; this is offline candidate scoring from prechoice states, not a pause
policy. Program fit/dev sources determine normalization, models and thresholds.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap
from .capture import capture_answer
from .data import fit_normalization, load_partition, read_fields
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .source_programs import build_source_programs
from .source_readout import SourceCompatibilityReader


VARIANTS = ('real', 'self', 'rewired', 'native_only_real')
SEED = 42


def snapshot_code(output, name):
    destination = output / name
    destination.mkdir(exist_ok=True)
    sources = [Path(__file__).with_name(filename) for filename in
        ('run_source_selfsup.py', 'source_programs.py', 'source_readout.py',
         'readout.py', 'data.py', 'capture.py', 'messages.py')]
    sources += [Path('experiments/decision_risk_flow/run.py'),
        Path('experiments/decision_risk_flow/precision.py'),
        Path('experiments/flow_latent/provenance_joint_state/measure.py'),
        Path('experiments/token_backtrace/grounded_projection.py')]
    for source in sources:
        target = destination / ('observer_loader.py' if source.name == 'run.py' else source.name)
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError(f'Execution code changed since saved snapshot: {source}')
        if not target.exists():
            shutil.copy2(source, target)
    return {path.name: file_hash(path) for path in destination.iterdir()}


def prepare(args):
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    records, coverage = build_source_programs(args.cache, tokenizer, args.fit_limit,
                                             args.dev_limit, args.value_tokens)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'programs.json', records)
    write_json(args.output / 'coverage.json', coverage)
    code = snapshot_code(args.output, 'prepare_code')
    partitions = {name: sorted({row['source_id'] for row in records
                                if row['partition'] == name}) for name in ('fit', 'dev')}
    write_json(args.output / 'PROTOCOL.json', dict(model=manifest['model'], layer=args.layer,
        fit_limit=args.fit_limit, dev_limit=args.dev_limit, value_tokens=args.value_tokens,
        partitions=partitions, primary='real', controls=['self', 'rewired', 'native_only_real'], seed=SEED,
        natural_labels_used_for_targets_fit_or_selection=False, natural_answers_read=False,
        label_origin='source program literal ownership/prefix compatibility',
        candidate_input='current candidate native unembedding E_y', scalar_inputs='eight zeros',
        timing='prechoice node row t predicts answer y_t; candidate y_t supplied to reader only',
        historical_natural_test_exposure=True, program_sha256=file_hash(args.output / 'programs.json'),
        code_hashes=code))
    print(json.dumps(coverage), flush=True)


def export_unembedding(model, destination):
    """Stream frozen BF16 native rows as FP16; never materialize full FP32 E."""
    weight = model.lm_head.weight
    temporary = destination.with_suffix('.partial.npy')
    mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                      shape=tuple(weight.shape))
    for start in range(0, len(weight), 1024):
        stop = min(start + 1024, len(weight))
        mapped[start:stop] = weight[start:stop].detach().to(device='cpu', dtype=torch.float16).numpy()
    mapped.flush()
    del mapped
    temporary.replace(destination)
    return dict(shape=list(weight.shape), dtype='float16 from frozen native BF16 rows',
                sha256=file_hash(destination), chunk_rows=1024)


def capture(args):
    started = time.time()
    protocol = json.loads((args.output / 'PROTOCOL.json').read_text())
    records = json.loads((args.output / 'programs.json').read_text())
    if file_hash(args.output / 'programs.json') != protocol['program_sha256']:
        raise ValueError('Source program inputs changed after preparation')
    snapshot_code(args.output, 'capture_code')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = ModelAdapter(load_model(protocol['model']))
    fresh = 0
    try:
        embedding_path = args.output / 'native_unembedding.npy'
        if not embedding_path.exists():
            write_json(args.output / 'UNEMBEDDING.json', export_unembedding(adapter.native, embedding_path))
        elif file_hash(embedding_path) != json.loads((args.output / 'UNEMBEDDING.json').read_text())['sha256']:
            raise ValueError('Frozen native unembedding archive changed before program capture')
        for index, record in enumerate(records):
            directory = args.output / 'capture' / record['id']
            if (directory / 'record.json').exists():
                continue
            arrays, audit = capture_answer(adapter, record['prompt_with_source'], record['answer_ids'],
                record['source_mask'], layer=protocol['layer'], stop_after_layer=True)
            for world in audit['worlds']:
                if world['head_reconstruction_max_abs'] > .001 or world['state_equation_max_abs'] > .001:
                    raise ValueError(f"{record['id']}: native state/message reconstruction failed")
                if world['forbidden_attention_max_abs'] != 0:
                    raise ValueError(f"{record['id']}: blocked source reads leaked")
            directory.mkdir(parents=True, exist_ok=True)
            np.savez(directory / 'arrays.npz', **arrays)
            write_json(directory / 'record.json', dict(record, audit=audit))
            fresh += 1
            if index < 4 or (index + 1) % 50 == 0:
                print(f'SOURCE CAPTURE {index + 1}/{len(records)} wall={time.time()-started:.1f}', flush=True)
    finally:
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / 'CAPTURE_COMPLETE.json', dict(records=len(records), new_records=fresh,
        completed_records=sum((args.output / 'capture' / row['id'] / 'record.json').exists() for row in records),
        new_layer_stop_forwards=2 * fresh, full_model_forwards=0, layer=protocol['layer'],
        wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


def program_pack(records):
    """Supply zero scalar normalization rows; no program or natural label read."""
    development = np.concatenate([np.full(row['tokens'], row['partition'] == 'dev') for row in records])
    return dict(scalars=np.zeros((len(development), 8), np.float32), development=development)


def normalize_fields(fields, normalization):
    for name in ('node_fields', 'boundary_fields'):
        fields[name] = (fields[name] - normalization[name + '_mean']) / normalization[name + '_scale']
    fields['scalars'] = np.zeros((len(fields['node_fields']), 8), dtype=np.float32)
    return fields


def concatenate_fields(parts, device):
    result = {}
    for name, values in parts.items():
        axis = 1 if name == 'local_attention' else 0
        tensor = torch.from_numpy(np.concatenate(values, axis=axis)).to(device)
        result[name] = tensor.float() if tensor.is_floating_point() else tensor
    return result


def reader_inputs(fields, variant):
    """Remove paired differences after shared normalization for the native-only control."""
    if variant != 'native_only_real':
        return fields, variant
    native = dict(fields)
    native['node_fields'] = fields['node_fields'].clone()
    native['node_fields'][:, 3:] = 0
    native['boundary_fields'] = fields['boundary_fields'].clone()
    native['boundary_fields'][:, 5:] = 0
    native['value_fields'] = fields['value_fields'].clone()
    head_dimension = fields['value_fields'].shape[-1] // 2
    native['value_fields'][..., head_dimension:] = 0
    native['local_attention'] = fields['local_attention'][0:1].expand_as(fields['local_attention']).clone()
    return native, 'real'


def program_batch(records, capture_root, normalization, embedding, device='cuda'):
    """Disjoint graphs; prechoice rows align to explicitly supplied candidate IDs."""
    parts = {}
    selected_rows, candidates, labels, token_sources = [], [], [], []
    offset = 0
    for record in records:
        fields = normalize_fields(read_fields(capture_root / record['id']), normalization)
        fields['indices'] += offset
        for name, values in fields.items():
            parts.setdefault(name, []).append(values)
        rows = np.flatnonzero(record['valid'])
        selected_rows.append(rows + offset)
        candidates.extend(np.asarray(record['answer_ids'])[rows])
        labels.extend(np.asarray(record['labels'])[rows])
        token_sources.extend([record['source_id']] * len(rows))
        offset += len(fields['node_fields'])
    fields = concatenate_fields(parts, device)
    fields['candidate_vectors'] = torch.from_numpy(np.array(embedding[candidates], dtype=np.float32)).to(device)
    fields['rows'] = torch.from_numpy(np.concatenate(selected_rows)).to(device)
    return fields, np.asarray(labels, dtype=np.float32), np.asarray(token_sources)


def program_source_weights(records):
    counts = {}
    for row in records:
        counts[row['source_id']] = counts.get(row['source_id'], 0) + sum(row['valid'])
    total = sum(counts.values())
    return {source: total / (len(counts) * count) for source, count in counts.items()}, total


@torch.no_grad()
def program_predictions(models, records, capture_root, normalization, embedding, batch_answers):
    predictions = {variant: {} for variant in VARIANTS}
    for model in models.values():
        model.eval()
    for start in range(0, len(records), batch_answers):
        selected = records[start:start + batch_answers]
        fields, _, _ = program_batch(selected, capture_root, normalization, embedding)
        for variant, model in models.items():
            inputs, graph_variant = reader_inputs(fields, variant)
            values = model(**inputs, variant=graph_variant).cpu().numpy()
            offset = 0
            for row in selected:
                count = sum(row['valid'])
                predictions[variant][row['id']] = values[offset:offset + count]
                offset += count
    return predictions


def program_metrics(records, predictions):
    labels = np.concatenate([np.asarray(row['labels'])[row['valid']] for row in records])
    scores = np.concatenate([predictions[row['id']] for row in records])
    result = ranking(labels, scores)
    result.update(bce=float(np.mean(np.logaddexp(0, scores) - labels * scores)),
        accuracy_at_zero=float(np.mean((scores > 0) == labels)),
        compatible95_threshold=float(np.quantile(scores[labels == 0], .95, method='higher')))
    grouped = {}
    for row in records:
        grouped.setdefault((row['source_id'], row['proposal_index']), []).append(row)
    differences, both_correct = [], []
    for pair in grouped.values():
        if len(pair) != 2:
            continue
        correct = next(row for row in pair if row['proposal_index'] == row['correct_index'])
        wrong = next(row for row in pair if row['proposal_index'] != row['correct_index'])
        low = float(predictions[correct['id']][0])
        high = float(predictions[wrong['id']][0])
        differences.append(high - low)
        both_correct.append(low <= 0 < high)
    result['swap_first_divergence'] = dict(pairs=len(differences),
        wrong_above_correct=float(np.mean(np.asarray(differences) > 0)),
        both_signs_correct=float(np.mean(both_correct)), mean_risk_difference=float(np.mean(differences)))
    return result


def fit_batch(models, optimizers, fields, labels, token_sources, weights, denominator, step):
    targets = torch.from_numpy(labels).to('cuda')
    weight = torch.tensor([weights[source] for source in token_sources], device='cuda')
    losses = {}
    for variant, model in models.items():
        model.train()
        torch.manual_seed(SEED + step)
        optimizers[variant].zero_grad(set_to_none=True)
        inputs, graph_variant = reader_inputs(fields, variant)
        logits = model(**inputs, variant=graph_variant)
        loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * weight).sum() / denominator
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizers[variant].step()
        losses[variant] = float(loss.detach())
    return losses


def save_best_models(directory, models, normalization, metrics, predictions, best, epoch):
    for variant, result in metrics.items():
        if result['auroc'] > best[variant]:
            best[variant] = result['auroc']
            torch.save(dict(state_dict=models[variant].state_dict(), epoch=epoch,
                normalization=normalization, variant=variant, dev_metrics=result,
                labels='source_program_only', seed=SEED), directory / f'{variant}.pt')
            np.savez(directory / f'{variant}_dev_scores.npz', **predictions[variant])


def fit(args):
    started = time.time()
    records = json.loads((args.output / 'programs.json').read_text())
    ledger = json.loads((args.output / 'CAPTURE_COMPLETE.json').read_text())
    if ledger['completed_records'] != len(records):
        raise ValueError('Source-program capture is incomplete')
    directory = args.output / 'fits'
    directory.mkdir(exist_ok=False)
    code = snapshot_code(args.output, 'fit_code')
    normalization = fit_normalization(args.output / 'capture', records, program_pack(records))
    np.savez(directory / 'normalization.npz', **normalization)
    embedding = np.load(args.output / 'native_unembedding.npy', mmap_mode='r')
    fit_records = [row for row in records if row['partition'] == 'fit']
    dev_records = [row for row in records if row['partition'] == 'dev']
    weights, total = program_source_weights(fit_records)
    mean_tokens = total / len(fit_records)
    models, optimizers = {}, {}
    for variant in VARIANTS:
        torch.manual_seed(SEED)
        models[variant] = SourceCompatibilityReader().to('cuda')
        optimizers[variant] = torch.optim.AdamW(models[variant].parameters(),
            lr=args.learning_rate, weight_decay=.01)
    write_json(directory / 'FIT_PROTOCOL.json', dict(epochs=args.epochs, learning_rate=args.learning_rate,
        batch_answers=args.batch_answers, seed=SEED, variants=VARIANTS, primary='real',
        parameter_count=sum(p.numel() for p in models['real'].parameters()),
        objective='equal-source program incompatible-prefix BCE; eight scalar inputs zero',
        selection='best per-model source-heldout program dev token AUROC; no natural labels',
        normalization_fit_partition='program fit only', normalization_sha256=file_hash(directory / 'normalization.npz'),
        program_sha256=file_hash(args.output / 'programs.json'), code_hashes=code))
    best = {variant: -float('inf') for variant in VARIANTS}
    history = []
    for epoch in range(args.epochs):
        order = np.random.default_rng(SEED + epoch).permutation(len(fit_records))
        for start in range(0, len(order), args.batch_answers):
            selected = [fit_records[index] for index in order[start:start + args.batch_answers]]
            fields, labels, sources = program_batch(selected, args.output / 'capture', normalization, embedding)
            fit_batch(models, optimizers, fields, labels, sources, weights,
                      mean_tokens * len(selected), epoch * 100000 + start)
            if start % 400 == 0:
                print(f'SOURCE FIT epoch={epoch + 1}/{args.epochs} records={start}/{len(order)} '
                      f'wall={time.time()-started:.1f}', flush=True)
        predictions = program_predictions(models, dev_records, args.output / 'capture', normalization,
                                          embedding, args.batch_answers)
        metrics = {variant: program_metrics(dev_records, values) for variant, values in predictions.items()}
        save_best_models(directory, models, normalization, metrics, predictions, best, epoch + 1)
        history.append(dict(epoch=epoch + 1, metrics=metrics))
        write_json(directory / 'HISTORY.json', history)
        print(f'SOURCE DEV epoch={epoch + 1} {json.dumps(metrics)}', flush=True)
    write_json(directory / 'COMPLETE.json', dict(fits=len(VARIANTS), program_fit_records=len(fit_records),
        program_dev_records=len(dev_records), program_fit_valid_tokens=total,
        natural_label_fits=0, best_program_dev_auroc=best, wall_seconds=time.time()-started))


def natural_batch(records, capture_root, pack, normalization, embedding):
    """Use prechoice graph nodes plus candidate E_y; legacy scalar inputs stay zero."""
    parts = {}
    rows, candidates, packed = [], [], []
    offset = 0
    for record in records:
        fields = normalize_fields(read_fields(capture_root / record['id']), normalization)
        fields['indices'] += offset
        for name, values in fields.items():
            parts.setdefault(name, []).append(values)
        selected = np.arange(record['packed_start'], record['packed_stop'])
        rows.append(pack['target'][selected] + offset)
        candidates.extend(pack['token_id'][selected])
        packed.append(selected)
        offset += len(fields['node_fields'])
    fields = concatenate_fields(parts, 'cuda')
    fields['candidate_vectors'] = torch.from_numpy(np.array(embedding[candidates], dtype=np.float32)).to('cuda')
    fields['rows'] = torch.from_numpy(np.concatenate(rows)).to('cuda')
    return fields, np.concatenate(packed)


@torch.no_grad()
def natural_predictions(models, records, capture_root, pack, normalization, embedding, batch_answers):
    scores = {variant: np.full(len(pack['token_id']), np.nan) for variant in VARIANTS}
    for model in models.values():
        model.eval()
    for start in range(0, len(records), batch_answers):
        selected = records[start:start + batch_answers]
        fields, indices = natural_batch(selected, capture_root, pack, normalization, embedding)
        for variant, model in models.items():
            inputs, graph_variant = reader_inputs(fields, variant)
            scores[variant][indices] = model(**inputs, variant=graph_variant).cpu().numpy()
        if start % 100 == 0:
            print(f'SOURCE TRANSFER answers={start}/{len(records)}', flush=True)
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError('Source-program transfer has incomplete/nonfinite natural scores')
    return scores


def evaluate_thresholds(directory, test, scores, thresholds):
    policies = dict(zero={variant: 0. for variant in VARIANTS}, program_correct95=thresholds)
    results = {}
    for name, limits in policies.items():
        metrics = evaluate_all(test, scores, limits)
        for result in metrics.values():
            result['threshold_rule'] = ('fixed_risk_logit_zero_strict_gt' if name == 'zero'
                                        else 'source_program_dev_compatible95_strict_gt')
            result['natural_labels_used_for_fit_or_threshold'] = False
        write_json(directory / f'test_metrics_{name}.json', metrics)
        results[name] = metrics
    return results, policies


def natural_generator_metrics(pack, metadata, scores, policies):
    grouped = {}
    for generator in sorted({row['generator'] for row in metadata['records']}):
        selected = np.concatenate([np.arange(row['packed_start'], row['packed_stop'])
            for row in metadata['records'] if row['generator'] == generator])
        subset = {name: value[selected] for name, value in pack.items()}
        grouped[generator] = {}
        for policy, thresholds in policies.items():
            metrics = evaluate_all(subset, {name: value[selected] for name, value in scores.items()}, thresholds)
            for result in metrics.values():
                result['threshold_rule'] = policy
                result['natural_labels_used_for_fit_or_threshold'] = False
            grouped[generator][policy] = metrics
    return grouped


def evaluate(args):
    started = time.time()
    directory = args.output / 'natural_test'
    directory.mkdir(exist_ok=False)
    code = snapshot_code(args.output, 'natural_score_code')
    test, metadata = load_partition('test')
    program_protocol = json.loads((args.output / 'PROTOCOL.json').read_text())
    used_sources = set(program_protocol['partitions']['fit']) | set(program_protocol['partitions']['dev'])
    if used_sources & set(metadata['sources']):
        raise ValueError('Natural test sources overlap source-program fitting or selection')
    capture_protocol = json.loads((args.natural_capture.parent / 'CAPTURE_PROTOCOL.json').read_text())
    if capture_protocol['layer'] != program_protocol['layer']:
        raise ValueError('Program and natural captures use different selected layers')
    models, thresholds = {}, {}
    for variant in VARIANTS:
        checkpoint = torch.load(args.output / 'fits' / f'{variant}.pt', map_location='cuda', weights_only=False)
        model = SourceCompatibilityReader().to('cuda')
        model.load_state_dict(checkpoint['state_dict'])
        models[variant] = model
        thresholds[variant] = checkpoint['dev_metrics']['compatible95_threshold']
    normalization = checkpoint['normalization']
    embedding = np.load(args.output / 'native_unembedding.npy', mmap_mode='r')
    scores = natural_predictions(models, metadata['records'], args.natural_capture,
        test, normalization, embedding, args.natural_batch_answers)
    np.savez(directory / 'scores.npz', **scores)
    write_json(directory / 'FREEZE.json', dict(status='all_natural_test_predictions_frozen',
        test_answers=len(metadata['records']), test_sources=len(metadata['sources']),
        test_tokens=len(test['token_id']), prediction_sha256=file_hash(directory / 'scores.npz'),
        checkpoints={variant: file_hash(args.output / 'fits' / f'{variant}.pt') for variant in VARIANTS},
        normalization_sha256=file_hash(args.output / 'fits' / 'normalization.npz'),
        program_sha256=file_hash(args.output / 'programs.json'), code_hashes=code,
        native_unembedding_sha256=file_hash(args.output / 'native_unembedding.npy'),
        primary='real', program_correct95_thresholds=thresholds, natural_labels_used_for_fit_selection=False,
        timing='offline observed-candidate compatibility from prechoice observer states',
        zero_scalar_features=True, historical_natural_test_exposure=True))
    test.update(evaluation_labels(args.cache, test, metadata))
    metrics, policies = evaluate_thresholds(directory, test, scores, thresholds)
    write_json(directory / 'test_by_generator.json', natural_generator_metrics(test, metadata, scores, policies))
    comparisons = {f'real_minus_{control}': source_bootstrap(test, scores['real'], scores[control], args.bootstrap)
                   for control in ('self', 'rewired', 'native_only_real')}
    write_json(directory / 'bootstrap.json', comparisons)
    records = json.loads((args.output / 'programs.json').read_text())
    fit_ids = {token for row in records if row['partition'] == 'fit' for token in row['answer_ids']}
    write_json(directory / 'COMPLETE.json', dict(natural_label_fits=0, program_fits=len(VARIANTS),
        test_answers=len(metadata['records']), test_tokens=len(test['token_id']),
        natural_candidate_fit_token_coverage=float(np.isin(test['token_id'], list(fit_ids)).mean()),
        natural_test_sources_disjoint=True, primary='real', wall_seconds=time.time()-started))
    print(json.dumps({name: dict(auroc=result['auroc'], ap=result['ap'],
        within_answer=result['within_answer_auroc']) for name, result in metrics['zero'].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'capture', 'fit', 'evaluate', 'all'))
    parser.add_argument('--output', type=Path, default=OUTPUT / 'source_selfsup')
    parser.add_argument('--natural-capture', type=Path, default=OUTPUT / 'capture')
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--fit-limit', type=int)
    parser.add_argument('--dev-limit', type=int)
    parser.add_argument('--value-tokens', type=int, default=12)
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--batch-answers', type=int, default=16)
    parser.add_argument('--natural-batch-answers', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(4)
    phases = dict(prepare=prepare, capture=capture, fit=fit, evaluate=evaluate)
    for name in phases if args.phase == 'all' else [args.phase]:
        phases[name](args)


if __name__ == '__main__':
    main()
