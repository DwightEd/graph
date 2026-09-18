"""Matched loss interventions: original inputs, fixed epochs, explicit resume.

All four losses preserve the original labels. Same seed means the same model
initialization, answer order and update budget. Final epoch, not pooled-AP selection.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from .model import CHARM
from .data import read_json, write_json, load_graph, original_parts, token_frame
from .positions import merge_spans
from .train import calibrate
from .continuity_weights import prepare_weights
from .continuity import save_analysis, plots


ROLE_NAMES = ('normal', 'onset', 'continuation')


def load_input(record, prepared, variant):
    """Node path reads no edges. Metadata path reads no model inputs."""
    path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
    with np.load(path, allow_pickle=False) as saved:
        identity = read_identity(saved, record)
        sample = {key: saved[key] for key in ('gold', 'spans', 'offsets', 'response')}
        sample.update(id=str(identity['id']), source_id=str(identity['source_id']))
        sample['onset'] = np.zeros(len(sample['gold']), bool)
        for start, _ in merge_spans(sample['spans']):
            sample['onset'][start] = True
        if variant == 'metadata':
            return None, sample
        if variant == 'node_only':
            return saved['x'][int(saved['prompt_length']):].copy(), sample
    graph, _ = load_graph(record, prepared)
    return graph, sample


def read_identity(saved, record):
    import json
    identity = json.loads(str(saved['record_json']))
    if any(str(identity[key]) != str(record[key]) for key in ('id', 'source_id')):
        raise ValueError('Prepared input identity differs from the registered partition')
    return identity


def node_logits(model, values):
    state = model.in_proj(values).relu()
    for layer in model.mp_layers:
        state = layer.update(state, torch.zeros_like(state))
    return model.pred(state).view(-1)


def score_record(model, inputs, variant):
    if variant == 'node_only':
        values = torch.as_tensor(inputs, dtype=torch.float32, device=model.in_proj.weight.device)
        return node_logits(model, values)
    return model(inputs)[int(inputs['prompt_length']):]


def predict(model, records, prepared, variant):
    samples = []
    model.eval()
    with torch.no_grad():
        for record in records:
            inputs, sample = load_input(record, prepared, variant)
            sample['score'] = score_record(model, inputs, variant).sigmoid().cpu().numpy()
            samples.append(sample)
    return samples


def measure_gradients(logits, labels, weights, alpha, onset, raw, gradient_sum, loss_sum):
    """ROLE_NAMES orders exact |d(unreduced weighted BCE)/dz|, NOT parameter gradients."""
    with torch.no_grad():
        probability = logits.sigmoid()
        derivative = weights * ((1 - labels) * probability + alpha * labels * (probability - 1))
        is_onset = torch.as_tensor(onset, device=labels.device)
        masks = (labels == 0, is_onset, (labels == 1) & ~is_onset)
        for index, mask in enumerate(masks):
            gradient_sum[index] += float(derivative[mask].abs().sum())
            loss_sum[index] += float((raw * weights)[mask].sum())


def backward_answer(model, record, args, weights, variant, alpha, denominator, role_gradient, role_loss):
    inputs, sample = load_input(record, args.prepared, variant)
    logits = score_record(model, inputs, variant)
    labels = torch.as_tensor(sample['gold'], dtype=logits.dtype, device=logits.device)
    weight = torch.as_tensor(weights[str(record['id'])], device=logits.device)
    raw = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=alpha, reduction='none')
    loss = (raw * weight).sum() / denominator
    loss.backward()
    measure_gradients(logits, labels, weight, alpha, sample['onset'], raw, role_gradient, role_loss)
    return float(loss.detach())


def epoch_train(model, records, args, recipe, weights, variant, epoch, optimizer, scheduler, positive_weight):
    order = np.random.default_rng(recipe['seed'] + epoch).permutation(len(records))
    loss_total = 0.
    role_gradient = np.zeros(len(ROLE_NAMES))
    role_loss = np.zeros(len(ROLE_NAMES))
    model.train()
    batches = range(0, len(records), recipe['batch_size'])
    for start in tqdm(batches, desc=f'{variant} epoch {epoch + 1}', unit='batch'):
        batch = [records[index] for index in order[start:start + recipe['batch_size']]]
        denominator = sum(row['response_tokens'] for row in batch)
        optimizer.zero_grad(set_to_none=True)
        for record in batch:
            loss_total += backward_answer(model, record, args, weights, variant, positive_weight,
                                          denominator, role_gradient, role_loss)
        optimizer.step()
        scheduler.step()
    return dict(loss=loss_total, logit_abs_gradient=role_gradient.tolist(), weighted_loss=role_loss.tolist())


def make_optimizer(model, recipe, fit_count):
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=.001)
    total_steps = recipe['epochs'] * math.ceil(fit_count / recipe['batch_size'])
    warmup = max(1, int(.05 * total_steps))

    def schedule(step):
        if step < warmup:
            return step / warmup
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_epoch(path, model, optimizer, scheduler, history):
    """Write one completed optimization epoch atomically; do not mark evaluation complete."""
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 history=history, cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(path)


def resume_epoch(path, model, optimizer, scheduler):
    if not path.is_file():
        return []
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['cpu_rng'])
    if saved['cuda_rng'] and next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    print(f'Resume after completed epoch {len(saved["history"])}: {path.parent}', flush=True)
    return saved['history']


def create_model(args, parts, variant, recipe):
    torch.manual_seed(recipe['seed'])
    inputs, _ = load_input(parts['fit'][0], args.prepared, variant)
    width = inputs.shape[1] if variant == 'node_only' else inputs['x'].shape[1]
    hp = dict(hidden_dim=recipe['hidden_dim'], gnn_layers=recipe['gnn_layers'], residual_mp=True)
    return CHARM(width, width, hp, 'in', args.edge_chunk).to(args.device), hp


def fit_model(args, parts, variant, scheme, recipe, directory):
    weights, details = prepare_weights(parts['fit'], args.prepared, load_input, scheme, recipe['seed'])
    pd.DataFrame(details).to_csv(directory / 'fit_weights.csv', index=False)
    model, hp = create_model(args, parts, variant, recipe)
    optimizer, scheduler = make_optimizer(model, recipe, len(parts['fit']))
    positive = sum(row['positives'] for row in parts['fit'])
    count = sum(row['response_tokens'] for row in parts['fit'])
    alpha = torch.tensor(min((count - positive) / positive, 8.), device=args.device)
    checkpoint = directory / 'epoch_state.pt'
    history = resume_epoch(checkpoint, model, optimizer, scheduler)
    for epoch in range(len(history), recipe['epochs']):
        stats = epoch_train(model, parts['fit'], args, recipe, weights, variant, epoch, optimizer, scheduler, alpha)
        selected = predict(model, parts['select'], args.prepared, variant)
        labels = np.concatenate([row['gold'] for row in selected])
        score = np.concatenate([row['score'] for row in selected])
        history.append(dict(epoch=epoch + 1, selection_ap=float(average_precision_score(labels, score)), **stats))
        save_epoch(checkpoint, model, optimizer, scheduler, history)
        write_json(directory / 'history.json', history)
    write_json(directory / 'history.json', history)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(dict(model_state=state, hp=hp, loss_scheme=scheme), directory / 'checkpoint.partial')
    (directory / 'checkpoint.partial').replace(directory / 'checkpoint.pt')
    return model.eval()


def sample_table(samples, threshold):
    return pd.concat([token_frame(sample, sample['score'], threshold) for sample in samples], ignore_index=True)


def sample_spans(samples):
    rows = []
    for sample in samples:
        for start, end in merge_spans(sample['spans']):
            rows.append(dict(id=sample['id'], source_id=sample['source_id'], start=start, end=end))
    return pd.DataFrame(rows, columns=['id', 'source_id', 'start', 'end'])


def evaluate_fit(model, args, parts, variant, pairs, directory):
    from .evaluate import analyze
    from .continuity_history import analyze_history

    calibration = predict(model, parts['calibration'], args.prepared, variant)
    recipe = read_json(directory / 'config.json')['recipe']
    threshold = calibrate(calibration, recipe['fpr'])
    write_json(directory / 'threshold.json', threshold)
    write_json(directory / 'training.json', recipe)
    calibration_table = sample_table(calibration, threshold['value'])
    (directory / 'calibration').mkdir(exist_ok=True)
    calibration_table.to_csv(directory / 'calibration/tokens.csv', index=False)

    samples = predict(model, parts['test'], args.prepared, variant)
    table = sample_table(samples, threshold['value'])
    spans = sample_spans(samples)
    analyze(table, spans, threshold['value'], pairs, directory / 'test', args.bootstrap)
    table.to_csv(directory / 'test/tokens.csv', index=False)
    spans.to_csv(directory / 'test/spans.csv', index=False)
    counts, _, growth, _ = save_analysis(table, spans, threshold['value'], pairs, directory / 'continuity', args.bootstrap)
    plots(counts, growth, directory / 'continuity')
    analyze_history(table, spans, threshold['value'], pairs, directory / 'history_controls', args.bootstrap,
                    calibration=calibration_table, fpr=recipe['fpr'])


def fit_rows(seed_dir, summary):
    scheme, variant = seed_dir.parent.name, seed_dir.parent.parent.name
    identity = dict(model=variant, scheme=scheme, seed=int(seed_dir.name.split('_')[-1]))
    rows = [dict(identity, role='all', **summary['overall'])]
    rows.extend(dict(identity, **row) for row in summary['error_roles'])
    path = seed_dir / 'continuity/conditional_ranking.csv'
    if path.is_file():
        for row in pd.read_csv(path).to_dict('records'):
            role = 'previous_error' if row['previous_gold'] == 1 else 'previous_normal'
            rows.append(dict(identity, role=role, **row))
    return rows


def seed_comparison(frame, reference):
    keys = ['model', 'seed', 'role']
    baseline = frame[frame.scheme == reference]
    if baseline.empty:
        return pd.DataFrame()
    joined = frame.merge(baseline, on=keys, suffixes=('', '_reference'), validate='many_to_one')
    for field in ('auroc', 'ap', 'recall', 'fpr'):
        joined[field + '_delta'] = joined[field] - joined[field + '_reference']
    joined['reference'] = reference
    return joined


def training_growth(output, bootstrap):
    """Compare actual paired growth changes, not two unrelated group averages."""
    from .evaluate import source_interval

    captures = []
    for path in sorted(output.glob('*/*/seed_*/continuity/paired_growth.csv')):
        if not (path.parents[1] / 'complete.json').exists() or path.stat().st_size < 2:
            continue
        frame = pd.read_csv(path, dtype={'id': str, 'source_id': str})
        frame['scheme'], frame['model'] = path.parents[2].name, path.parents[3].name
        frame['seed'] = int(path.parents[1].name.split('_')[-1])
        captures.append(frame)
    if not captures:
        return pd.DataFrame()
    frame = pd.concat(captures, ignore_index=True)
    keys = ['model', 'seed', 'id', 'source_id', 'error_start']
    rows = []
    for reference in ('token', 'random_onset_half'):
        baseline = frame[frame.scheme == reference]
        compared = frame.merge(baseline, on=keys, suffixes=('', '_reference'), validate='many_to_one')
        for (model, scheme, seed), group in compared.groupby(['model', 'scheme', 'seed']):
            for field in ('gap_growth', 'auc_growth'):
                changes = group.assign(change=group[field] - group[field + '_reference'])
                rows.append(dict(model=model, scheme=scheme, seed=int(seed), reference=reference,
                                 measure=field, pairs=len(group), **source_interval(changes, 'change', bootstrap)))
    return pd.DataFrame(rows)


def summarize_fits(output, bootstrap=200):
    rows = []
    for path in sorted(output.glob('*/*/seed_*/test/summary.json')):
        if (path.parents[1] / 'complete.json').is_file():
            rows.extend(fit_rows(path.parents[1], read_json(path)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'fits.csv', index=False)
    if frame.empty:
        return frame
    joined = seed_comparison(frame, 'token')
    # Retain historical column names in the original comparison file.
    joined.rename(columns={name: name.replace('_reference', '_token') for name in joined}).to_csv(
        output / 'paired_seed_deltas.csv', index=False)
    fields = ['auroc_delta', 'ap_delta', 'recall_delta', 'fpr_delta']
    across = joined.groupby(['model', 'scheme', 'role'])[fields].agg(['mean', 'std', 'count'])
    across.columns = ['_'.join(column) for column in across.columns]
    across.reset_index().to_csv(output / 'across_seed_summary.csv', index=False)
    compared = seed_comparison(frame, 'random_onset_half')
    if not compared.empty:
        compared[compared.scheme == 'onset_half'].to_csv(output / 'onset_vs_random.csv', index=False)
    training_growth(output, bootstrap).to_csv(output / 'growth_deltas.csv', index=False)
    return frame


def run_training(args, output, pairs):
    from .data import prepare_output
    models = [name for name in args.models if name in ('node_only', 'charm_in')]
    if not models:
        raise ValueError('Use --models node_only and/or charm_in')
    if args.epochs is not None and args.epochs < 1:
        raise ValueError('Epoch count must be positive')
    if 'token' not in args.continuity_schemes:
        raise ValueError('The new token-weighted reference fit is required')
    print('Training models:', models, 'Fixed final epoch; no early stopping.', flush=True)
    for variant in models:
        original = read_json(Path(args.root) / variant / 'training.json')
        parts = original_parts(args.prepared, original)
        for seed in args.continuity_seeds:
            recipe = dict(original, seed=seed, epochs=args.epochs or original['epochs'], selection='fixed_final_epoch')
            for scheme in args.continuity_schemes:
                directory = output / variant / scheme / f'seed_{seed}'
                prepare_output(directory, dict(recipe=recipe, scheme=scheme, variant=variant,
                                               prepared=str(Path(args.prepared).resolve())))
                if (directory / 'complete.json').exists():
                    continue
                model = fit_model(args, parts, variant, scheme, recipe, directory)
                evaluate_fit(model, args, parts, variant, pairs, directory)
                write_json(directory / 'complete.json', dict(complete=True, epochs=recipe['epochs']))
                del model
    summarize_fits(output, args.bootstrap)
    write_json(output / 'protocol.json', dict(stage='supervised_training_loss_intervention_v2', models=models,
        schemes=args.continuity_schemes, seeds=args.continuity_seeds, selection='fixed_final_epoch',
        diagnostic_role_order=ROLE_NAMES, resume='last complete epoch with optimizer/scheduler/RNG',
        primary=['first-error ranking', 'continuation ranking', 'normal exits', 'paired front/back growth'],
        limitations=['Weighting changes the empirical objective, not all input-history dependence.',
                     'Random-onset weights isolate position choice from concentrating weight.',
                     'No text, native LLM states, labels, graph edges, or splits are edited.',
                     'Original early-stopped checkpoints are not the matched training baseline.']))
