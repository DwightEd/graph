"""Matched optimization budgets for loss-weight experiments on original inputs.

Default experiment: node_only. New fits never overwrite prior checkpoints.
Final fixed-epoch checkpoints are primary; no pooled-AP early stopping.
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


def load_input(record, prepared, variant):
    """Node path reads no edge members. Metadata-only path reads no model inputs."""
    path = Path(prepared)/'graphs'/record['split']/(str(record['id'])+'.npz')
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


def epoch_train(model, records, args, recipe, weights, variant, epoch, optimizer, scheduler, positive_weight):
    order = np.random.default_rng(recipe['seed']+epoch).permutation(len(records))
    loss_total = 0.
    role_gradient = np.zeros(3)
    role_loss = np.zeros(3)
    model.train()
    for start in tqdm(range(0, len(records), recipe['batch_size']), desc=f'{variant} epoch {epoch+1}', unit='batch'):
        batch = [records[index] for index in order[start:start+recipe['batch_size']]]
        denominator = sum(row['response_tokens'] for row in batch)
        optimizer.zero_grad(set_to_none=True)
        for record in batch:
            inputs, sample = load_input(record, args.prepared, variant)
            logits = score_record(model, inputs, variant)
            labels = torch.as_tensor(sample['gold'], dtype=logits.dtype, device=logits.device)
            weight = torch.as_tensor(weights[str(record['id'])], device=logits.device)
            raw = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight, reduction='none')
            loss = (raw * weight).sum()/denominator
            loss.backward()
            loss_total += float(loss.detach())
            measure_gradients(logits, labels, weight, positive_weight, sample['onset'], raw, role_gradient, role_loss)
        optimizer.step()
        scheduler.step()
    return dict(loss=loss_total, logit_abs_gradient=role_gradient.tolist(), weighted_loss=role_loss.tolist())


def measure_gradients(logits, labels, weights, alpha, onset, raw, gradient_sum, loss_sum):
    """Exact |d(unreduced weighted BCE)/dz|; NOT a parameter-gradient norm."""
    with torch.no_grad():
        probability = logits.sigmoid()
        derivative = weights * ((1-labels)*probability + alpha*labels*(probability-1))
        masks = [labels == 0, torch.as_tensor(onset, device=labels.device), (labels == 1) & ~torch.as_tensor(onset, device=labels.device)]
        for number, mask in enumerate(masks):
            gradient_sum[number] += float(derivative[mask].abs().sum())
            loss_sum[number] += float((raw*weights)[mask].sum())


def fit_model(args, parts, variant, scheme, recipe, directory):
    weights, details = prepare_weights(parts['fit'], args.prepared, load_input, scheme, recipe['seed'])
    pd.DataFrame(details).to_csv(directory/'fit_weights.csv', index=False)
    torch.manual_seed(recipe['seed'])
    inputs, _ = load_input(parts['fit'][0], args.prepared, variant)
    width = inputs.shape[1] if variant == 'node_only' else inputs['x'].shape[1]
    hp = dict(hidden_dim=recipe['hidden_dim'], gnn_layers=recipe['gnn_layers'], residual_mp=True)
    model = CHARM(width, width, hp, 'in', args.edge_chunk).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=.001)
    total_steps = recipe['epochs'] * math.ceil(len(parts['fit'])/recipe['batch_size'])
    warmup = max(1, int(.05*total_steps))
    def schedule(step):
        if step < warmup:
            return step/warmup
        return .5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    positive = sum(row['positives'] for row in parts['fit'])
    count = sum(row['response_tokens'] for row in parts['fit'])
    alpha = torch.tensor(min((count-positive)/positive, 8.), device=args.device)
    history = []
    for epoch in range(recipe['epochs']):
        stats = epoch_train(model, parts['fit'], args, recipe, weights, variant, epoch, optimizer, scheduler, alpha)
        selected = predict(model, parts['select'], args.prepared, variant)
        labels = np.concatenate([row['gold'] for row in selected])
        score = np.concatenate([row['score'] for row in selected])
        history.append(dict(epoch=epoch+1, selection_ap=float(average_precision_score(labels, score)), **stats))
        write_json(directory/'history.json', history)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(dict(model_state=state, hp=hp, loss_scheme=scheme), directory/'checkpoint.partial')
    (directory/'checkpoint.partial').replace(directory/'checkpoint.pt')
    return model.eval()


def evaluate_fit(model, args, parts, variant, pairs, directory):
    calibration = predict(model, parts['calibration'], args.prepared, variant)
    recipe = read_json(directory/'config.json')['recipe']
    threshold = calibrate(calibration, recipe['fpr'])
    write_json(directory/'threshold.json', threshold)
    samples = predict(model, parts['test'], args.prepared, variant)
    table = pd.concat([token_frame(sample, sample['score'], threshold['value']) for sample in samples], ignore_index=True)
    annotations = []
    for sample in samples:
        for start, end in merge_spans(sample['spans']):
            annotations.append(dict(id=sample['id'], source_id=sample['source_id'], start=start, end=end))
    spans = pd.DataFrame(annotations)
    from .evaluate import analyze
    analyze(table, spans, threshold['value'], pairs, directory/'test', args.bootstrap)
    table.to_csv(directory/'test/tokens.csv', index=False)
    spans.to_csv(directory/'test/spans.csv', index=False)
    counts, _, growth, _ = save_analysis(table, spans, threshold['value'], pairs, directory/'continuity', args.bootstrap)
    plots(counts, growth, directory/'continuity')


def summarize_fits(output):
    rows = []
    for path in sorted(output.glob('*/*/seed_*/test/summary.json')):
        summary = read_json(path)
        seed_dir = path.parent.parent
        scheme, variant = seed_dir.parent.name, seed_dir.parent.parent.name
        rows.append(dict(model=variant, scheme=scheme, seed=int(seed_dir.name.split('_')[-1]), role='all', **summary['overall']))
        for row in summary['error_roles']:
            rows.append(dict(model=variant, scheme=scheme, seed=int(seed_dir.name.split('_')[-1]), **row))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'fits.csv', index=False)
    keys = ['model', 'seed', 'role']
    reference = frame[frame.scheme == 'token']
    joined = frame.merge(reference, on=keys, suffixes=('', '_token'), validate='many_to_one')
    for field in ('auroc', 'ap', 'recall', 'fpr'):
        joined[field+'_delta'] = joined[field] - joined[field+'_token']
    joined.to_csv(output/'paired_seed_deltas.csv', index=False)
    fields = ['auroc_delta', 'ap_delta', 'recall_delta', 'fpr_delta']
    across = joined.groupby(['model', 'scheme', 'role'])[fields].agg(['mean', 'std', 'count'])
    across.columns = ['_'.join(column) for column in across.columns]
    across.reset_index().to_csv(output/'across_seed_summary.csv', index=False)
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
        original = read_json(Path(args.root)/variant/'training.json')
        parts = original_parts(args.prepared, original)
        for seed in args.continuity_seeds:
            recipe = dict(original, seed=seed, epochs=args.epochs or original['epochs'], selection='fixed_final_epoch')
            for scheme in args.continuity_schemes:
                directory = output/variant/scheme/f'seed_{seed}'
                prepare_output(directory, dict(recipe=recipe, scheme=scheme, variant=variant, prepared=str(Path(args.prepared).resolve())))
                if (directory/'complete.json').exists():
                    continue
                model = fit_model(args, parts, variant, scheme, recipe, directory)
                evaluate_fit(model, args, parts, variant, pairs, directory)
                write_json(directory/'complete.json', dict(complete=True, epochs=recipe['epochs']))
                del model
    summarize_fits(output)
    write_json(output/'protocol.json', dict(stage='supervised_training_loss_intervention', models=models,
        schemes=args.continuity_schemes, seeds=args.continuity_seeds, selection='fixed_final_epoch',
        primary=['first-error AUROC', 'onset AUROC', 'continuation AUROC', 'calibrated recall/FPR', 'matched position gaps'],
        limitations=['Weighting changes the empirical target distribution, not solely dependence.',
                     'Random-onset weights isolate position choice from concentrating weight.',
                     'No text, native LLM states, label identities, graph edges, or splits are edited.',
                     'Interrupted fits restart; only completed fits are reused.',
                     'Original previously selected checkpoints are not paired causal baselines; the new token fit is.']))
