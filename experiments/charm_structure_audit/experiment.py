"""Train controlled CHARM replicas, or inspect an existing original checkpoint."""

import json
import math
from pathlib import Path
import zlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from tqdm import tqdm

from .data import load_graph, save_npz, write_json
from .graph import PERTURBATIONS, degree, node_statistics, prefix_graph, transform_graph
from .metrics import paired_delta, report, threshold_at_fpr
from .model import CHARM, load_checkpoint, smooth_scores


def stable_seed(identity, seed=0):
    return zlib.crc32(str(identity).encode()) + seed


def normalization(variant):
    return 'out' if variant == 'charm_out' else 'in'


def check_settings(path, config):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('settings changed; choose a new output: ' + str(path))
    write_json(path, config)


def inference(model, graph):
    model.eval()
    with torch.no_grad():
        logits, hidden = model(graph, return_hidden=True)
    p = int(graph['prompt_length'])
    return torch.sigmoid(logits[p:]).cpu().numpy(), hidden.cpu().numpy()


def collect(model, records, variant, seed):
    samples = []
    for row in records:
        graph = load_graph(row['graph'])
        view, _ = transform_graph(graph, variant, stable_seed(row['id'], seed))
        score, _ = inference(model, view)
        samples.append(dict(row, gold=graph['gold'].astype(bool), score=score, offsets=graph['offsets']))
    return samples


def fit(parts, output, variant, seed=0, epochs=50, patience=5, hidden_dim=128,
        layers=3, batch_size=32, learning_rate=5e-4, device='cpu', edge_chunk=4096, fpr=.05):
    """Supervised BCE; graph-wise accumulation equals a token-weighted batch.

    No BatchNorm exists in the active upstream MLP. Select uses select labels;
    thresholds use separate calibration negatives; test is not read here.
    A finished checkpoint is reused; an interrupted fit restarts training.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = dict(variant=variant, seed=seed, epochs=epochs, patience=patience,
                  hidden_dim=hidden_dim, gnn_layers=layers, batch_size=batch_size,
                  learning_rate=learning_rate, fpr=fpr,
                  partitions={k: [r['id'] for r in v] for k, v in parts.items()})
    check_settings(output / 'training.json', config)
    checkpoint = output / 'checkpoint.pt'
    if (output / 'fit_complete.json').exists():
        return checkpoint
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    first = load_graph(parts['fit'][0]['graph'])
    hp = dict(hidden_dim=hidden_dim, gnn_layers=layers, residual_mp=True)
    model = CHARM(first['x'].shape[1], first['edge_attr'].shape[1], hp, normalization(variant), edge_chunk).to(device)
    del first
    positive = sum(r['positives'] for r in parts['fit'])
    total = sum(r['response_tokens'] for r in parts['fit'])
    if positive == 0 or positive == total:
        raise ValueError('fit partition needs both classes; increase pilot sample count')
    weight = torch.tensor(min((total - positive) / positive, 8.), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.001)
    steps = epochs * math.ceil(len(parts['fit']) / batch_size)
    warmup = max(1, int(.05 * steps))
    def schedule(step):
        return step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history, best_ap, stale = [], -1., 0
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(parts['fit']))
        losses = []
        for start in tqdm(range(0, len(order), batch_size), desc=f'{variant} seed={seed} epoch={epoch+1}', unit='batch'):
            rows = [parts['fit'][i] for i in order[start:start+batch_size]]
            denominator = sum(r['response_tokens'] for r in rows)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for row in rows:
                graph = load_graph(row['graph'])
                view, _ = transform_graph(graph, variant, stable_seed(row['id'], seed))
                logits = model(view)[int(graph['prompt_length']):]
                labels = torch.as_tensor(graph['gold'], dtype=logits.dtype, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction='sum') / denominator
                loss.backward(); batch_loss += float(loss.detach())
                del graph, view, logits, loss
            optimizer.step(); scheduler.step(); losses.append(batch_loss)
        selected = collect(model, parts['select'], variant, seed)
        y = np.concatenate([s['gold'] for s in selected]); score = np.concatenate([s['score'] for s in selected])
        if not y.any():
            raise ValueError('select partition has no positive tokens; increase pilot sample count')
        ap = float(average_precision_score(y, score))
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), selection_ap=ap)
        history.append(row); print(json.dumps(dict(variant=variant, seed=seed, **row)), flush=True)
        if ap > best_ap:
            best_ap, stale = ap, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(dict(model_state=state, hp=hp, variant=variant, normalization=normalization(variant)), output / 'best.partial.pt')
            (output / 'best.partial.pt').replace(checkpoint)
        else:
            stale += 1
        write_json(output / 'history.json', history)
        if stale >= patience:
            break
    model, _ = load_checkpoint(checkpoint, device, normalization(variant), edge_chunk)
    calibration = collect(model, parts['calibration'], variant, seed)
    threshold = threshold_at_fpr(calibration, fpr)
    if variant == 'node_only':
        smoothed = [dict(s, score=smooth_scores(s['score'])) for s in calibration]
        threshold['ewma'] = threshold_at_fpr(smoothed, fpr)
    write_json(output / 'threshold.json', threshold)
    write_json(output / 'fit_complete.json', dict(complete=True, best_selection_ap=best_ap))
    return checkpoint


def representation_statistics(graph, hidden):
    """Post-hoc label homophily and cosine similarity; not a model input."""
    p = int(graph['prompt_length'])
    source, target = graph['edge_index']
    keep = source >= p
    pairs = dict(rr=np.stack((source[keep]-p, target[keep]-p)),
                 chain=np.stack((np.arange(len(graph['gold'])-1), np.arange(1, len(graph['gold'])))))
    y = graph['gold'].astype(bool)
    result = {}
    for name, edge in pairs.items():
        left, right = edge
        for feature, x in [('input', graph['x'][p:]), ('encoded', hidden[p:])]:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            cosine = np.sum(x[left] * x[right], axis=-1)
            for role, mask in [('HH', y[left] & y[right]), ('NN', ~y[left] & ~y[right]), ('mixed', y[left] != y[right])]:
                result[f'{name}_{feature}_{role}'] = dict(pairs=int(mask.sum()), mean_cosine=float(cosine[mask].mean()) if mask.any() else None)
        result[name + '_same_label_fraction'] = float((y[left] == y[right]).mean()) if len(left) else None
    return result


def audit(checkpoint, records, output, variant='charm_out', seed=0, device='cpu', edge_chunk=4096,
          threshold=None, perturbations=True, prefix_sites=8):
    """Frozen-model inference: no optimizer and no test-selected thresholds.

    Perturbations hold the ORIGINAL divisor fixed, to avoid mixing edge removal
    with re-normalization. Prefix recomputes its divisor to expose future use.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    checkpoint_stamp = [str(checkpoint), checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns]
    config = dict(checkpoint=checkpoint_stamp, records=[r['id'] for r in records], variant=variant,
                  seed=seed, perturbations=perturbations, prefix_sites=prefix_sites,
                  threshold=threshold, alignment='post_token_i_to_label_i')
    check_settings(output / 'prediction_settings.json', config)
    model, _ = load_checkpoint(checkpoint, device, normalization(variant), edge_chunk)
    paths = []
    for row in tqdm(records, desc='audit ' + variant, unit='sample'):
        path = output / 'samples' / (row['id'] + '.npz')
        graph_stamp = [Path(row['graph']).stat().st_size, Path(row['graph']).stat().st_mtime_ns]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['graph_stamp'].tolist() != graph_stamp:
                    raise ValueError('prepared graph changed: ' + row['id'])
            paths.append(str(path)); continue
        graph = load_graph(row['graph'])
        view, changed = transform_graph(graph, variant, stable_seed(row['id'], seed))
        score, hidden = inference(model, view)
        arrays = dict(score=score, gold=graph['gold'], onset=graph['onset'], spans=graph['spans'],
                      offsets=graph['offsets'], response=graph['response'], embedding=hidden[int(graph['prompt_length']):],
                      graph_stamp=np.asarray(graph_stamp), record_json=np.asarray(json.dumps(row)))
        arrays.update({'structure_' + k: v for k, v in node_statistics(graph).items()})
        diagnostic = dict(training_graph_change=changed, representation=representation_statistics(view, hidden))
        if variant == 'node_only':
            arrays['score_ewma'] = smooth_scores(score)
        if perturbations:
            divisor = degree(view, model.normalization)
            for control in PERTURBATIONS[1:]:
                altered, change = transform_graph(view, control, stable_seed(row['id'], seed))
                with torch.no_grad():
                    values = torch.sigmoid(model(altered, divisor=divisor)[int(graph['prompt_length']):]).cpu().numpy()
                arrays['score_' + control] = values
                diagnostic[control] = change
        if prefix_sites:
            # Include annotated onsets for post-hoc first-error attribution only.
            chosen = np.unique(np.r_[np.linspace(0, len(score)-1, min(prefix_sites, len(score)), dtype=int), np.flatnonzero(graph['onset'])])
            prefix_score = np.full(len(score), np.nan)
            for t in chosen:
                end = int(graph['prompt_length']) + int(t) + 1
                cropped = prefix_graph(view, end)
                with torch.no_grad():
                    prefix_score[t] = float(torch.sigmoid(model(cropped)[-1]).cpu())
            arrays['score_prefix'] = prefix_score
            diagnostic['prefix_selection'] = 'uniform response positions plus annotated onsets, diagnostic only'
        arrays['diagnostic_json'] = np.asarray(json.dumps(diagnostic))
        save_npz(path, **arrays)
        paths.append(str(path))
    write_json(output / 'predictions.json', paths)
    return load_predictions(output)


def load_predictions(output, completed_only=False):
    root = Path(output)
    files = sorted((root / 'samples').glob('*.npz')) if completed_only else json.loads((root / 'predictions.json').read_text())
    if not files:
        raise ValueError('no finalized CHARM prediction samples found')
    samples = []
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files if k != 'embedding'}
        row = json.loads(str(arrays.pop('record_json')))
        arrays['diagnostic'] = json.loads(str(arrays.pop('diagnostic_json')))
        arrays['response'] = str(arrays['response'])
        samples.append(dict(row, **arrays))
    return samples


def compare_models(prediction_dirs, output, bootstrap=200):
    """Paired held-out comparison; independently fitted variants, identical IDs."""
    samples = {name: load_predictions(path) for name, path in prediction_dirs.items()}
    if 'charm_in' not in samples:
        return {}
    base = samples['charm_in']
    result = {}
    for name, other in samples.items():
        mapping = {s['id']: s for s in other}
        if set(mapping) != {s['id'] for s in base}:
            raise ValueError('cannot compare models on different response sets')
        paired = []
        for s in base:
            other_sample = mapping[s['id']]
            if not np.array_equal(s['gold'], other_sample['gold']):
                raise ValueError('label/token mismatch across model outputs')
            paired.append(dict(s, alternative=other_sample['score']))
        result[name] = paired_delta(paired, 'alternative', bootstrap)
    write_json(output, dict(reference='charm_in', experiment='independently_retrained_ablation', models=result))
    return result
