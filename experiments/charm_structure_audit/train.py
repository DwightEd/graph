"""Independent supervised module ablations using the ORIGINAL saved split."""

import math
from pathlib import Path
import zlib

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from .ablations import change_graph
from .data import load_graph, write_json
from .model import CHARM


def graph_for(record, prepared, name, seed):
    graph, sample = load_graph(record, prepared)
    seed = zlib.crc32(str(record['id']).encode()) + seed
    return change_graph(graph, name, seed), sample


def predict(model, records, prepared, name, seed):
    samples = []
    model.eval()
    with torch.no_grad():
        for record in records:
            graph, sample = graph_for(record, prepared, name, seed)
            logits = model(graph, ablation=name)[int(graph['prompt_length']):]
            samples.append(dict(record, **sample, score=torch.sigmoid(logits).cpu().numpy()))
    return samples


def calibrate(samples, fpr):
    negative = np.concatenate([s['score'][(~s['gold'].astype(bool)) & (s['offsets'][:, 1]>s['offsets'][:, 0])] for s in samples])
    if not len(negative):
        raise ValueError('Calibration has no normal text tokens')
    threshold = float(np.quantile(negative, 1-fpr, method='higher'))
    return dict(value=threshold, target_fpr=fpr, achieved_fpr=float((negative>threshold).mean()),
                negatives=len(negative), rule='score > threshold', origin='original_source_disjoint_calibration')


def train_epoch(model, rows, prepared, name, recipe, epoch, optimizer, scheduler, weight):
    order = np.random.default_rng(recipe['seed']+epoch).permutation(len(rows))
    batches = range(0, len(rows), recipe['batch_size'])
    losses = []
    model.train()
    for start in tqdm(batches, desc=f'{name} epoch {epoch+1}', unit='batch'):
        batch = [rows[i] for i in order[start:start+recipe['batch_size']]]
        tokens = sum(r['response_tokens'] for r in batch)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for record in batch:
            graph, sample = graph_for(record, prepared, name, recipe['seed'])
            logits = model(graph, ablation=name)[int(graph['prompt_length']):]
            labels = torch.as_tensor(sample['gold'], device=logits.device, dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction='sum')/tokens
            loss.backward()
            loss_value += float(loss.detach())
        optimizer.step()
        scheduler.step()
        losses.append(loss_value)
    return float(np.mean(losses))


def fit(parts, prepared, name, recipe, output, device='cpu', chunk=4096):
    """Completed fits may be reused; interrupted fitting restarts, not hidden resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'checkpoint.pt'
    if (output/'fit_complete.json').exists():
        return path
    torch.manual_seed(recipe['seed'])
    graph, _ = load_graph(parts['fit'][0], prepared)
    hp = dict(hidden_dim=recipe['hidden_dim'], gnn_layers=recipe['gnn_layers'], residual_mp=True)
    if name == 'one_layer':
        hp['gnn_layers'] = 1
    model = CHARM(graph['x'].shape[1], graph['edge_attr'].shape[1], hp, 'in', chunk).to(device)
    positive = sum(r['positives'] for r in parts['fit'])
    total = sum(r['response_tokens'] for r in parts['fit'])
    if not 0 < positive < total:
        raise ValueError('Fit split must include normal and error tokens')
    weight = torch.tensor(min((total-positive)/positive, 8.), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=.001)
    steps = recipe['epochs']*math.ceil(len(parts['fit'])/recipe['batch_size'])
    warmup = max(1, int(.05*steps))
    def schedule(step):
        return step/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_select(model, parts, prepared, name, recipe, output, optimizer, scheduler, weight, hp)
    return path


def train_select(model, parts, prepared, name, recipe, output, optimizer, scheduler, weight, hp):
    best, stale, history = -1., 0, []
    for epoch in range(recipe['epochs']):
        loss = train_epoch(model, parts['fit'], prepared, name, recipe, epoch, optimizer, scheduler, weight)
        selected = predict(model, parts['select'], prepared, name, recipe['seed'])
        labels = np.concatenate([s['gold'] for s in selected])
        if not labels.any():
            raise ValueError('Selection split has no error tokens')
        ap = float(average_precision_score(labels, np.concatenate([s['score'] for s in selected])))
        history.append(dict(epoch=epoch+1, loss=loss, selection_ap=ap))
        print(history[-1], flush=True)
        if ap > best:
            best, stale = ap, 0
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            temporary = output/'checkpoint.partial'
            torch.save(dict(model_state=state, hp=hp, ablation=name), temporary)
            temporary.replace(output/'checkpoint.pt')
        else:
            stale += 1
        write_json(output/'history.json', history)
        if stale >= recipe['patience']:
            break
    write_json(output/'fit_complete.json', dict(best_selection_ap=best))
