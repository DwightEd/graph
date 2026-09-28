"""Source-cross-fitted ordinary probes; case labels never enter fitting."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from experiments.role_free_flow.risk_response import CandidateRiskProbe, rms_scale
from .data import labels, read_json, write_json


class CandidateMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden_size * 4 + 8, 9), nn.GELU(), nn.Linear(9, 1))

    def forward(self, states, embedding, confidence):
        values = torch.cat((rms_scale(states).flatten(1), rms_scale(embedding), confidence), -1)
        return self.network(values).squeeze(-1)


def load_inputs(output, records):
    saved, regions, start = [], {}, 0
    for record in records:
        with np.load(output / 'static' / f"{record['key']}.npz") as data:
            item = {key: data[key] for key in ('states', 'embedding', 'confidence', 'valid')}
        regions[record['key']] = slice(start, start + len(item['valid']))
        start += len(item['valid'])
        saved.append(item)
    arrays = {key: np.concatenate([item[key] for item in saved]) for key in saved[0]}
    tensors = {key: torch.from_numpy(arrays[key]).float() for key in ('states', 'embedding', 'confidence')}
    return tensors, arrays['valid'], regions


def source_weights(records, regions, valid, size):
    weights = np.zeros(size)
    for record in records:
        region = regions[record['key']]
        weights[region] = valid[region] / max(int(valid[region].sum()), 1)
    return weights


def forward(model, kind, inputs, indices):
    args = [inputs['states'][indices], inputs['embedding'][indices]]
    if kind == 'mlp9':
        args.append(inputs['confidence'][indices])
    return model(*args)


@torch.no_grad()
def predict(model, kind, inputs):
    return torch.cat([forward(model, kind, inputs, slice(start, start + 512))
                      for start in range(0, len(inputs['states']), 512)])


def fit_fold(kind, fold, inputs, target, training_weights, development_weights):
    torch.manual_seed(4200 + fold)
    hidden = inputs['embedding'].shape[-1]
    model = CandidateRiskProbe(hidden) if kind == 'rank8' else CandidateMLP(hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    training = torch.nonzero(training_weights > 0, as_tuple=True)[0]
    scale = training_weights[training].mean()
    best, best_state, epoch_best = float('inf'), None, 0
    for epoch in range(1, 31):
        order = training[torch.randperm(len(training))]
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                forward(model, kind, inputs, indices), target[indices], reduction='none')
            objective = (loss * training_weights[indices] / scale).mean()
            objective.backward()
            optimizer.step()
        if epoch % 5 == 0:
            logits = predict(model, kind, inputs)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
            development = float((loss * development_weights).sum() / development_weights.sum())
            if development < best:
                best, best_state, epoch_best = development, deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, predict(model, kind, inputs).numpy(), dict(epoch=epoch_best, dev_bce=best,
        parameters=sum(parameter.numel() for parameter in model.parameters()))


def fit(args):
    torch.set_num_threads(4)
    records = read_json(args.output / 'manifest.json')['records']
    supervised = [r for r in records if r['role'] in ('fit', 'dev')]
    truth = labels(supervised)
    inputs, valid, regions = load_inputs(args.output, records)
    size = len(valid)
    target = torch.zeros(size)
    for record in supervised:
        target[regions[record['key']]] = torch.tensor(truth[record['key']]).float()
    development = [r for r in records if r['role'] == 'dev']
    dev_weights = torch.tensor(source_weights(development, regions, valid, size)).float()
    fitting = [r for r in records if r['role'] == 'fit']
    predictions, training = {}, {}
    for kind in ('rank8', 'mlp9'):
        predictions[kind] = []
        for fold in range(5):
            fold_rows = [r for r in fitting if r['fold'] != fold]
            weights = torch.tensor(source_weights(fold_rows, regions, valid, size)).float()
            mean = inputs['confidence'][weights > 0].mean(0)
            scale = inputs['confidence'][weights > 0].std(0).clamp_min(.01)
            fold_inputs = dict(inputs, confidence=(inputs['confidence'] - mean) / scale)
            model, scores, details = fit_fold(kind, fold, fold_inputs, target, weights, dev_weights)
            torch.save(model.state_dict(), args.output / 'models' / f'{kind}_{fold}.pt')
            np.savez(args.output / 'models' / f'{kind}_{fold}_confidence.npz',
                     mean=mean.numpy(), scale=scale.numpy())
            predictions[kind].append(scores)
            training[f'{kind}_{fold}'] = details
            print(dict(stage='probe', kind=kind, fold=fold, **details), flush=True)
        predictions[kind] = np.stack(predictions[kind])
    for record in records:
        region = regions[record['key']]
        np.savez_compressed(args.output / 'scores' / f"probe_{record['key']}.npz",
            **{kind: score[:, region] for kind, score in predictions.items()})
    write_json(args.output / 'models' / 'probe_training.json', dict(folds=5, details=training,
        labels_used='fit and dev only; source-disjoint regression',
        fitting_sources=[r['source_id'] for r in fitting],
        development_sources=[r['source_id'] for r in development]))
