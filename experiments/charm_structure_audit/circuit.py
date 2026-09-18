"""Read the exported real checkpoint, decode its direction, and test it directly."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .model import CHARM
from .circuit_math import axis_experiments, paired_accounting, project, tail, local_gradients


def read_inputs(path):
    """Named archive members only; do not execute pickle or extract filesystem paths."""
    records, errors, normals = [], [], []
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        geometry = json.load(archive.extractfile('geometry.json'))
        checkpoint = io.BytesIO(archive.extractfile('node_only/checkpoint.pt').read())
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for pair in manifest['pairs']:
            content = io.BytesIO(archive.extractfile('inputs/' + pair['file']).read())
            with np.load(content, allow_pickle=False) as arrays:
                errors.append(arrays['error_x'])
                normals.append(arrays['normal_x'])
                for offset in range(pair['length']):
                    records.append(dict(id=pair['id'], source_id=pair['source_id'], start=pair['error_start'],
                        normal_start=pair['normal_start'], offset=offset, length=pair['length'],
                        error_text=str(arrays['text'][0, offset]), normal_text=str(arrays['text'][1, offset]),
                        error_score=float(arrays['baseline_score'][0, offset]),
                        normal_score=float(arrays['baseline_score'][1, offset]), threshold=float(arrays['threshold'])))
    width = geometry['layers'] * geometry['heads']
    model = CHARM(width, width, saved['hp']).double().eval()
    model.load_state_dict(saved['model_state'])
    model.requires_grad_(False)
    return model, torch.from_numpy(np.concatenate(errors)), torch.from_numpy(np.concatenate(normals)), pd.DataFrame(records), geometry


def token_weights(table):
    """Equal source, then equal pair, then equal token. Never bootstrap heads."""
    weights = np.zeros(len(table))
    for _, source in table.groupby('source_id'):
        pairs = source.groupby(['id', 'start'])
        for _, group in pairs:
            weights[group.index] = 1 / (table.source_id.nunique() * len(pairs) * len(group))
    return weights


def pair_metrics(scores, table):
    rows = []
    for name, values in scores.items():
        for (identity, start), indices in table.groupby(['id', 'start']).indices.items():
            group = table.iloc[indices]
            labels = np.r_[np.ones(len(indices)), np.zeros(len(indices))]
            rows.append(dict(variant=name, id=identity, start=start, source_id=group.source_id.iloc[0],
                auroc=roc_auc_score(labels, values[:, indices].reshape(-1)),
                logit_gap=(values[0, indices]-values[1, indices]).mean()))
    return pd.DataFrame(rows)


def summarize_pairs(frame, bootstrap):
    rows = []
    base = frame[frame.variant == 'full'].set_index(['id', 'start'])
    for name, group in frame.groupby('variant', sort=False):
        group = group.set_index(['id', 'start']).copy()
        group['auc_change'] = group.auroc - base.auroc
        group['gap_change'] = group.logit_gap - base.logit_gap
        source = group.groupby('source_id')[['auroc', 'auc_change', 'logit_gap', 'gap_change']].mean()
        row = dict(variant=name, pairs=len(group), sources=len(source), pair_macro_auc=group.auroc.mean())
        generator = np.random.default_rng(42)
        samples = generator.integers(len(source), size=(bootstrap, len(source)))
        for field in source:
            values = source[field].to_numpy()
            row[field] = values.mean()
            if bootstrap and len(source) > 1:
                row[field+'_low'], row[field+'_high'] = np.quantile(values[samples].mean(1), [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)


def save_rules(model, error, normal, accounting, direction, table, geometry, output):
    weights = token_weights(table)
    mean = lambda values: weights @ values.detach().cpu().numpy()
    width = len(direction)
    heads = geometry['heads']
    coefficients = model.in_proj.weight
    pd.DataFrame(coefficients.cpu().numpy()).to_csv(output/'projection_weights.csv.gz', index=False)
    pd.DataFrame(dict(channel=np.arange(width), layer=np.arange(width)//heads, head=np.arange(width)%heads,
        axis=direction.cpu().numpy(), error_mean=mean(error), normal_mean=mean(normal),
        contribution=mean(accounting['head_contribution']), multiplier=mean(accounting['head_multiplier']))).to_csv(output/'head_rules.csv', index=False)
    error_state, normal_state = project(model, error), project(model, normal)
    alignment = (coefficients @ direction) / coefficients.norm(dim=1)
    pd.DataFrame(dict(unit=np.arange(len(coefficients)), bias=model.in_proj.bias.cpu().numpy(),
        direction_alignment=alignment.cpu().numpy(), contribution=mean(accounting['unit_contribution']),
        error_activation=mean(error_state), normal_activation=mean(normal_state),
        error_on=mean((error_state>0).double()), normal_on=mean((normal_state>0).double()),
        downstream=mean(accounting['downstream_multiplier']))).to_csv(output/'projection_units.csv', index=False)
    for name, values in accounting.items():
        np.savez_compressed(output/(name+'.npz'), values=values.cpu().numpy())


def swap_all_units(model, error, normal, weights):
    """All input-projection units, no test-selected shortlist."""
    error_state, normal_state = project(model, error), project(model, normal)
    error_logit, normal_logit = tail(model, error_state), tail(model, normal_state)
    rows = []
    for unit in range(error_state.shape[1]):
        removed, added = error_state.clone(), normal_state.clone()
        removed[:, unit], added[:, unit] = normal_state[:, unit], error_state[:, unit]
        rows.append(dict(unit=unit, removal_effect=weights @ (error_logit-tail(model, removed)).cpu().numpy(),
                         insertion_effect=weights @ (tail(model, added)-normal_logit).cpu().numpy()))
    return pd.DataFrame(rows)


def position_summary(table):
    rows = []
    masks = dict(all=np.ones(len(table), bool), front=(table.offset+.5)/table.length < .5,
                 back=(table.offset+.5)/table.length >= .5)
    for region, mask in masks.items():
        group = table[mask].reset_index(drop=True)
        weights = token_weights(group)
        for side in ('error', 'normal'):
            rows.append(dict(region=region, side=side, tokens=len(group),
                score=weights @ group[side+'_score'], axis=weights @ group[side+'_axis'],
                alarm_rate=float((group[side+'_score']>group.threshold).mean())))
    return pd.DataFrame(rows)


def plot_results(summary, rules, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['full', 'keep_axis', 'remove_axis', 'replace_error_axis', 'insert_axis_into_normal']
    selected = summary.set_index('variant').loc[names]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(np.arange(len(names)), selected.auroc)
    axis.set_xticks(np.arange(len(names)), names, rotation=15)
    axis.set_ylim(0, 1)
    axis.set_ylabel('Within-pair AUROC, equal-source mean')
    for i, value in enumerate(selected.auroc):
        axis.text(i, value+.02, f'{value:.4f}', ha='center')
    figure.tight_layout()
    figure.savefig(output/'axis_controls.png', dpi=160)
    plt.close(figure)
    matrix = rules.pivot(index='layer', columns='head', values='axis')
    figure, axis = plt.subplots(figsize=(8, 5))
    scale = abs(matrix.to_numpy()).max()
    image = axis.imshow(matrix, aspect='auto', vmin=-scale, vmax=scale)
    axis.set(xlabel='Original LLM head', ylabel='Original LLM layer', title='Direction extracted from trained projection weights')
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output/'weight_direction.png', dpi=160)
    plt.close(figure)


def run_circuit(input_path, output, bootstrap=2000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, error, normal, table, geometry = read_inputs(input_path)
    with torch.no_grad():
        scores, direction, loading, singular = axis_experiments(model, error, normal)
        accounting = paired_accounting(model, error, normal)
    full = scores['full']
    saved = table[['error_score', 'normal_score']].to_numpy().T
    np.testing.assert_allclose(full.sigmoid().cpu().numpy(), saved, atol=2e-5, rtol=0)
    gap = full[0] - full[1]
    for name in ('head_contribution', 'unit_contribution'):
        torch.testing.assert_close(accounting[name].sum(1), gap, atol=1e-8, rtol=1e-7)
    values = torch.cat((error, normal))
    gradients = local_gradients(model, values)
    cosines = (gradients @ direction) / gradients.norm(dim=1).clamp_min(1e-30)
    table['error_axis'], table['normal_axis'] = (error@direction).numpy(), (normal@direction).numpy()
    with torch.no_grad():
        save_rules(model, error, normal, accounting, direction, table, geometry, output)
        swap_all_units(model, error, normal, token_weights(table)).to_csv(output/'unit_swaps.csv', index=False)
    frame = pair_metrics({name: value.cpu().numpy() for name, value in scores.items()}, table)
    summary = summarize_pairs(frame, bootstrap)
    frame.to_csv(output/'pair_metrics.csv', index=False)
    summary.to_csv(output/'axis_summary.csv', index=False)
    position_summary(table).to_csv(output/'positions.csv', index=False)
    table.to_csv(output/'tokens.csv.gz', index=False)
    status = dict(pairs=table[['id','start']].drop_duplicates().shape[0], sources=table.source_id.nunique(),
        paired_positions=len(table), replay_error=float(abs(full.sigmoid().numpy()-saved).max()),
        first_weight_energy=float(singular[0].square()/singular.square().sum()),
        gradient_axis_cosine_median=float(cosines.median()),
        axis_logit_spearman=float(spearmanr((values@direction).numpy(), full.numpy().reshape(-1)).statistic),
        rank1_logit_mae=float((scores['keep_axis']-full).abs().mean()),
        mean_rank1_logit_mae=float((scores['keep_axis_mean_background']-full).abs().mean()),
        accounting='Rescale finite-difference accounting, not numerical integrated gradients',
        scope='One supervised node checkpoint, fixed labeled pairs; no LLM or graph-message execution')
    (output/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    plot_results(summary, pd.read_csv(output/'head_rules.csv'), output)
    print(summary[['variant','auroc','auc_change','logit_gap']].to_string(index=False), flush=True)
    print('Circuit results:', output, flush=True)
