"""Supervised-direction audit of complete local self-head matrices, not a detector."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.token_backtrace.repetition_analysis import pair_statistics
from experiments.token_backtrace.repetition_teacher import BASE, TEACHERS, SelfReadout


RANKS = (1, 2, 4, 8)
WINDOW = 8
OUTPUT = Path('outputs/source_transfer_matrix_20261008')


def local_projection(values, direction, ranks=RANKS, window=WINDOW):
    """Full 8x1024 matrices; project trajectories through temporal Gram eigenvectors.

    X_r w = U_r U_r.T X w, without constructing a truncated head matrix.
    Zero rows at answer start are padding, never observations from future tokens.
    """
    padded = np.pad(values.astype(np.float64), ((window - 1, 0), (0, 0)))
    matrices = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    matrices = matrices.transpose(0, 2, 1)
    gram = matrices @ matrices.transpose(0, 2, 1)
    eigenvalues, temporal_axes = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues[:, ::-1], 0)
    temporal_axes = temporal_axes[:, :, ::-1]
    trajectory = matrices @ direction
    coefficients = np.einsum('tij,ti->tj', temporal_axes, trajectory)
    variance = eigenvalues.sum(axis=1)
    energy = (trajectory ** 2).sum(axis=1)

    result = {}
    for rank in ranks:
        projected = np.einsum('tij,tj->ti', temporal_axes[:, :, :rank], coefficients[:, :rank])
        retained_energy = (coefficients[:, :rank] ** 2).sum(axis=1)
        result[rank] = dict(score=projected[:, -1],
                            variance=eigenvalues[:, :rank].sum(axis=1) / variance,
                            readout=retained_energy / np.maximum(energy, 1e-30))
    result['full_score'] = trajectory[:, -1]
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n')


def freeze_inputs(output):
    """Input/model identity is recorded before scoring; no parameters are fitted."""
    paths = [BASE / 'prepared.pt', Path(__file__),
             Path('experiments/token_backtrace/repetition_teacher.py'),
             Path('outputs/repetition_dominance_20261007/matched_tokens.json')]
    paths.extend(TEACHERS / f'self_only_seed{seed}/model.pt' for seed in (42, 123))
    manifest = dict(experiment='supervised_matrix_direction_audit',
                    supervision='frozen naturally supervised teacher tangents; diagnostic only',
                    window=WINDOW, ranks=list(RANKS), preprocessing='teacher fit-only standardization',
                    projection='uncentered local Gram; full head axes retained in prepared.pt',
                    score='last temporal row only; no span mean or broadcast',
                    fit_operations=0, llm_forwards=0,
                    inputs={str(path): sha256(path) for path in paths})
    write_json(output / 'INPUT_MANIFEST.json', manifest)


def measure_rows(rows):
    scores = {row['id']: {} for row in rows}
    retentions = {rank: dict(variance=[], readout=[]) for rank in RANKS}
    reconstruction_error = 0.
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    for row in rows:
        per_seed = []
        for teacher in teachers:
            values = teacher.standardized(row).numpy()
            measured = local_projection(values, teacher.tangent())
            reconstruction_error = max(reconstruction_error, float(np.max(
                np.abs(measured[WINDOW]['score'] - measured['full_score']))))
            per_seed.append(measured)
            for rank in RANKS:
                for name in ('variance', 'readout'):
                    retentions[rank][name].extend(measured[rank][name])

        scores[row['id']]['full'] = np.mean([seed['full_score'] for seed in per_seed], axis=0)
        for rank in RANKS:
            scores[row['id']][f'rank{rank}'] = np.mean(
                [seed[rank]['score'] for seed in per_seed], axis=0)
        counts = np.arange(1, len(row['node']) + 1)
        scores[row['id']]['complete_window'] = counts >= WINDOW
    return scores, retentions, reconstruction_error


def score_metrics(labels, values):
    return dict(auroc=float(roc_auc_score(labels, values)),
                ap=float(average_precision_score(labels, values)))


def source_difference(rows, scores, baseline, target, draws=1000):
    """Paired resampling of whole sources; compare frozen score rankings."""
    sources = sorted({row['source_id'] for row in rows})
    labels = {source: np.concatenate([row['labels'] for row in rows if row['source_id'] == source])
              for source in sources}
    values = {name: {source: np.concatenate([scores[row['id']][name] for row in rows
                                            if row['source_id'] == source])
                     for source in sources} for name in (baseline, target)}
    generator = np.random.default_rng(42)
    differences = []
    for _ in range(draws):
        selected = generator.choice(sources, len(sources), replace=True)
        truth = np.concatenate([labels[source] for source in selected])
        first = np.concatenate([values[baseline][source] for source in selected])
        second = np.concatenate([values[target][source] for source in selected])
        if len(np.unique(truth)) == 2:
            differences.append(roc_auc_score(truth, second) - roc_auc_score(truth, first))
    return dict(auroc_difference_ci95=np.quantile(differences, [.025, .975]).tolist(),
                sources=len(sources), valid_draws=len(differences))


def evaluate_rows(rows, scores, retentions, reconstruction_error):
    names = ['full'] + [f'rank{rank}' for rank in RANKS]
    labels = np.concatenate([row['labels'] for row in rows])
    complete = np.concatenate([scores[row['id']]['complete_window'] for row in rows])
    metrics = {}
    for name in names:
        values = np.concatenate([scores[row['id']][name] for row in rows])
        metrics[name] = score_metrics(labels, values)
        metrics[name]['complete_8_token_windows'] = score_metrics(labels[complete], values[complete])
    matched = json.loads(Path('outputs/repetition_dominance_20261007/matched_tokens.json').read_text())
    pairs = matched['strict']
    pairing = pair_statistics(rows, scores, pairs, names)
    retention_report = {}
    for rank, retention in retentions.items():
        variance = np.asarray(retention['variance'])
        readout = np.asarray(retention['readout'])
        retention_report[rank] = dict(variance_mean=float(variance.mean()),
                                      readout_mean=float(readout.mean()),
                                      variance_quantiles=np.quantile(variance, [.1, .5, .9]).tolist(),
                                      readout_quantiles=np.quantile(readout, [.1, .5, .9]).tolist(),
                                      high_variance_low_readout_fraction=float(
                                          ((variance >= .95) & (readout < .8)).mean()))
    differences = {f'rank{rank}_minus_full': source_difference(rows, scores, 'full', f'rank{rank}')
                   for rank in (1, 2, 4)}
    return dict(supervision='naturally supervised frozen readout, not unsupervised',
                answers=len(rows), tokens=len(labels), wrong=int(labels.sum()),
                metrics=metrics, paired_same_BPE=pairing, retention=retention_report,
                source_bootstrap=differences, full_rank_max_abs_error=reconstruction_error,
                evidence_scope='1024 self-head observation matrices, not native edge compression or causal transport')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    freeze_inputs(args.output)

    prepared = torch.load(BASE / 'prepared.pt', weights_only=False)
    rows = prepared['test']
    scores, retentions, error = measure_rows(rows)
    frozen = {f'{identity}/{name}': values for identity, features in scores.items()
              for name, values in features.items()}
    score_path = args.output / 'frozen_scores.npz'
    np.savez_compressed(score_path, **frozen)
    write_json(args.output / 'SCORE_FREEZE.json', dict(sha256=sha256(score_path),
               input_manifest_sha256=sha256(args.output / 'INPUT_MANIFEST.json')))

    result = evaluate_rows(rows, scores, retentions, error)
    write_json(args.output / 'RESULTS.json', result)
    print(json.dumps(dict(metrics=result['metrics'], retention=result['retention'],
                         max_error=error), indent=2), flush=True)


if __name__ == '__main__':
    main()
