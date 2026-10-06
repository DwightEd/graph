"""Check native target alignment and embedding-gate derivatives on real 8B inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL


@torch.no_grad()
def native_target(model, ids, token_id, root=None, change=0.):
    embeddings = model.model.embed_tokens(torch.tensor([ids], device=model.device))
    if root is not None:
        embeddings[:, root] *= 1 + change
    hidden = model.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state[0, -1]
    return float(model.lm_head(hidden).float().log_softmax(-1)[token_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('roots', 'sparse'), default='roots')
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/automatic_evidence_20260930_v2/manifest.json'))
    parser.add_argument('--keys', nargs='+', default=['15604', '9022'])
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 15, 31])
    parser.add_argument('--query-count', type=int, default=8)
    parser.add_argument('--vocabulary-chunk', type=int, default=8192)
    args = parser.parse_args()
    if args.stage == 'sparse':
        sparse_validate(args)
        return
    records = read_json(args.trace / 'manifest.json')['records']
    model = load_model(MODEL)
    checks = []
    for row in records:
        positions = (96, 100) if row['key'] == '15604' else (22, 30)
        with np.load(args.trace / row['key'] / 'trace.npz') as saved:
            for target in positions:
                ids = row['prompt'] + row['response']['answer_ids'][:target]
                token_id = row['response']['answer_ids'][target]
                original = native_target(model, ids, token_id)
                effect = saved['root_effect'][target]
                root = int(np.argmax(np.abs(effect)))
                for epsilon in (.001, .005):
                    positive = native_target(model, ids, token_id, root, epsilon)
                    negative = native_target(model, ids, token_id, root, -epsilon)
                    finite = (positive - negative) / (2 * epsilon)
                    checks.append(dict(key=row['key'], target=target, root=root, epsilon=epsilon,
                        analytic=float(effect[root]), finite=finite,
                        slope_error=abs(finite - float(effect[root])),
                        prefix_error=abs(original - float(saved['logp'][target]))))
    write_json(args.trace / 'numerical_checks.json', dict(checks=checks, forward_calls=20,
        description='exact prefix recomputation vs full causal pass; symmetric embedding gate'))
    assert all(row['prefix_error'] < .001 for row in checks)
    assert all(row['slope_error'] < .02 + .05 * abs(row['analytic']) for row in checks)
    print('8 finite differences and 4 target/prefix checks passed', flush=True)


def sparse_addresses(row, layers, query_count):
    """Label-free physical grid; source keys are positions, not evidence labels."""
    prompt_length = len(row['prompt'])
    answer_length = len(row['response']['answer_ids'])
    source_keys = np.flatnonzero(row['source']['source_mask'])
    source_id = 'external-source:' + row['original']['source_id']
    history_id = 'answer-history:' + row['key']
    positions = np.unique(np.linspace(1, answer_length - 1, query_count, dtype=int))
    addresses = []
    for layer in layers:
        for slot, target in enumerate(positions):
            head = slot * 7 % 32
            query = prompt_length + int(target) - 1
            key = int(source_keys[slot * len(source_keys) // len(positions)])
            addresses.append((layer, head, query, key, source_id))
            addresses.append((layer, head, query, query - 1, history_id))
    addresses.append((layers[-1], 0, prompt_length + answer_length - 1, 0, 'terminal-unobservable'))
    return addresses


def sparse_validate(args):
    import csv
    import gc
    from state_audit.model.adapter import ModelAdapter
    from .messages import native_edge_trace, sparse_edge_vjp, finite_edge_effect
    from .trace import checkpoint_ffn
    from experiments.decision_risk_flow.precision import chunk_unembedding
    from time import perf_counter

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read_json(args.manifest)
    rows = [row for row in manifest['records'] if row['key'] in args.keys]
    write_json(args.output / 'manifest.json', dict(model=manifest['model'], records=rows,
                                                  labels_read=False))
    write_json(args.output / 'protocol.json', dict(keys=args.keys, layers=args.layers,
        query_count=args.query_count, vocabulary_chunk=args.vocabulary_chunk,
        manifest=str(args.manifest), model=manifest['model'],
        selection='uniform positions, lexical physical heads, uniform source-key grid; no gold',
        finite_selection='largest absolute measured derivative per layer; fidelity diagnostic, not random population',
        objective='each original-token logp independently; sparse amplitude deletion, no renormalization',
        full_original_history=True, labels_used=False, detection_performance=False))
    total_started = perf_counter()
    model = load_model(manifest['model'])
    chunk_unembedding(model, args.vocabulary_chunk)
    adapter = ModelAdapter(model)
    model.set_attn_implementation('sdpa')
    checkpoint_ffn(model)
    summaries = []
    for row in rows:
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        trace = native_edge_trace(adapter, row['prompt'], row['response']['answer_ids'],
                                  sparse_addresses(row, args.layers, args.query_count))
        result = sparse_edge_vjp(trace)
        directory = args.output / row['key']
        directory.mkdir()
        with (directory / 'edges.csv').open('w') as file:
            writer = csv.writer(file)
            writer.writerow(('layer', 'head', 'query', 'key', 'source_id', 'target', 'valid', 'attention', 'delete_derivative'))
            for edge, address in enumerate(result['addresses']):
                for column, target in enumerate(result['targets']):
                    valid = bool(result['valid'][edge, column])
                    effect = float(result['delete_derivative'][edge, column]) if valid else ''
                    writer.writerow((*address, target, valid, float(trace.attention[edge]), effect))
        np.savez_compressed(directory / 'measurement.npz',
            token_ids=row['response']['answer_ids'], offsets=row['response']['offsets'],
            delete_derivative=result['delete_derivative'].numpy(), valid=result['valid'].numpy(),
            attention=trace.attention.cpu().numpy())
        checks = []
        for layer in args.layers:
            candidates = [i for i, address in enumerate(trace.addresses)
                          if address[0] == layer and result['valid'][i].any()]
            edge = max(candidates, key=lambda i: float(result['delete_derivative'][i].abs().nan_to_num().max()))
            target = int(result['delete_derivative'][edge].abs().nan_to_num().argmax())
            analytic = float(result['delete_derivative'][edge, target])
            finite = {str(alpha): finite_edge_effect(adapter, trace, edge, alpha)
                      for alpha in (0., -.01, .01, .25, 1.)}
            derivative = float((finite['0.01'][target] - finite['-0.01'][target]) / .02)
            first_target = trace.addresses[edge][2] - len(row['prompt']) + 1
            checks.append(dict(address=trace.addresses[edge], target=target, analytic=analytic,
                symmetric_slope=derivative, slope_absolute_error=abs(derivative-analytic),
                slope_relative_error=abs(derivative-analytic)/max(abs(analytic), 1e-12),
                sham_max=float(finite['0.0'].abs().max()),
                doses={alpha: float(values[target]) for alpha, values in finite.items()},
                noncausal_max=max(float(values[:first_target].abs().max()) if first_target else 0.
                                 for values in finite.values())))
        summary = dict(key=row['key'], tokens=len(row['response']['answer_ids']),
            sparse_edges=len(trace.addresses), backward_calls=result['backward_calls'],
            forward_calls=1 + 5 * len(args.layers), seconds=perf_counter()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), checks=checks)
        write_json(directory / 'checks.json', summary)
        summaries.append(summary)
        print(summary, flush=True)
        del trace, result
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / 'complete.json', dict(status='complete', records=summaries,
        tokens=sum(row['tokens'] for row in summaries), labels_read=False, new_detection_scores=False,
        total_seconds=perf_counter()-total_started, cuda_version=torch.version.cuda,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name()))


if __name__ == '__main__':
    main()
