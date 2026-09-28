"""Identify read/update/carrier candidates and list every exposed false alarm/miss."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import inputs, read_json, write_json
from .score import restart_weights


def token_pieces(row, manifest):
    if row['kind']=='observer':
        return inputs(row)[1]['token_text']
    with np.load(Path(manifest['samples'])/row['trace']) as saved:
        return saved['token_text'][int(saved['prompt_length']):].tolist()


def top_head_mean(values):
    return np.partition(values, -8, axis=0)[-8:].mean(0)


def node_measures(directory):
    with np.load(directory/'head_readouts.npz') as saved:
        value = saved['values'].reshape(1024, -1, 6)
    with np.load(directory/'graph.npz') as saved:
        graph = {name:saved[name] for name in ('selective', 'joint', 'attention')}
    source = value[:,:,1]+value[:,:,2]
    history = value[:,:,3]+value[:,:,4]
    use = source/np.maximum(source+history, 1e-30)
    observed = dict(source_read=top_head_mean(value[:,:,0]), source_use=top_head_mean(use),
        source_update=top_head_mean(value[:,:,5]), carrier=graph['selective'].sum(0),
        carrier_raw=graph['joint'].sum(0), reception=graph['selective'].sum(1),
        direct_heads_reading_source=(value[:,:,0]>.1).sum(0),
        pooling_self_weight=restart_weights(graph['selective']).diagonal())
    return value, observed


def summarize_nodes(output, manifest):
    nodes, candidates = [], []
    for row in manifest['records']:
        directory = output/row['key']
        value, observed = node_measures(directory)
        pieces = token_pieces(row, manifest)
        with np.load(Path(manifest['base'])/row['key']/'readouts.npz') as saved:
            confidence = saved['confidence']
        for token, piece in enumerate(pieces):
            nodes.append(dict(key=row['key'], role=row['role'], task=row['task'], token=token,
                text=piece, entropy=float(confidence[token,0]), margin=float(confidence[token,2]),
                **{name:float(values[token]) for name,values in observed.items()}))
        if row['role'] not in ('regression', 'natural'):
            continue
        for name in ('source_read', 'source_use', 'source_update', 'carrier'):
            positions = np.argsort(observed[name])[-5:][::-1]
            for token in positions:
                candidates.append(dict(key=row['key'], measure=name, token=int(token),
                    text=pieces[token], value=float(observed[name][token]),
                    interpretation='carrier is an input-key state; other measures describe the query predicting this output token'))
    with (output/'nodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=nodes[0])
        writer.writeheader()
        writer.writerows(nodes)
    write_json(output/'node_candidates.json', dict(candidates=candidates, labels_used_for_selection=False,
        scope='ranking candidates, not demonstrated semantic reanchor or error starts'))
    return nodes


def error_lists(output, manifest, nodes):
    previous = Path(manifest['previous'])
    with (previous/'token_audit.csv').open() as stream:
        old = list(csv.DictReader(stream))
    with (output/'token_audit.csv').open() as stream:
        new = list(csv.DictReader(stream))
    methods = ('original_full_reference_fixed', 'corroborated_only', 'original_corroborated')
    old = [r for r in old if r['method'] in methods]
    lookup = {(r['key'],str(r['token'])):r for r in nodes}
    rows = []
    for version, tokens in (('previous',old), ('new',new)):
        for row in tokens:
            if row['status'] not in ('FP', 'FN'):
                continue
            observed = lookup[row['key'], row['position']]
            rows.append(dict(version=version, key=row['key'], method=row['method'],
                token=int(row['position']), text=row['text'], status=row['status'], score=float(row['score']),
                **{name:observed[name] for name in ('source_read','source_use','source_update','carrier',
                    'direct_heads_reading_source','margin','entropy','pooling_self_weight')}))
    with (output/'all_errors.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r['version']=='previous' and r['method']=='corroborated_only' and r['status']=='FN']
    lines = ['# 上轮融合双起点全部漏检token', '', '0-based原token索引；71个FN。空白/subword依原样保留，不等于71个独立事实。', '']
    for key in dict.fromkeys(r['key'] for r in selected):
        group = [r for r in selected if r['key']==key]
        lines.append(f"## {key}: {len(group)} token")
        lines.extend(['', '|位置|原token|读来源>0.1的头数|observer margin|', '|---:|---|---:|---:|'])
        for row in group:
            lines.append(f"|{row['token']}|{repr(row['text'])}|{int(row['direct_heads_reading_source'])}|{row['margin']:.4f}|")
        lines.append('')
    (output/'PREVIOUS_MISSES.md').write_text('\n'.join(lines)+'\n')


def verify_axes(output, manifest):
    maxima = []
    for row in manifest['records']:
        with np.load(output/row['key']/'graph.npz') as saved:
            for name in ('attention','joint','selective','matched'):
                graph = saved[name]
                assert np.isfinite(graph).all() and (graph>=0).all()
                assert not np.triu(graph).any()
            error = np.max(abs(saved['joint'].sum(-1)-saved['matched'].sum(-1)))
            maxima.append(float(error))
            assert error<1e-6
        with np.load(output/row['key']/'head_readouts.npz') as saved:
            assert saved['values'].shape[:2]==(32,32)
            assert np.isfinite(saved['values']).all()
    write_json(output/'axis_verification.json', dict(status='passed', answers=len(maxima),
        maximum_matched_row_mass_error=max(maxima), physical_heads=1024,
        scope='causal key indexing, finite full-head measurements, conditional endpoint mass conservation'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    read_json(args.output/'evaluation.json')
    verify_axes(args.output, manifest)
    nodes = summarize_nodes(args.output, manifest)
    error_lists(args.output, manifest, nodes)
    print('Audited', len(nodes), 'nodes and saved all old/new FP/FN', flush=True)
