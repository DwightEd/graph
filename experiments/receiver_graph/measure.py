"""Measure observed history edges; do not infer causality from state similarity."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import rel_entr
from experiments.decision_risk_flow.data import read_json, write_json

FIELDS = ('source_attention', 'source_positive', 'source_negative',
          'history_positive', 'history_negative', 'source_address_change_js')


def conditional_mean(values, ids):
    """Exact endpoint mean within each receiver's lag/copy group, per head."""
    result = np.zeros_like(values)
    for receiver in range(1, len(ids)):
        sender = np.arange(receiver)
        lag_bin = np.ceil(np.log2(receiver-sender)).astype(int)
        copied = ids[:receiver]==ids[receiver]
        group = lag_bin*2+copied
        for label in np.unique(group):
            indices = np.flatnonzero(group==label)
            result[:, receiver, indices] = values[:, receiver, indices].mean(-1)[:, None]
    return result


def history_joint(attention, derivative, prompt):
    count = attention.shape[1]
    reading = np.zeros((attention.shape[0], count, count), dtype=np.float32)
    influence = np.zeros_like(reading)
    reading[..., :-1] = attention[..., prompt:]
    absolute = np.abs(derivative)
    influence[..., :-1] = absolute[..., prompt:] / np.maximum(absolute.sum(-1, keepdims=True), 1e-30)
    return reading, np.sqrt(reading*influence)


def source_features(attention, derivative, source_indices, prompt):
    reading = attention[..., source_indices]
    effect = derivative[..., source_indices]
    history = derivative[..., prompt:]
    distribution = reading / np.maximum(reading.sum(-1, keepdims=True), 1e-30)
    middle = (distribution[:, 1:]+distribution[:, :-1])/2
    divergence = (rel_entr(distribution[:, 1:], middle).sum(-1)+
                  rel_entr(distribution[:, :-1], middle).sum(-1))/(2*np.log(2))
    change = np.pad(divergence, ((0,0), (1,0)))
    return np.stack((reading.sum(-1), np.maximum(effect, 0).sum(-1),
        np.maximum(-effect, 0).sum(-1), np.maximum(history, 0).sum(-1),
        np.maximum(-history, 0).sum(-1), change), -1)


def matched_sender(ids, receiver, sender):
    candidates = np.arange(receiver)
    lag = np.ceil(np.log2(receiver-candidates)).astype(int)
    copy = ids[candidates]==ids[receiver]
    eligible = candidates[(lag==lag[sender]) & (copy==copy[sender])]
    position = int(np.flatnonzero(eligible==sender)[0])
    return int(eligible[(position+1)%len(eligible)])


def strong_edges(graph, ids):
    selected = []
    receivers = set()
    for flat in np.argsort(graph.ravel())[::-1]:
        receiver, sender = np.unravel_index(flat, graph.shape)
        if graph[receiver, sender]<=0 or len(selected)==2:
            break
        if receiver in receivers or receiver-sender<3:
            continue
        other = matched_sender(ids, receiver, sender)
        if other==sender:
            continue
        selected.append(dict(receiver=int(receiver), sender=int(sender), control_sender=other))
        receivers.add(receiver)
    return selected


def record_measurements(row, manifest, output):
    original = Path(manifest['base'])/row['key']
    with np.load(original/'readouts.npz') as saved:
        prompt = int(saved['prompt_length'])
        ids = saved['token_ids']
    source = Path(manifest['relation_base'])/'sources'/row['source_id']/'input.npz'
    source_indices = np.flatnonzero(np.load(source)['source_mask'])
    attention = np.load(original/'attention.npy', mmap_mode='r')
    derivative = np.load(original/'derivative.npy', mmap_mode='r')
    count = len(ids)
    graphs = {name: np.zeros((count, count)) for name in ('attention', 'joint', 'selective', 'matched')}
    head_values = np.empty((32, 32, count, len(FIELDS)), dtype=np.float32)
    for layer in range(32):
        reading, joint = history_joint(attention[layer], derivative[layer], prompt)
        matched = conditional_mean(joint, ids)
        for name, values in (('attention', reading), ('joint', joint),
                             ('selective', np.maximum(joint-matched, 0)), ('matched', matched)):
            graphs[name] += values.sum(0)/1024
        head_values[layer] = source_features(attention[layer], derivative[layer], source_indices, prompt)
    directory = output/row['key']
    directory.mkdir()
    np.savez_compressed(directory/'graph.npz', **graphs, token_ids=ids)
    np.savez_compressed(directory/'head_readouts.npz', values=head_values, fields=FIELDS)
    return graphs, ids, prompt


def select_interventions(row, manifest, graphs, ids, prompt):
    original = Path(manifest['base'])/row['key']
    attention = np.load(original/'attention.npy', mmap_mode='r')
    derivative = np.load(original/'derivative.npy', mmap_mode='r')
    selected = strong_edges(graphs['selective'], ids)
    for edge in selected:
        receiver, sender = edge['receiver'], edge['sender']
        reading = np.asarray(attention[:, :, receiver])
        effect = np.asarray(derivative[:, :, receiver])
        joint = np.sqrt(reading[..., prompt+sender]*np.abs(effect[..., prompt+sender])/
                        np.maximum(np.abs(effect).sum(-1), 1e-30))
        layer, head = np.unravel_index(joint.argmax(), (32,32))
        edge.update(layer=int(layer), head=int(head), key=row['key'],
                    attention=float(reading[layer,head,prompt+sender]),
                    derivative=float(effect[layer,head,prompt+sender]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('outputs/context_response_recurrence_20260929_v3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    interventions = []
    for row in manifest['records']:
        graphs, ids, prompt = record_measurements(row, manifest, args.output)
        for name in ('context.npz', 'responses.npz'):
            (args.output/row['key']/name).symlink_to((args.previous/row['key']/name).resolve())
        if row['role']=='regression' or row['key'] in ('00005', '00006', '00012', '00013'):
            interventions.extend(select_interventions(row, manifest, graphs, ids, prompt))
        print('measured', row['key'], len(ids), flush=True)
    write_json(args.output/'interventions.json', dict(edges=interventions, labels_used=False,
        selection='two highest selective graph edges at distinct receivers, lag>=3; strongest joint head; cyclic matched endpoint'))
    write_json(args.output/'features_complete.json', dict(status='complete', answers=len(manifest['records']),
        physical_heads=1024, labels_used=False, scope='observed fixed-past message sensitivity, not causal propagation across generated tokens'))


if __name__=='__main__':
    main()
