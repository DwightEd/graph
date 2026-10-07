"""Natural token detection using full-vector source projection and native replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from .grounded_projection import (capture_projection_inputs, content_candidates,
    history_transport, observed_attention, projected_carriers, replay_queries, token_logp)
from .grounded_projection_data import prepare, write_json


CUTS = (7, 15, 23, 31)
KINDS = ('graph', 'flat', 'rewired')


@torch.no_grad()
def capture(adapter, case):
    prompt = len(case['source']['prompt_with_source'])
    tokens = case['source']['prompt_with_source'] + case['response']['answer_ids']
    ids = adapter.input_ids(tokens)
    with capture_projection_inputs(adapter.native, prompt) as records:
        hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
    positions = torch.arange(len(tokens), device=ids.device)
    cosine, sine = adapter.native.model.rotary_emb(adapter.native.model.embed_tokens(ids), positions[None])
    target = ids[0, prompt:]
    baseline = token_logp(adapter.native, hidden[prompt - 1:-1], target)
    source, kernel, similarity = content_candidates(adapter.native, tokens, prompt, case['source']['source_mask'])
    del hidden
    return records, cosine[0], sine[0], target, baseline, source, kernel, similarity


@torch.no_grad()
def directions(adapter, records, prompt, cosine, sine, source, kernel):
    positions = torch.arange(prompt - 1, len(records[0]['key']), device=adapter.native.device)
    nodes, deltas, errors = [], {}, []
    for layer, record in enumerate(records):
        attention, values = observed_attention(adapter, layer, record, positions, cosine, sine)
        reconstructed = torch.einsum('thk,khd->thd', attention, values).flatten(1)
        errors.append(float((reconstructed - record['head'].to(reconstructed.device)).abs().max()))
        graph, flat, projected = projected_carriers(attention, values, source, kernel, prompt)
        nodes.append((1 - cosine_similarity(values[prompt:], projected, dim=-1)).cpu())
        if layer in CUTS:
            deltas[layer] = dict(graph=history_transport(attention, graph, prompt).cpu(),
                flat=history_transport(attention, flat, prompt).cpu(),
                rewired=history_transport(attention, graph, prompt, True).cpu())
    return torch.stack(nodes, 1), deltas, max(errors)


@torch.no_grad()
def response(adapter, records, layer, delta, cosine, sine, target, prompt, batch=64):
    scores = []
    for begin in range(0, len(target), batch):
        rows = torch.arange(begin, min(begin + batch, len(target)), device=target.device)
        hidden = replay_queries(adapter, records, layer, rows, rows + prompt - 1,
                                delta[rows.cpu()].to(target.device), cosine, sine)
        scores.append(token_logp(adapter.native, hidden, target[rows]))
    return torch.cat(scores)


@torch.no_grad()
def finite_canary(adapter, case, records, delta, cosine, sine, target, prompt):
    """Three fixed token positions; compare independent replay to full native calls."""
    rows = torch.tensor(sorted({0, len(target) // 2, len(target) - 1}), device=target.device)
    positions = rows + prompt - 1
    selected = delta[rows.cpu()].to(target.device)
    replay = replay_queries(adapter, records, 15, rows, positions, selected, cosine, sine)
    tokens = case['source']['prompt_with_source'] + case['response']['answer_ids']
    errors = []
    for offset, position in enumerate(positions):
        def change(message):
            modified = message.clone()
            modified[position] += selected[offset]
            return modified
        with adapter.bind('head_readout', 15, change, edit=True):
            native = adapter.forward(tokens)[position]
        errors.append(float((native - replay[offset]).abs().max()))
    assert max(errors) < 1e-3, f'independent finite replay failed: {errors}'
    return dict(positions=rows.cpu().tolist(), hidden_max_error=errors)


@torch.no_grad()
def score_case(adapter, case, output):
    started = time.time()
    prompt = len(case['source']['prompt_with_source'])
    records, cosine, sine, target, baseline, source, kernel, similarity = capture(adapter, case)
    nodes, deltas, reconstruction = directions(adapter, records, prompt, cosine, sine, source, kernel)
    identity = torch.stack([response(adapter, records, layer, torch.zeros_like(deltas[layer]['graph']),
                                    cosine, sine, target, prompt) for layer in CUTS], 1)
    identity_error = float((identity - baseline[:, None]).abs().max())
    assert identity_error < 2e-4, f'{case["id"]}: identity error {identity_error}'
    assert reconstruction < 2e-4, f'{case["id"]}: message reconstruction {reconstruction}'
    finite = (finite_canary(adapter, case, records, deltas[15]['graph'], cosine, sine, target, prompt)
              if case['id'] == '15604' else None)
    scores = dict(token_id=target.cpu().numpy(), nll=-baseline.numpy(),
                  node=nodes.mean((1, 2)).numpy(), static=1 - similarity.cpu().numpy())
    for kind in KINDS:
        patched = [response(adapter, records, layer, deltas[layer][kind], cosine, sine, target, prompt)
                   for layer in CUTS]
        effects = identity - torch.stack(patched, 1)
        scores[kind] = effects.mean(1).numpy()
        scores[kind + '_layers'] = effects.numpy()
    scores['edge_increment'] = scores['graph'] - scores['rewired']
    scores['zero_replay_bias'] = baseline[:, None].numpy() - identity.numpy()
    directory = output / case['id']
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / 'scores.npz', **scores)
    torch.save(dict(node_per_head=nodes, deltas=deltas, source_indices=source.cpu(),
                    content_kernel=kernel.cpu()), directory / 'directions.pt')
    write_json(directory / 'canaries.json', dict(identity_logp_max=identity_error,
        reconstruction_max=reconstruction, finite=finite, tokens=len(target), seconds=time.time() - started))
    print(f'SCORE {case["id"]} {case["cohort"]} T={len(target)} seconds={time.time()-started:.1f}', flush=True)


def freeze(output, cases):
    files = [Path(__file__), Path(__file__).with_name('grounded_projection.py'),
             Path(__file__).with_name('grounded_projection_data.py'),
             Path(__file__).with_name('grounded_projection_evaluate.py'), output / 'inputs.json']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if (output / 'protocol.json').exists():
        previous = json.loads((output / 'protocol.json').read_text())
        if previous['hashes'] != hashes:
            raise ValueError('Frozen code/input changed; use a new output directory')
        return
    write_json(output / 'protocol.json', dict(cuts=CUTS, kinds=KINDS, dose=1, nearest=16,
        numerical_baseline='zero-dose replay, same token batch and structural cut',
        source_kernel_temperature=.1, rewire_seed=73, no_label_fit=True,
        primary='joint_graph: source-reconstruction node + native graph correction, reference Fisher tails',
        threshold='task-specific unlabelled-reference 95th percentile; not calibrated normal FPR',
        heads='all 32 layers x 32 heads x 128 coordinates; no head selection',
        hashes=hashes))
    Path(output / 'code_snapshot').mkdir(exist_ok=True)
    for path in files[:-1]:
        (output / 'code_snapshot' / path.name).write_bytes(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=int, default=4)
    args = parser.parse_args()
    cases = (json.loads((args.output / 'inputs.json').read_text())['cases'] if args.output.exists()
             else prepare(args.output, args.references))
    freeze(args.output, cases)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(json.loads((args.output / 'inputs.json').read_text())['model']))
    started = time.time()
    for case in cases:
        if not (args.output / case['id'] / 'scores.npz').exists():
            score_case(adapter, case, args.output)
    write_json(args.output / 'execution.json', dict(status='DONE', cases=len(cases),
        seconds=time.time() - started, peak_memory=torch.cuda.max_memory_allocated(),
        natural_annotation_fields_used_for_scoring=False))


if __name__ == '__main__':
    main()
