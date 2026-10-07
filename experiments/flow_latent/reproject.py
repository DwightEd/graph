"""Fit per-head coordinate axes on unlabeled fit nodes; reuse native raw graph."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from experiments.token_backtrace.grounded_projection_data import write_json
from .data import digest_files
from .flow import coordinate_projection, matched_mappings, rooted_paths


def fit_head_axes(capture, cases):
    total = 0
    sums = torch.zeros(32, 32, 128, dtype=torch.float64, device='cuda')
    scatter = torch.zeros(32, 32, 128, 128, dtype=torch.float64, device='cuda')
    for case in cases:
        if case['cohort'] != 'fit':
            continue
        nodes = np.load(capture / case['id'] / 'node.npy', mmap_mode='r')
        values = torch.tensor(np.asarray(nodes), device='cuda', dtype=torch.float64)
        sums += values.sum(0)
        batched = values.reshape(len(nodes), 1024, 128).transpose(0, 1)
        scatter += torch.bmm(batched.transpose(1, 2), batched).reshape(32, 32, 128, 128)
        total += len(nodes)
    covariance = (scatter - sums[..., :, None] * sums[..., None, :] / total) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.cpu().numpy())
    axes = eigenvectors[..., -16:].astype(np.float32)
    retained = eigenvalues[..., -16:].sum(-1) / eigenvalues.sum(-1)
    fixed = coordinate_projection().astype(np.float64)
    old_energy = np.einsum('dr,lhde,er->lh', fixed, covariance.cpu().numpy(), fixed).sum()
    return axes, dict(fit_tokens=total, head_variance=retained.tolist(),
        weighted_head_variance=float(eigenvalues[..., -16:].sum() / eigenvalues.sum()),
        initial_fixed4_raw_variance=float(old_energy / eigenvalues.sum()),
        axes_rule='top16 eigenvectors of each physical head fit-node covariance; no centering messages')


def reproject_case(capture, output, case, axes):
    directory = capture / case['id']
    metadata = json.loads((directory / 'capture.json').read_text())
    prompt, tokens = metadata['prompt'], metadata['original_token_ids']
    mappings, _ = matched_mappings(tokens, prompt, case['source']['source_mask'])
    mappings = torch.tensor(np.stack(mappings), device='cuda')
    stored_weights = np.load(directory / 'attention.npy', mmap_mode='r')
    stored_values = np.load(directory / 'values.npy', mmap_mode='r')
    stored_nodes = np.load(directory / 'node.npy', mmap_mode='r')
    source = torch.zeros(len(tokens), dtype=torch.bool, device='cuda')
    source[:prompt] = torch.tensor(case['source']['source_mask'], device='cuda')
    history = torch.arange(len(tokens), device='cuda') >= prompt
    other = ~(source | history)
    parts = {}
    for layer in range(32):
        projection = torch.tensor(axes[layer], device='cuda')
        values = torch.tensor(np.asarray(stored_values[:, layer]), device='cuda', dtype=torch.float32)
        nodes = torch.tensor(np.asarray(stored_nodes[:, layer]), device='cuda', dtype=torch.float32)
        projected = torch.einsum('khd,hdr->khr', values, projection)
        parts.setdefault('node', []).append(torch.einsum('thd,hdr->thr', nodes, projection).cpu().numpy())
        attention = torch.tensor(np.asarray(stored_weights[layer]), device='cuda', dtype=torch.float32)
        for kind in ('native', 'rewired'):
            weights = attention if kind == 'native' else attention.gather(2, mappings)
            local = {}
            for name, mask in (('source', source), ('history', history), ('other', other)):
                message = torch.einsum('thk,khd->thd', weights[:, :, mask], projected[mask])
                local[name] = message.cpu().numpy()
            local['root'] = rooted_paths(local['source'], weights[:, :, prompt:].cpu().numpy())
            for name, message in local.items():
                parts.setdefault(kind + '_' + name, []).append(message)
    features = {name: np.stack(layers, axis=1) for name, layers in parts.items()}
    original = np.load(directory / 'features.npz')
    target = output / case['id']
    target.mkdir()
    np.savez(target / 'features.npz', token_id=original['token_id'], nll=original['nll'], **features)
    print(f'REPROJECT {case["id"]} {case["cohort"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.capture / 'inputs.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    axes, diagnostics = fit_head_axes(args.capture, inputs['cases'])
    np.save(args.output / 'axes.npy', axes)
    write_json(args.output / 'inputs.json', inputs)
    write_json(args.output / 'projection.json', dict(base_capture=str(args.capture),
        diagnostics=diagnostics, hashes=digest_files([Path(__file__), args.capture / 'inputs.json']),
        natural_label_fit=False, raw_storage_precision='float16; recompute transport in float32'))
    for case in inputs['cases']:
        reproject_case(args.capture, args.output, case, axes)
    write_json(args.output / 'execution.json', dict(status='DONE', cases=len(inputs['cases']),
        new_8b_forwards=0, seconds=time.time() - started,
        peak_memory=torch.cuda.max_memory_allocated(), base_capture=str(args.capture)))


if __name__ == '__main__':
    main()
