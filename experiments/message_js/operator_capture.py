"""Residual/FFN message decomposition and exact output responses on 48 answers."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.decision_risk_flow.kernel import contract_edges
from experiments.decision_risk_flow.native import prefill, replay
from experiments.decision_risk_flow.run import load_model
from .operator import response_tangents, full_vocabulary_metric, DIRECTIONS


def roster(base):
    manifest = read_json(base / 'manifest.json')
    records = manifest['records']
    selected = [row for row in records if row['role'] in ('regression', 'natural')]
    for task in ('QA', 'Summary', 'Data2txt'):
        for role in ('fit', 'dev'):
            selected.extend([row for row in records if row['role'] == role and row['task'] == task][:4])
    return selected, Path(manifest['samples'])


def head_decomposition(model, capture, logits, actual, alternatives, positions):
    margin = logits.gather(1, actual[:, None]) - logits.gather(1, alternatives[:, None])
    layers = len(model.model.layers)
    gradients = torch.autograd.grad(margin.sum(), capture.writes + capture.mlp_writes)
    residual, ffn = [], []
    for layer, module in enumerate(model.model.layers):
        total = contract_edges(gradients[layer], capture.messages[layer], module.self_attn.o_proj.weight)
        direct = contract_edges(gradients[layers + layer], capture.messages[layer], module.self_attn.o_proj.weight)
        for offset, position in enumerate(positions.tolist()):
            total[offset, :, position] += total[offset, :, -1]
            direct[offset, :, position] += direct[offset, :, -1]
        residual.append(direct[..., :-1].permute(1, 0, 2).detach().cpu().numpy())
        ffn.append((total - direct)[..., :-1].permute(1, 0, 2).detach().cpu().numpy())
    return np.stack(residual), np.stack(ffn)


def cancellation(direct, transformed):
    denominator = np.abs(direct).sum(-1) + np.abs(transformed).sum(-1)
    total = np.abs(direct + transformed).sum(-1)
    return 1 - np.divide(total, denominator, out=np.ones_like(total), where=denominator > 0)


def validate_tangents(model, cache, tokens, prompt, tangents, output):
    token = min(10, len(tangents) - 1)
    positions = torch.tensor([prompt - 1 + token], device=model.device)
    results = []
    for channel in range(3):
        expected = torch.tensor(tangents[token, channel], device=model.device)
        direction = torch.eye(3, device=model.device)[channel]
        changed = []
        with torch.no_grad():
            for sign in (-1, 1):
                final, _, _ = replay(model, cache, tokens, positions, checkpoints=(32,),
                    gate=dict(scales=torch.ones(3, device=model.device) + sign * .005 * direction,
                              prompt_length=prompt))
                changed.append(final[0])
        finite = (changed[1] - changed[0]) / .01
        relative = float((finite - expected).norm() / expected.norm().clamp_min(1e-8))
        results.append(dict(direction=DIRECTIONS[channel], relative_error=relative))
        if relative > .03:
            write_json(output / 'finite_validation_failed.json', results)
            raise ValueError(f'JVP finite difference mismatch: {results[-1]}')
    write_json(output / 'finite_validation.json', dict(token=token, epsilon=.005, results=results))


def capture_answer(model, prompt, answer, base, output, validate=False):
    output.mkdir()
    tokens = prompt + answer[:-1]
    count = len(answer)
    cache, _, baseline = prefill(model, prompt, answer, checkpoints=(32,))
    old = np.load(base / 'readouts.npz')
    alternatives = old['alternative']
    shape = (32, 32, count, len(tokens))
    residual = np.lib.format.open_memmap(output / 'residual.npy', mode='w+', dtype='float32', shape=shape)
    ffn = np.lib.format.open_memmap(output / 'ffn.npy', mode='w+', dtype='float32', shape=shape)
    metric_rows, margin_rows, tangent_rows, errors = [], [], [], []
    batch = 16 if len(tokens) <= 1500 else 8
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        positions = torch.arange(len(prompt) - 1 + start, len(prompt) - 1 + stop, device=model.device)
        actual = torch.tensor(answer[start:stop], device=model.device)
        other = torch.tensor(alternatives[start:stop], device=model.device)
        final, tangent = response_tangents(model, cache, tokens, positions, len(prompt))
        gram, margin = full_vocabulary_metric(model, final, tangent, actual, other)
        error = float((final - baseline[start:stop]).norm() / baseline[start:stop].norm())
        if error > .0005:
            raise ValueError(f'JVP baseline mismatch: {output.name} {start}: {error}')
        errors.append(error)
        metric_rows.append(gram.cpu().numpy())
        margin_rows.append(margin.cpu().numpy())
        tangent_rows.append(tangent.cpu().numpy())
        current, _, capture = replay(model, cache, tokens, positions, checkpoints=(32,))
        logits = model.lm_head(current)
        direct, transformed = head_decomposition(model, capture, logits, actual, other, positions)
        residual[:, :, start:stop] = direct
        ffn[:, :, start:stop] = transformed
        del capture, current, logits, final, tangent, gram, margin
    residual.flush()
    ffn.flush()
    prompt_cancel = cancellation(residual[..., :len(prompt)], ffn[..., :len(prompt)])
    history_cancel = cancellation(residual[..., len(prompt):], ffn[..., len(prompt):])
    tangents = np.concatenate(tangent_rows)
    if validate:
        validate_tangents(model, cache, tokens, len(prompt), tangents, output)
    np.savez_compressed(output / 'operator.npz', gram=np.concatenate(metric_rows), margin=np.concatenate(margin_rows),
        final_state=baseline.cpu().numpy(), tangent=tangents, token_ids=answer,
        prompt_cancel=prompt_cancel, history_cancel=history_cancel, replay_errors=errors, directions=DIRECTIONS)
    write_json(output / 'execution.json', dict(query_batch=batch, labels_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('outputs/message_js_20260928_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model', default='/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    records, samples = roster(args.base)
    if not args.resume:
        args.output.mkdir(exist_ok=False)
        write_json(args.output / 'manifest.json', dict(records=records, base=str(args.base.resolve()), samples=str(samples)))
        write_json(args.output / 'protocol.json', dict(method='residual_ffn_full_output_operator_v2',
            labels_used=False, directions=DIRECTIONS, fisher='exact full vocabulary for three specified directions',
            full_jacobian=False, output_sketch=False, query_batch='16 when length<=1500 else 8',
            retained='full head/key residual and FFN derivatives; complete final states and tangents',
            past='original fixed KV; current query downstream exact, historical formation not measured',
            reference='first four sources/task/fit and dev from original blind hash roster',
            baseline='message_js v1, all outputs preserved'))
    else:
        assert records == read_json(args.output / 'manifest.json')['records']
    model = load_model(args.model)
    started = perf_counter()
    for index, row in enumerate(records):
        if (args.output / row['key'] / 'operator.npz').exists():
            continue
        if row['kind'] == 'observer':
            prompt, response = inputs(row)
            answer = response['answer_ids']
        else:
            with np.load(samples / row['trace']) as trace:
                length = int(trace['prompt_length'])
                prompt, answer = trace['token_ids'][:length].tolist(), trace['token_ids'][length:].tolist()
        capture_answer(model, prompt, answer, args.base / row['key'], args.output / row['key'], validate=index == 0)
        print(dict(done=index+1, total=len(records), key=row['key'], tokens=len(answer),
            seconds=round(perf_counter()-started, 1), peak_gib=round(torch.cuda.max_memory_allocated()/2**30, 2)), flush=True)
    write_json(args.output / 'capture_complete.json', dict(status='complete', answers=len(records),
        seconds=perf_counter()-started, labels_read=False, peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
