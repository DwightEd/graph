"""Native fixed-past message gate derivatives without head/window compression."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.decision_risk_flow.kernel import contract_edges
from experiments.decision_risk_flow.native import prefill, replay, confidence
from experiments.decision_risk_flow.run import load_model
from .measure import FIELDS, measure_head


def capture_answer(model, prompt, answer, special_ids, directory):
    directory.mkdir()
    count, layers = len(answer), len(model.model.layers)
    tokens = prompt + answer[:-1]
    cache, base_states, base_final = prefill(model, prompt, answer)
    observed, alternatives = confidence(model, base_final, answer, len(prompt), special_ids)
    heads = model.config.num_attention_heads
    shape = (layers, heads, count, len(tokens))
    attention = np.lib.format.open_memmap(directory / 'attention.npy', mode='w+', dtype='float32', shape=shape)
    derivative = np.lib.format.open_memmap(directory / 'derivative.npy', mode='w+', dtype='float32', shape=shape)
    errors = []
    for start in range(0, count, 16):
        stop = min(start + 16, count)
        positions = torch.arange(len(prompt) - 1 + start, len(prompt) - 1 + stop, device=model.device)
        final, states, capture = replay(model, cache, tokens, positions)
        logits = model.lm_head(final).float()
        with torch.no_grad():
            expected = model.lm_head(base_final[start:stop]).float()
            error = float((logits - expected).abs().max())
            state_error = float((states - base_states[start:stop]).norm() / base_states[start:stop].norm())
        errors.append((error, state_error))
        if error > .005 or state_error > .0005:
            raise ValueError(f'Native replay mismatch: {directory.name} {start} {errors[-1]}')
        targets = torch.tensor(answer[start:stop], device=model.device)
        other = alternatives[start:stop].to(model.device)
        margin = logits.gather(1, targets[:, None]) - logits.gather(1, other[:, None])
        gradients = torch.autograd.grad(margin.sum(), capture.writes)
        for layer, gradient in enumerate(gradients):
            edges = contract_edges(gradient, capture.messages[layer], model.model.layers[layer].self_attn.o_proj.weight)
            weights = capture.messages[layer][0].permute(1, 0, 2)
            # Merge appended recomputed self into its absolute key; cached self is masked.
            for offset, position in enumerate(positions.tolist()):
                edges[offset, :, position] += edges[offset, :, -1]
                weights[offset, :, position] += weights[offset, :, -1]
            derivative[layer, :, start:stop] = edges[..., :-1].permute(1, 0, 2).detach().cpu().numpy()
            attention[layer, :, start:stop] = weights[..., :-1].permute(1, 0, 2).cpu().numpy()
        del gradients, capture, logits, final, states, edges, weights, margin
    attention.flush()
    derivative.flush()
    measured = np.empty((layers, heads, count, len(FIELDS)), dtype=np.float32)
    for layer in range(layers):
        for head in range(heads):
            measured[layer, head] = measure_head(attention[layer, head], derivative[layer, head], len(prompt))
    np.savez_compressed(directory / 'readouts.npz', measured=measured, confidence=observed.numpy(),
                        alternative=alternatives.numpy(), token_ids=answer, replay_errors=errors,
                        prompt_length=len(prompt), fields=FIELDS)


def prepare(old, output, samples):
    records = read_json(old / 'manifest.json')['records']
    records = [dict(row, kind='observer') for row in records]
    natural = [json.loads(line) for line in (samples / 'samples.jsonl').read_text().splitlines()]
    for row in natural:
        records.append(dict(key=Path(row['trace']).stem, source_id=str(row['source_id']),
                            task='QA', kind='natural_replay', role='natural', trace=row['trace']))
    output.mkdir(exist_ok=False)
    write_json(output / 'manifest.json', dict(records=records, samples=str(samples.resolve())))
    write_json(output / 'protocol.json', dict(method='message-js-v1', labels_used=False,
        main='equal percentile mean of influence JS, read/use JS, prompt positive deficit; headwise ancestry propagation; q90 heads',
        roots='exact prompt positions; same-channel DAG surrogate, not native cross-layer provenance',
        operator='native actual-vs-best-other margin derivative at each layer/head/query/key; fixed original past KV',
        retained='float32 full attention and signed gate derivative, all physical axes; no rank projection',
        reference='old fit 36 sources, unlabeled; old dev 24 sources for thresholds, unlabeled',
        threshold='per-task 95th percentile of all eligible unlabeled dev tokens, strict greater; not normal FPR guarantee',
        controls=['no propagation', 'lag-band permuted history', 'original attention JS', 'entropy', 'surprisal'],
        prior_discovery='historically label-informed; scoring and calibration use no labels',
        natural='all 16 original sampled texts replayed with same model, FP32 not bit-exact original generation',
        known_cases='eight exposed regression answers; labels for evaluation only'))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    parser.add_argument('--samples', type=Path, default=Path('../reanchor/outputs/samples_20260911_145421_235'))
    parser.add_argument('--model', default='/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    if args.output.exists():
        records = read_json(args.output / 'manifest.json')['records']
    else:
        records = prepare(args.old, args.output, args.samples)
    model = load_model(args.model)
    from transformers import AutoTokenizer
    special = AutoTokenizer.from_pretrained(args.model, local_files_only=True).all_special_ids
    started = perf_counter()
    records.sort(key=lambda row: row['role'] not in ('regression', 'natural'))
    for index, record in enumerate(records):
        directory = args.output / record['key']
        if (directory / 'readouts.npz').exists():
            continue
        if record['kind'] == 'observer':
            prompt, response = inputs(record)
            answer = response['answer_ids']
        else:
            with np.load(args.samples / record['trace']) as data:
                length = int(data['prompt_length'])
                prompt, answer = data['token_ids'][:length].tolist(), data['token_ids'][length:].tolist()
        capture_answer(model, prompt, answer, special, directory)
        print(dict(done=index+1, total=len(records), key=record['key'], tokens=len(answer),
                   seconds=round(perf_counter()-started, 1)), flush=True)
    write_json(args.output / 'capture_complete.json', dict(status='complete', answers=len(records),
        seconds=perf_counter()-started, peak_bytes=torch.cuda.max_memory_allocated(), labels_read=False))


if __name__ == '__main__':
    main()
