"""Collect native Fisher window-response kernels for complete original answers."""
from time import perf_counter

import numpy as np
import torch

from .data import inputs, read_json, write_json
from .geometry import describe
from .kernel import batch_responses, node_addresses
from .native import prefill, replay
from .run import load_model
from .validation import validate_gate


def capture_answer(model, record, static, query_batch, rank):
    prompt, response = inputs(record)
    answer = response['answer_ids']
    tokens = prompt + answer[:-1]
    cache, base_states, base_final = prefill(model, prompt, answer)
    rows, choice_rows, sketches, attentions = [], [], [], []
    errors = []
    for start in range(0, len(answer), query_batch):
        stop = min(start + query_batch, len(answer))
        positions = torch.arange(len(prompt) - 1 + start, len(prompt) - 1 + stop, device=model.device)
        final, states, capture = replay(model, cache, tokens, positions)
        logits = model.lm_head(final).float()
        with torch.no_grad():
            reference = model.lm_head(base_final[start:stop]).float()
            error = (logits.detach() - reference).abs().max().item()
            relative_state_error = ((states.detach().float() - base_states[start:stop].float()).norm()
                                    / base_states[start:stop].float().norm()).item()
        errors.append((error, relative_state_error))
        if error > .005 or relative_state_error > .0005:
            raise ValueError(f"Native replay fidelity failed: {record['id']} {start} {errors[-1]}")
        nodes, windows = node_addresses(tokens, positions, len(prompt), response['special_ids'])
        actual = torch.tensor(answer[start:stop], device=model.device)
        alternative = torch.tensor(static['alternative'][start:stop], device=model.device)
        choice, sketch, attention = batch_responses(model, capture, logits, actual,
            alternative, nodes, windows, rank)
        choice, sketch, attention = [x.cpu().numpy() for x in (choice, sketch, attention)]
        for index in range(stop - start):
            features, _ = describe(choice[index], sketch[index], attention[index], windows)
            rows.append(list(features.values()))
        choice_rows.append(choice)
        sketches.append(sketch)
        attentions.append(attention)
        del capture, logits, final, states
    return dict(features=np.asarray(rows), names=np.asarray(list(features)),
        choice=np.concatenate(choice_rows), sketch=np.concatenate(sketches),
        attention=np.concatenate(attentions), replay_errors=np.asarray(errors),
        windows=np.asarray(windows), token_ids=np.asarray(answer))


def capture_responses(args):
    model = load_model(args.model)
    records = read_json(args.output / 'manifest.json')['records']
    # Score the requested examples first, without inspecting their labels.
    records.sort(key=lambda row: row['role'] != 'regression')
    validation = validate_gate(model, records[0], args.output)
    print(dict(stage='native_gate_validation', **validation), flush=True)
    started = perf_counter()
    for index, record in enumerate(records):
        path = args.output / 'responses' / f"{record['key']}.npz"
        if path.exists():
            continue
        with np.load(args.output / 'static' / f"{record['key']}.npz") as source:
            static = {'alternative': source['alternative']}
        result = capture_answer(model, record, static, args.query_batch, args.sketch_rank)
        np.savez_compressed(path, **result)
        print(dict(stage='kernel', done=index + 1, total=len(records), id=record['key'],
            tokens=record['tokens'], seconds=round(perf_counter() - started, 2),
            max_logit_error=float(result['replay_errors'][:, 0].max()),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)), flush=True)
    write_json(args.output / 'responses_complete.json', dict(status='complete', answers=len(records),
        rank=args.sketch_rank, seconds=perf_counter() - started,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(), labels_read=False))
