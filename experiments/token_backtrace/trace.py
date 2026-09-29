"""Trace each actual-token objective to all original source/history embeddings."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL


def checkpoint_ffn(model):
    for layer in model.model.layers:
        original = layer.mlp.forward
        def forward(inputs, original=original):
            return checkpoint(original, inputs, use_reentrant=False)
        layer.mlp.forward = forward


def trace_answer(model, row, directory):
    prompt = row['prompt']
    answer = row['response']['answer_ids']
    tokens = torch.tensor([prompt + answer[:-1]], device=model.device)
    embeddings = model.model.embed_tokens(tokens).detach().requires_grad_(True)
    output = model.model(inputs_embeds=embeddings, use_cache=False)
    hidden = output.last_hidden_state[0, len(prompt) - 1:]
    roots = np.empty((len(answer), tokens.shape[1]), dtype=np.float32)
    norms = np.empty_like(roots)
    logp, entropy, alternative, margin = [], [], [], []
    started = perf_counter()
    for target, token_id in enumerate(answer):
        logits = model.lm_head(hidden[target]).float()
        distribution = logits.log_softmax(-1)
        objective = distribution[token_id]
        gradient = torch.autograd.grad(objective, embeddings, retain_graph=True)[0][0]
        roots[target] = (gradient * embeddings[0]).sum(-1).detach().cpu().numpy()
        norms[target] = gradient.norm(dim=-1).detach().cpu().numpy()
        rival_logits = logits.detach().clone()
        rival_logits[token_id] = -torch.inf
        foil = int(rival_logits.argmax())
        logp.append(float(objective.detach()))
        entropy.append(float(-(distribution.detach().exp() * distribution.detach()).sum()))
        alternative.append(foil)
        margin.append(float((logits[token_id] - logits[foil]).detach()))
        if target % 25 == 0:
            print(row['key'], target, len(answer), round(perf_counter()-started, 1), flush=True)
        assert not np.any(roots[target, len(prompt) + target:]), 'noncausal root gradient'
    np.savez_compressed(directory / 'trace.npz', token_ids=answer, root_effect=roots,
        root_norm=norms, logp=logp, entropy=entropy, alternative=alternative, margin=margin)
    write_json(directory / 'complete.json', dict(status='complete', targets=len(answer),
        backward_calls=len(answer), seconds=perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), objective='independent actual-token logp',
        full_history=True, labels_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='+', required=True)
    args = parser.parse_args()
    rows = read_json('outputs/token_evidence_20260929_v1/manifest.json')['records']
    rows = [row for row in rows if row['key'] in args.keys]
    args.output.mkdir(exist_ok=False, parents=True)
    write_json(args.output / 'manifest.json', dict(records=rows, labels_used=False,
        objective='each original target independently; full original history; native QK/RMS/SwiGLU',
        scope='exposed development diagnostics; embedding gate sensitivity, not semantic truth'))
    model = load_model(MODEL)
    checkpoint_ffn(model)
    for row in rows:
        directory = args.output / row['key']
        directory.mkdir()
        trace_answer(model, row, directory)
    write_json(args.output / 'complete.json', dict(status='complete', answers=len(rows),
        targets=sum(len(row['response']['answer_ids']) for row in rows)))


if __name__ == '__main__':
    main()
