"""Native source-span interventions, preserving positions and every answer token."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from .model import adjacent_change


@contextmanager
def message_capture(model, prompt_length, reference=None):
    """Observe pre-W_O head messages; changes are not semantic truth values."""
    messages = []
    handles = []
    def hook(index):
        def observe(module, inputs):
            current = inputs[0][0, prompt_length - 1:].detach().reshape(-1, 32, 128)
            if reference is None:
                messages.append(current.clone())
            else:
                baseline = reference[index]
                denominator = baseline.norm(dim=-1) + current.norm(dim=-1)
                change = (current - baseline).norm(dim=-1) / denominator.clamp_min(1e-30)
                messages.append(change.T.cpu().numpy())
        return observe
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook(index)))
    try:
        yield messages
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def forward(model, tokens, prompt_length, mask, reference=None):
    with message_capture(model, prompt_length, reference) as messages:
        output = model.model(input_ids=tokens, attention_mask=mask, use_cache=False)
    hidden = output.last_hidden_state[0, prompt_length - 1:]
    return hidden, messages


@torch.no_grad()
def distributions(model, hidden):
    # Chunk lm_head's temporary FP32 weights/activations without vocabulary truncation.
    return torch.cat([model.lm_head(block).float().log_softmax(-1)
                      for block in hidden.split(16)], dim=0)


def distribution_effect(reference, changed, targets):
    mixture = torch.logaddexp(reference, changed) - np.log(2)
    divergence = .5 * ((reference.exp() * (reference - mixture)).sum(-1)
                      + (changed.exp() * (changed - mixture)).sum(-1))
    logp = changed.gather(1, targets[:, None])[:, 0]
    return logp.cpu().numpy(), divergence.clamp_min(0).cpu().numpy()


def source_mask(tokens, keys):
    mask = torch.ones_like(tokens)
    mask[:, keys] = 0
    return mask


def receiver_mask(tokens, receiver, keys):
    length = tokens.shape[1]
    mask = torch.zeros((length, length), dtype=torch.float32, device=tokens.device)
    future = torch.ones_like(mask, dtype=torch.bool).triu(1)
    mask.masked_fill_(future, -torch.inf)
    mask[receiver, keys] = -torch.inf
    return mask[None, None]


def select_probes(prior, state_change, history):
    """Three automatic target choices, independent of gold and words."""
    candidates = [int(prior.max(-1).argmax()), int(state_change.argmax()), int(history.argmax())]
    return list(dict.fromkeys(candidates))


@torch.no_grad()
def direct_probes(model, row, groups, tokens, baseline, global_logp, prior, directory):
    hidden_change = adjacent_change(baseline['hidden'].cpu().numpy())
    targets = select_probes(prior['prior'], hidden_change, prior['history_fraction'])
    records = []
    for target in targets:
        selected = int(prior['prior'][target].argmax())
        keys = groups[selected]['keys']
        mask = receiver_mask(tokens, len(row['prompt']) - 1 + target, keys)
        hidden = model.model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[0, len(row['prompt']) - 1 + target]).float().log_softmax(-1)
        value = float(logits[row['response']['answer_ids'][target]])
        records.append(dict(target=target, source=selected, direct_logp=value,
            original_logp=float(baseline['logp'][target]), span_excluded_logp=float(global_logp[selected, target]),
            direct_effect=value-float(baseline['logp'][target]),
            full_effect=float(global_logp[selected, target]-baseline['logp'][target])))
    write_json(directory / 'direct_probes.json', dict(records=records,
        intervention='exclude source keys only at current prediction query in all layers; other queries unchanged',
        difference_is_not_identified_mediation=True))
    return len(records)


@torch.no_grad()
def capture_record(model, row, output):
    directory = output / row['key']
    groups = read_json(directory / 'sources.json')
    prompt, answer = row['prompt'], row['response']['answer_ids']
    tokens = torch.tensor([prompt + answer[:-1]], device=model.device)
    targets = torch.tensor(answer, device=model.device)
    original_hidden, original_messages = forward(model, tokens, len(prompt), torch.ones_like(tokens))
    original_distribution = distributions(model, original_hidden)
    original_logp = original_distribution.gather(1, targets[:, None])[:, 0].cpu().numpy()
    all_source_keys = [key for group in groups for key in group['keys']]
    empty_hidden, _ = forward(model, tokens, len(prompt), source_mask(tokens, all_source_keys), original_messages)
    empty_distribution = distributions(model, empty_hidden)
    empty_logp, empty_js = distribution_effect(original_distribution, empty_distribution, targets)
    foil = empty_distribution.argmax(-1).cpu().numpy()
    del empty_distribution, empty_hidden
    logp, divergence, hidden_changes, message_changes = [], [], [], []
    started = perf_counter()
    for index, group in enumerate(groups):
        hidden, changes = forward(model, tokens, len(prompt), source_mask(tokens, group['keys']), original_messages)
        changed_distribution = distributions(model, hidden)
        target_logp, js = distribution_effect(original_distribution, changed_distribution, targets)
        logp.append(target_logp)
        divergence.append(js)
        denominator = hidden.norm(dim=-1) + original_hidden.norm(dim=-1)
        hidden_changes.append(((hidden-original_hidden).norm(dim=-1) / denominator.clamp_min(1e-30)).cpu().numpy())
        message_changes.append(changes)
        del hidden, changed_distribution
        if index % 5 == 0:
            print(dict(key=row['key'], span=index+1, total=len(groups), seconds=round(perf_counter()-started, 1)), flush=True)
    baseline = dict(hidden=original_hidden, logp=original_logp)
    with np.load(directory / 'prior.npz') as prior:
        direct_count = direct_probes(model, row, groups, tokens, baseline, np.asarray(logp), prior, directory)
    # Same-world custom attention mask checks the current-query intervention implementation.
    sham_mask = receiver_mask(tokens, len(prompt)-1, [])
    sham = model.model(input_ids=tokens, attention_mask=sham_mask, use_cache=False).last_hidden_state[0, len(prompt)-1:]
    sham_distribution = distributions(model, sham)
    sham_logp, sham_js = distribution_effect(original_distribution, sham_distribution, targets)
    error = float(np.max(np.abs(sham_logp-original_logp)))
    assert error < 2e-4, error
    np.savez_compressed(directory / 'effects.npz', original_logp=original_logp, empty_logp=empty_logp,
        empty_js=empty_js, empty_foil=foil, masked_logp=np.asarray(logp).T,
        output_js=np.asarray(divergence).T, hidden_change=np.asarray(hidden_changes).T,
        message_change=np.asarray(message_changes, dtype=np.float32),
        original_hidden=original_hidden.cpu().numpy(), token_ids=answer,
        sham_logp_error=error, sham_js_max=float(sham_js.max()))
    write_json(directory / 'capture_complete.json', dict(status='complete', source_spans=len(groups),
        targets=len(answer), forward_calls=len(groups)+3+direct_count, seconds=perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), labels_read=False, sham_logp_error=error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='*')
    args = parser.parse_args()
    read_json(args.output / 'prepared.json')
    manifest = read_json(args.output / 'manifest.json')
    records = manifest['records']
    if args.keys:
        records = [row for row in records if row['key'] in args.keys]
    model = load_model(manifest['model'])
    for row in records:
        if (args.output / row['key'] / 'capture_complete.json').exists():
            raise FileExistsError('Capture already completed: ' + row['key'])
        capture_record(model, row, args.output)
        print('CAPTURED', row['key'], flush=True)
    write_json(args.output / 'capture_complete.json', dict(keys=[row['key'] for row in records],
        labels_read=False, status='complete'))


if __name__ == '__main__':
    main()
