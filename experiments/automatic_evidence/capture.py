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
            heads = model.config.num_attention_heads
            current = inputs[0][0, prompt_length - 1:].detach().reshape(-1, heads, model.config.head_dim)
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


@torch.no_grad()
def measure_intervention(model, tokens, prompt_length, targets, baseline, mask):
    """Reuse the same native measurement for key masks and source text edits."""
    hidden, messages = forward(model, tokens, prompt_length, mask, baseline['messages'])
    changed = distributions(model, hidden)
    logp, divergence = distribution_effect(baseline['distribution'], changed, targets)
    denominator = hidden.norm(dim=-1) + baseline['hidden'].norm(dim=-1)
    change = (hidden-baseline['hidden']).norm(dim=-1) / denominator.clamp_min(1e-30)
    return dict(logp=logp, js=divergence, hidden_change=change.cpu().numpy(),
                message_change=np.asarray(messages, dtype=np.float32))


def edited_tokens(row, candidate, variant, device):
    start, stop = candidate['keys']
    prompt = row['prompt'][:start] + candidate[variant+'_ids'] + row['prompt'][stop:]
    tokens = torch.tensor([prompt+row['response']['answer_ids'][:-1]], device=device)
    return tokens, len(prompt)


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
    measurement_baseline = dict(hidden=original_hidden, messages=original_messages, distribution=original_distribution)
    for index, group in enumerate(groups):
        measured = measure_intervention(model, tokens, len(prompt), targets, measurement_baseline,
                                        source_mask(tokens, group['keys']))
        logp.append(measured['logp'])
        divergence.append(measured['js'])
        hidden_changes.append(measured['hidden_change'])
        message_changes.append(measured['message_change'])
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


def pack_measurements(measurements, targets, model):
    arrays = {}
    for name in ('logp', 'js', 'hidden_change'):
        arrays[name] = np.asarray([item[name] for item in measurements], dtype=np.float32).reshape(-1, targets).T
    shape = (-1, model.config.num_hidden_layers, model.config.num_attention_heads, targets)
    arrays['message_change'] = np.asarray([item['message_change'] for item in measurements], dtype=np.float32).reshape(shape)
    return arrays


def matching_edits(candidates, previous, variant):
    """Cache only exactly identical source addresses and replacement token IDs."""
    def identity(candidate):
        return (candidate['source'], tuple(candidate['keys']), tuple(candidate[variant+'_ids']))
    lookup = {identity(candidate): index for index, candidate in enumerate(previous)}
    return [lookup.get(identity(candidate), -1) for candidate in candidates]


def cached_measurement(arrays, variant, index):
    item = {name: arrays[variant+'_'+name][:, index] for name in ('logp', 'js', 'hidden_change')}
    item['message_change'] = arrays[variant+'_message_change'][index]
    return item


def reusable_measurements(row, candidates, reuse_path):
    if reuse_path is None:
        return None, {name: [-1]*len(candidates['edits']) for name in ('flip', 'equivalent')}
    manifest = read_json(reuse_path/'manifest.json')
    original = next(item for item in manifest['records'] if item['key'] == row['key'])
    assert original['prompt'] == row['prompt']
    assert original['response']['answer_ids'] == row['response']['answer_ids']
    previous = read_json(reuse_path/row['key']/'candidates.json')
    assert previous['repeated_sets'] == candidates['repeated_sets']
    read_json(reuse_path/row['key']/'capture_complete.json')
    with np.load(reuse_path/row['key']/'relation_effects.npz') as saved:
        arrays = {name: saved[name] for name in saved.files}
    matches = {name: matching_edits(candidates['edits'], previous['edits'], name)
               for name in ('flip', 'equivalent')}
    return arrays, matches


@torch.no_grad()
def capture_relations(model, row, input_path, output, reuse_path=None):
    directory = output / row['key']
    candidates = read_json(directory / 'candidates.json')
    cached, matches = reusable_measurements(row, candidates, reuse_path)
    fresh_edits = sum(index < 0 for indices in matches.values() for index in indices)
    groups = read_json(input_path / row['key'] / 'sources.json')
    prompt_length = len(row['prompt'])
    answer = row['response']['answer_ids']
    targets = torch.tensor(answer, device=model.device)
    tokens = torch.tensor([row['prompt']+answer[:-1]], device=model.device)
    needs_forward = cached is None or fresh_edits > 0
    if needs_forward:
        hidden, messages = forward(model, tokens, prompt_length, torch.ones_like(tokens))
        baseline = dict(hidden=hidden, messages=messages, distribution=distributions(model, hidden))
        original_logp = baseline['distribution'].gather(1, targets[:, None])[:, 0].cpu().numpy()
    else:
        original_logp = cached['original_logp']
    with np.load(input_path / row['key'] / 'effects.npz') as previous:
        np.testing.assert_array_equal(previous['token_ids'], answer)
        error = float(np.max(np.abs(previous['original_logp']-original_logp)))
        assert error < 2e-4, (row['key'], error)
    arrays = dict(original_logp=original_logp, token_ids=answer)
    started = perf_counter()
    for variant in ('flip', 'equivalent'):
        measured = []
        for candidate, previous_index in zip(candidates['edits'], matches[variant]):
            if previous_index >= 0:
                measured.append(cached_measurement(cached, variant, previous_index))
            else:
                changed, length = edited_tokens(row, candidate, variant, model.device)
                measured.append(measure_intervention(model, changed, length, targets, baseline, torch.ones_like(changed)))
        arrays.update({variant+'_'+name: value for name, value in pack_measurements(measured, len(answer), model).items()})
        print(dict(key=row['key'], variant=variant, candidates=len(measured), seconds=perf_counter()-started), flush=True)
    if cached is None:
        repeat_measurements(model, tokens, prompt_length, targets, baseline, groups, candidates, arrays)
    else:
        arrays.update({name: value for name, value in cached.items() if name.startswith('repeat')})
    if needs_forward:
        sham = measure_intervention(model, tokens, prompt_length, targets, baseline, receiver_mask(tokens, prompt_length-1, []))
        sham_error = float(np.max(np.abs(sham['logp']-original_logp)))
    else:
        sham_error = float(cached['sham_error'])
    assert sham_error < 2e-4, sham_error
    arrays['sham_error'] = sham_error
    arrays['baseline_error'] = error
    np.savez_compressed(directory / 'relation_effects.npz', **arrays)
    write_json(directory / 'capture_complete.json', dict(status='complete', labels_read=False,
        forward_calls=2*int(needs_forward)+fresh_edits+(2*len(candidates['repeated_sets']) if cached is None else 0),
        reused_edits=2*len(candidates['edits'])-fresh_edits, reused_sets=cached is not None,
        fresh_baseline_and_sham=needs_forward, reuse_path=str(reuse_path) if reuse_path is not None else None,
        candidates=len(candidates['edits']), repeated_sets=candidates['repeated_sets'],
        seconds=perf_counter()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        baseline_logp_error=error, sham_logp_error=sham_error))


def repeat_measurements(model, tokens, prompt_length, targets, baseline, groups, candidates, arrays):
    all_keys = np.array([key for group in groups for key in group['keys']])
    measurements = {'repeat': [], 'repeat_control': []}
    rng = np.random.default_rng(42)
    for group in candidates['repeated_sets']:
        keys = sorted({key for source in group['sources'] for key in groups[source]['keys']})
        other_keys = np.setdiff1d(all_keys, keys)
        assert len(other_keys) >= len(keys), 'Insufficient disjoint keys for size-matched control'
        control = sorted(rng.choice(other_keys, len(keys), replace=False).tolist())
        group.update(keys=keys, control_keys=control)
        for name, selected in (('repeat', keys), ('repeat_control', control)):
            measurements[name].append(measure_intervention(model, tokens, prompt_length, targets,
                                                           baseline, source_mask(tokens, selected)))
    for name, values in measurements.items():
        packed = pack_measurements(values, len(targets), model)
        arrays.update({name+'_'+field: value for field, value in packed.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keys', nargs='*')
    args = parser.parse_args()
    read_json(args.output / 'prepared.json')
    manifest = read_json(args.output / 'manifest.json')
    protocol = read_json(args.output / 'protocol.json')
    records = manifest['records']
    if args.keys:
        records = [row for row in records if row['key'] in args.keys]
    model = load_model(manifest['model'])
    for row in records:
        if (args.output / row['key'] / 'capture_complete.json').exists():
            raise FileExistsError('Capture already completed: ' + row['key'])
        if protocol.get('mode') == 'relation':
            reuse = Path(protocol['reuse_effects']) if 'reuse_effects' in protocol else None
            capture_relations(model, row, Path(protocol['input']), args.output, reuse)
        else:
            capture_record(model, row, args.output)
        print('CAPTURED', row['key'], flush=True)
    write_json(args.output / 'capture_complete.json', dict(keys=[row['key'] for row in records],
        labels_read=False, status='complete'))


if __name__ == '__main__':
    main()
