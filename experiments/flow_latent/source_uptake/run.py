"""Frozen automatic token transfer gate; scoring never opens annotations.

Every target is predicted at P+t-1. Bulk causal prefill shares computation,
while the eight-row input ends strictly before the target token is input.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection import capture_projection_inputs, observed_attention
from ..ordered_source_transport.observe import capture_window, SITES, WINDOW
from ..ordered_source_transport.readout import row_permutations
from ..provenance_joint_state.matrix_audit import sha256, write_json

MODEL = Path('../../models/Meta-Llama-3.1-8B-Instruct').resolve()
CHECKPOINTS = Path('outputs/ordered_source_transport_mixed_fits_20261008').resolve()
VARIANTS = ('ordered', 'mean', 'single', 'shuffled', 'drop_source', 'linear')
SEEDS = (42, 123, 2026)


@torch.no_grad()
def capture_answer(adapter, prompt, answer, mask, query_chunk=64):
    """Return T+7 complete native rows, physical source heads, and logits facts."""
    model = adapter.native
    tokens = prompt + answer[:-1]
    start = len(prompt) - WINDOW
    if start < 0 or not answer:
        raise ValueError('Prompt must supply warm-up rows and answer must be nonempty')
    ids = adapter.input_ids(tokens)
    with capture_window(model, start) as states:
        with capture_projection_inputs(model, start + 1) as projections:
            hidden = model.model(ids, use_cache=False).last_hidden_state[0].cpu()
    positions = torch.arange(len(tokens), device=model.device)
    cosine, sine = model.model.rotary_emb(model.model.embed_tokens(ids), positions[None])
    source = torch.tensor(mask + [False] * (len(answer) - 1), device=model.device, dtype=torch.bool)
    total = len(tokens) - start
    heads = np.empty((total, len(states), 32, 128), dtype=np.float32)
    mass = np.empty((total, len(states), 32), dtype=np.float32)
    reconstruction = []
    for index, record in enumerate(projections):
        writes = []
        for begin in range(0, total, query_chunk):
            end = min(begin + query_chunk, total)
            selected = dict(record, query=record['query'][begin:end])
            query_positions = positions[start + begin:start + end]
            attention, values = observed_attention(adapter, index, selected, query_positions, cosine[0], sine[0])
            complete = torch.einsum('whk,khd->whd', attention, values).flatten(1)
            reconstruction.append(float((complete.cpu() - record['head'][begin:end]).abs().max()))
            message = torch.einsum('whk,khd->whd', attention[:, :, source], values[source])
            writes.append(adapter.layers[index].self_attn.o_proj(message.flatten(1)).cpu())
            heads[begin:end, index] = message.cpu().numpy()
            mass[begin:end, index] = attention[:, :, source].sum(-1).cpu().numpy()
        states[index]['source_write'] = torch.cat(writes)
    raw = torch.stack([torch.stack([row[name] for name in SITES], dim=1) for row in states], dim=1)
    equation = max(float(((row['residual_before'] + row['attention_write']) + row['mlp_write'] -
                         row['after']).abs().max()) for row in states)
    predicted_hidden = hidden[len(prompt) - 1:]
    top_ids, top_logp, actual_logp, entropy, candidate_vectors = [], [], [], [], []
    for begin in range(0, len(answer), 16):
        logits = model.lm_head(predicted_hidden[begin:begin + 16].to(model.device)).float()
        logp = logits.log_softmax(-1)
        values, candidates = logp.topk(32, dim=-1)
        actual = torch.tensor(answer[begin:begin + 16], device=model.device)
        top_ids.append(candidates.cpu().numpy())
        top_logp.append(values.cpu().numpy())
        actual_logp.append(logp.gather(1, actual[:, None])[:, 0].cpu().numpy())
        entropy.append(-(logp.exp() * logp).sum(-1).cpu().numpy())
        combined = torch.cat([candidates, actual[:, None]], dim=1)
        embeddings = model.lm_head.weight[combined].float()
        candidate_vectors.append(torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy())
    native = dict(candidate_ids=np.concatenate(top_ids), candidate_logp=np.concatenate(top_logp),
                  actual_ids=np.asarray(answer, dtype=np.int64), actual_logp=np.concatenate(actual_logp),
                  entropy=np.concatenate(entropy))
    native['greedy_ids'] = native['candidate_ids'][:, 0]
    native['surprisal'] = -native['actual_logp']
    native['actual_in_topk'] = (native['candidate_ids'] == native['actual_ids'][:, None]).any(-1)
    return raw.numpy(), heads, mass, native, np.concatenate(candidate_vectors), predicted_hidden.numpy(), {
        'head_reconstruction': max(reconstruction), 'state_equation': equation,
        'input_tokens': len(tokens), 'first_prediction_position': len(prompt) - 1,
        'state_start_position': start, 'prediction_rows': len(answer)}


def checkpoint_paths():
    return {f'{variant}_seed{seed}': CHECKPOINTS / f'{variant}_seed{seed}' / 'model.pt'
            for variant in VARIANTS for seed in SEEDS}


def freeze_outputs(output, roster_path, protocol, files):
    """Bind metric settings/code before the first annotation-file access."""
    from .evaluate import metric_code_paths
    evaluation = dict(primary_key='ordered_seed42_gap', fixed_replication_seeds=[123, 2026],
        threshold=0, bootstrap_draws=2000, bootstrap_seed=73, bootstrap_unit='source',
        metric_code_hashes={str(p): sha256(p) for p in metric_code_paths()},
        natural_annotations_opened=False,
        interpretation='Exploratory source-self-supervised pre-token observer, not prevention')
    evaluation_path = output / 'EVALUATION_PROTOCOL.json'
    write_json(evaluation_path, evaluation)
    bound = {str(p.relative_to(output)): sha256(p) for p in files + [evaluation_path]}
    bound.update(evaluation['metric_code_hashes'])
    for name in ('REUSE.json', 'PILOT.json', 'LEDGER.executed.json', 'FREEZE.executed.json'):
        if (output / name).exists():
            bound[name] = sha256(output / name)
    freeze = dict(status='COMPLETE', files=bound,
        roster=dict(path=str(roster_path), sha256=sha256(roster_path)),
        checkpoints=protocol['checkpoints'], primary_key='ordered_seed42_gap',
        annotation_file_opened=False, natural_label_fits=0)
    write_json(output / 'FREEZE.json', freeze)
    return freeze


def score_answer(raw, native, vectors, checkpoints):
    from .scoring import score_windows
    total = len(native['actual_ids'])
    permutations = row_permutations(total, WINDOW, seed=73)
    candidates = np.concatenate([native['candidate_ids'], native['actual_ids'][:, None]], axis=1)
    logp = np.concatenate([native['candidate_logp'], native['actual_logp'][:, None]], axis=1)
    scores = dict(native)
    for name, checkpoint in checkpoints.items():
        variant = checkpoint['variant']
        result = []
        for begin in range(0, total, 4):
            end = min(begin + 4, total)
            # This copies only one small batch, never eight duplicated full answers.
            consecutive = raw[begin:end + WINDOW - 1]
            result.append(score_windows(torch.from_numpy(consecutive).to('cuda'),
                torch.from_numpy(vectors[begin:end]).to('cuda'),
                torch.from_numpy(candidates[begin:end]).to('cuda'),
                torch.from_numpy(native['actual_ids'][begin:end]).to('cuda'),
                torch.from_numpy(logp[begin:end]).to('cuda'), checkpoint, variant,
                permutations=permutations[begin:end].to('cuda')))
        for key in ('gap', 'logp_gap', 'native_gap', 'rival_ids', 'logp_rival_ids', 'native_rival_ids'):
            scores[f'{name}_{key}'] = np.concatenate([row[key].detach().cpu().numpy() for row in result])
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Nonfinite output; preserve failure and stop')
    return scores


def collect(args):
    args.roster = args.roster.resolve()
    args.output = args.output.resolve()
    roster = json.loads(args.roster.read_text())
    records = roster['natural']['records']
    if not 1 <= args.limit <= len(records):
        raise ValueError('Limit exceeds fixed roster')
    args.output.mkdir(exist_ok=True)
    protocol_path = args.output / 'PROTOCOL.json'
    dependencies = [Path(__file__), Path(__file__).with_name('scoring.py'),
                    Path('experiments/token_backtrace/grounded_projection.py'),
                    Path('experiments/flow_latent/ordered_source_transport/observe.py'),
                    Path('experiments/flow_latent/ordered_source_transport/readout.py'),
                    Path('experiments/decision_risk_flow/precision.py'),
                    Path('experiments/decision_risk_flow/run.py')]
    paths = checkpoint_paths()
    protocol = dict(schema='source_uptake_automatic_v1', roster=str(args.roster),
        roster_sha256=sha256(args.roster), fixed_answers=len(records), fixed_tokens=roster['natural']['tokens'],
        model=str(MODEL), K=32, W=WINDOW, sites=SITES, seeds=SEEDS, variants=VARIANTS,
        primary='max native-top32 other-token g - g(actual)',
        auxiliary='max native-top32 other-token (g+logp) - (g+logp)(actual), coefficient=1',
        native_greedy='max native-top32 other-token g - g(native_greedy); unlabelled alternatives',
        supervision='frozen source-program self-supervised; zero natural label fits',
        interpretation=roster['natural_interpretation'],
        annotation_file_opened_by_scorer=False, planned_forwards=len(records) + 4,
        primary_checkpoint='ordered_seed42', diagnostic_threshold=0,
        code_hashes={str(p.resolve()): sha256(p) for p in dependencies},
        checkpoints={str(p): sha256(p) for p in paths.values()})
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Resume protocol changed')
    else:
        write_json(protocol_path, protocol)
        snapshot = args.output / 'executed_code'
        snapshot.mkdir()
        for p in dependencies:
            destination = snapshot / p
            if p.is_absolute():
                destination = snapshot / p.relative_to(Path.cwd())
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoints = {name: torch.load(p, map_location='cpu', weights_only=True) for name, p in paths.items()}
    for checkpoint in checkpoints.values():
        for key in ('mean', 'scale'):
            checkpoint[key] = checkpoint[key].to('cuda')
        checkpoint['state_dict'] = {k: v.to('cuda') for k, v in checkpoint['state_dict'].items()}
    started = time.time()
    adapter = ModelAdapter(load_model(MODEL))
    ledger_path = args.output / 'LEDGER.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        full_forwards=0, canary_forwards=0, natural_detector_fits=0, source_program_fits=0, answers=[])
    ledger['new_full_forwards'] = ledger['full_forwards'] - len(ledger.get('reused_answers', []))
    ledger['new_canary_forwards'] = ledger['canary_forwards'] - ledger.get('reused_canary_forwards', 0)
    with torch.no_grad():
        for index, record in enumerate(records[:args.limit]):
            directory = args.output / 'answers' / record['id']
            if (directory / 'data.json').exists():
                continue
            if directory.exists():
                raise ValueError(f'Incomplete existing answer {directory}; preserve and inspect')
            source_path, response_path = Path(record['source_cache_path']), Path(record['response_cache_path'])
            source, response = json.loads(source_path.read_text()), json.loads(response_path.read_text())
            prompt, answer = source['prompt_with_source'], response['answer_ids']
            raw, heads, mass, native, vectors, hidden, errors = capture_answer(
                adapter, prompt, answer, source['source_mask'])
            ledger['full_forwards'] += 1
            ledger['new_full_forwards'] += 1
            if errors['head_reconstruction'] >= 2e-4 or errors['state_equation'] != 0:
                raise ValueError(f'Native decomposition failed: {errors}')
            canaries = []
            # Four actual prefix-only observations in the first two roster answers.
            if index < 2:
                for target in sorted({0, len(answer) // 2}):
                    observed = capture_answer(adapter, prompt, answer[:target + 1], source['source_mask'])
                    ledger['canary_forwards'] += 1
                    ledger['new_canary_forwards'] += 1
                    canary_raw, _, _, canary_native, _, canary_hidden, _ = observed
                    error = float(np.abs(canary_raw - raw[:target + WINDOW]).max())
                    hidden_error = float(np.abs(canary_hidden[-1] - hidden[target]).max())
                    greedy_equal = int(canary_native['greedy_ids'][-1]) == int(native['greedy_ids'][target])
                    canaries.append(dict(target=target, state_max_error=error,
                                         hidden_max_error=hidden_error, greedy_equal=greedy_equal))
                    if error >= 3e-4 or hidden_error >= 3e-4 or not greedy_equal:
                        raise ValueError(f'Prefix causality canary failed: {canaries[-1]}')
            scores = score_answer(raw, native, vectors, checkpoints)
            prior_path = response_path.with_name('scores.npz')
            with np.load(prior_path) as prior:
                if not np.array_equal(prior['token_id'], answer):
                    raise ValueError('Prior scalar token alignment failed')
                for key in ('source_pair_unit_mean', 'source_local_unit_mean', 'raw_route'):
                    scores[f'prior_{key}'] = prior[key]
            directory.mkdir(parents=True)
            for name, values in [('states', raw), ('source_heads', heads), ('source_mass', mass),
                                 ('candidate_vectors', vectors), ('prediction_hidden', hidden)]:
                np.save(directory / f'{name}.npy', values)
            np.savez(directory / 'scores.npz', **scores)
            data = dict(response_id=record['id'], source_id=record['source_id'], generator=record['generator'],
                text=response['text'], offsets=response['offsets'], special_ids=response['special_ids'],
                prompt_tokens=len(prompt), answer_tokens=len(answer), errors=errors, canaries=canaries,
                input_hashes={str(source_path): sha256(source_path), str(response_path): sha256(response_path),
                              str(prior_path): sha256(prior_path)},
                raw_hashes={p.name: sha256(p) for p in directory.glob('*.npy')})
            write_json(directory / 'data.json', data)
            ledger['answers'].append(record['id'])
            ledger['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
            ledger['last_call_seconds'] = time.time() - started
            write_json(ledger_path, ledger)
            print(json.dumps(dict(answer=index + 1, total=args.limit, id=record['id'], tokens=len(answer),
                                  errors=errors, canaries=canaries)), flush=True)
            del raw, heads, mass, native, vectors, hidden, scores
            gc.collect()
    del adapter, checkpoints
    torch.cuda.empty_cache()
    if args.limit == len(records):
        if set(ledger['answers']) != {row['id'] for row in records}:
            raise ValueError('Roster incomplete')
        files = [protocol_path, ledger_path] + list((args.output / 'answers').glob('*/scores.npz')) + list(
            (args.output / 'answers').glob('*/data.json'))
        freeze_outputs(args.output, args.roster, protocol, files)
    print(json.dumps(ledger), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=8)
    collect(parser.parse_args())


if __name__ == '__main__':
    main()
