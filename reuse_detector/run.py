"""One foreground pipeline: capture -> unlabeled reference -> score -> optional evaluation."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from .core import Config, fit_reference, score_trace, source_splits


DEFAULT_POPULATION = '../reanchor/outputs/ragtruth_population_20260912'
CAPTURE_SCHEMA = 'local-reuse-capture-v1'
SCORE_SCHEMA = 'local-reuse-unsupervised-v1'


def write_json(path, value):
    path = Path(path); partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def save_npz(path, **values):
    path = Path(path); temp = path.with_suffix('.partial')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def single_writer(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another process is writing this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_roster(population, tasks='all', generators='llama-2-7b-chat', limit=None):
    """Uses the existing label-free inputs.jsonl; never opens response annotations."""
    rows = []
    with (Path(population) / 'inputs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if tasks != 'all' and r['task'] not in tasks.split(','):
                continue
            if generators != 'all' and r['generator'] not in generators.split(','):
                continue
            keys = ('id', 'source_id', 'task', 'generator', 'official_split', 'prompt_length',
                    'token_ids', 'source_mask', 'offsets', 'response_sha256')
            row = {k: r[k] for k in keys}
            row['id'], row['source_id'] = str(row['id']), str(row['source_id'])
            if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
                raise ValueError('unsafe response ID')
            if len(row['token_ids']) - row['prompt_length'] != len(row['offsets']):
                raise ValueError('unaligned original response token IDs')
            rows.append(row)
    if len({r['id'] for r in rows}) != len(rows) or not rows:
        raise ValueError('empty/duplicate selected roster')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        rows = rows[:limit]
    return rows


def capture(rows, population, out, args):
    """Resume at sample boundaries. No cached scalar features invent missing edges."""
    settings = json.loads((population / 'settings.json').read_text())
    model_path = str(Path(args.model or settings['model']).resolve())
    expected = dict(schema=CAPTURE_SCHEMA, model=model_path, window=args.window,
                    roster_identity=digest(rows), query_chunk=args.query_chunk, dtype=args.dtype,
                    attention='sdpa_native_with_readonly_qk_replay', labels_read=False)
    cache = out / 'capture'; cache.mkdir(exist_ok=True)
    contract = cache / 'settings.json'
    if contract.exists():
        if json.loads(contract.read_text()) != expected:
            raise ValueError('capture input/model/window changed; use a different output (no silent resume)')
    else:
        write_json(contract, expected)
        write_json(cache / 'records.json', rows)
    pending = [r for r in rows if not (cache / (r['id'] + '.npz')).exists()]
    too_long = [(r['id'], len(r['token_ids']) - 1) for r in pending if len(r['token_ids']) - 1 > args.max_tokens]
    if too_long:
        raise ValueError(f'{len(too_long)} selected sequences exceed --max-tokens; no truncation. First: {too_long[:5]}')
    print(f'local edge cache: complete={len(rows)-len(pending)}/{len(rows)}, pending={len(pending)}', flush=True)
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .capture import capture_response
        torch.set_num_threads(args.cpu_threads)
        if args.device.startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
        dtype = getattr(torch, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                torch_dtype=dtype, attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        for i, row in enumerate(pending):
            if len(row['token_ids']) - 1 > model.config.max_position_embeddings:
                raise ValueError('input exceeds observer context; no truncation')
            begin = time.monotonic()
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(model.get_input_embeddings().weight.device)
            arrays = capture_response(model, row, window=args.window, chunk=args.query_chunk,
                                      special_ids=tokenizer.all_special_ids)
            save_npz(cache / (row['id'] + '.npz'), **arrays,
                     input_identity=np.asarray(digest(row)))
            write_json(cache / 'progress.json', dict(completed=len(rows)-len(pending)+i+1, planned=len(rows),
                       current=row['id'], seconds=time.monotonic()-begin))
            print(f"capture {i+1}/{len(pending)} {row['official_split']}/{row['task']}/{row['id']} "
                  f"tokens={len(row['offsets'])} seconds={time.monotonic()-begin:.2f} "
                  f"native_replay_error={arrays['attention_replay_relative_error'].max():.3g}", flush=True)
            del arrays
            gc.collect()
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    write_json(cache / 'complete.json', dict(complete=True, responses=len(rows), schema=CAPTURE_SCHEMA))


def load_capture(cache, row):
    with np.load(cache / (row['id'] + '.npz'), allow_pickle=False) as data:
        if str(data['input_identity']) != digest(row) or not np.array_equal(data['offsets'], row['offsets']):
            raise ValueError('captured response identity changed: ' + row['id'])
        return {k: data[k].copy() for k in data.files}


def score(rows, out, config, alarm_budget=.05):
    cache, scores_dir = out / 'capture', out / 'scores'
    if not (cache / 'complete.json').exists():
        raise ValueError('finish selected capture roster before scoring; no completed-only selection drift')
    if json.loads((cache / 'records.json').read_text()) != rows:
        raise ValueError('selected records differ from edge cache')
    splits = source_splits(rows, config.seed)
    expected = dict(schema=SCORE_SCHEMA, config=asdict(config), splits=splits,
                    roster_identity=digest(rows), alarm_budget=alarm_budget, labels_used=False,
                    probability_claim=False, supervised_weights_loaded=False)
    scores_dir.mkdir(exist_ok=True)
    if (scores_dir / 'settings.json').exists():
        if json.loads((scores_dir / 'settings.json').read_text()) != expected:
            raise ValueError('score settings changed; preserve old scores and use a new output directory')
    else:
        write_json(scores_dir / 'settings.json', expected)
    reference_path = scores_dir / 'reference.json'
    refs = [r for r in rows if splits[r['id']] == 'reference']
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        if reference['reference_ids'] != [r['id'] for r in refs] or reference['config'] != asdict(config):
            raise ValueError('reference identity mismatch')
    else:
        def read_entropy(r):
            with np.load(cache / (r['id'] + '.npz'), allow_pickle=False) as a:
                return a['entropy'].copy()
        reference = fit_reference(refs, read_entropy, config)
        write_json(reference_path, reference)
    calibration = []
    for i, r in enumerate(rows):
        file = scores_dir / (r['id'] + '.npz')
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                if str(saved['input_identity']) != digest(r):
                    raise ValueError('score identity mismatch')
                scored = {k: saved[k].copy() for k in saved.files if k.startswith('score__')}
        else:
            arrays = load_capture(cache, r)
            result, details = score_trace(arrays, r, reference)
            scored = {'score__' + k: v.astype(np.float32) for k, v in result.items()}
            save_npz(file, **scored, **details, offsets=r['offsets'], input_identity=np.asarray(digest(r)))
        if splits[r['id']] == 'calibration':
            calibration.append((r, {k[7:]: float(v.max()) for k, v in scored.items()}))
        if i % 25 == 0 or i + 1 == len(rows):
            print(f'score {i+1}/{len(rows)}; no labels; head-resolved local recurrence', flush=True)
    if not calibration:
        raise ValueError('no unlabeled calibration responses')
    # Calibrate anomaly budgets from per-SOURCE answer maxima, without knowing correctness.
    groups = {}
    for r, values in calibration:
        key = r['task'] + '|' + r['generator']
        group = groups.setdefault(key, {})
        sid = r['source_id']
        group[sid] = {k: max(v, group.get(sid, {}).get(k, -np.inf)) for k, v in values.items()}
    thresholds = {}
    for group, sources in groups.items():
        names = list(next(iter(sources.values())))
        thresholds[group] = {k: float(np.quantile([v[k] for v in sources.values()], 1-alarm_budget, method='higher')) for k in names}
    write_json(scores_dir / 'thresholds.json', dict(values=thresholds, budget=alarm_budget,
               calibration_unit='source maximum over its answers/tokens', labels_used=False,
               interpretation='unlabeled mixture anomaly budget; NOT normal-response FPR control'))
    # Evaluation is a separate phase and is the ONLY code allowed to open annotations.
    write_json(scores_dir / 'prediction_freeze.json', dict(complete=True, responses=len(rows),
               reference_ids=reference['reference_ids'], labels_read=False, settings=expected))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population', default=DEFAULT_POPULATION)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=('all', 'capture', 'score', 'evaluate'), default='all')
    p.add_argument('--model', help='local observer path; default population/settings.json model')
    p.add_argument('--tasks', default='all', help='all or comma-separated QA,Summary,Data2txt')
    p.add_argument('--generators', default='llama-2-7b-chat', help='all or exact comma-separated original model IDs')
    p.add_argument('--limit', type=int, help='capture-only software smoke subset, never a benchmark')
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--seed-tail', type=float, default=.05)
    p.add_argument('--survival', type=float, default=.9)
    p.add_argument('--channel-quantile', type=float, default=.9)
    p.add_argument('--query-chunk', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('bfloat16', 'float32', 'float16'), default='bfloat16')
    p.add_argument('--alarm-budget', type=float, default=.05)
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--annotations', help='evaluation only; defaults to the parent dataset/response.jsonl')
    p.add_argument('--resume', action='store_true')
    return p


def run(args):
    config = Config(window=args.window, seed_tail=args.seed_tail, survival=args.survival,
                    channel_quantile=args.channel_quantile)
    if not 0 < args.alarm_budget < 1 or args.bootstrap < 0 or args.cpu_threads < 1:
        raise ValueError('invalid budget, bootstrap, or CPU threads')
    if args.limit is not None and args.phase != 'capture':
        raise ValueError('--limit only supports capture smoke; use a full selected roster for scoring')
    population, out = Path(args.population), Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError('use --resume for identical settings or a fresh output directory')
    rows = load_roster(population, args.tasks, args.generators, args.limit)
    with single_writer(out):
        if args.phase in ('all', 'capture'):
            capture(rows, population, out, args)
        if args.phase in ('all', 'score'):
            score(rows, out, config, args.alarm_budget)
        if args.phase in ('all', 'evaluate'):
            if not (out / 'scores/prediction_freeze.json').exists():
                raise ValueError('freeze label-free predictions before evaluation')
            annotations = args.annotations
            if not annotations:
                parent = json.loads((population / 'settings.json').read_text())
                annotations = str(Path(parent['dataset']) / 'response.jsonl')
            from .evaluation import evaluate
            evaluate(rows, out, annotations, args.bootstrap)


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
