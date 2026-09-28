"""Run source-excluded case regression with native Fisher response topology."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'teaching/state_audit/src'))

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .data import inputs, prepare, read_json, write_json
from .native import prefill, confidence
from .precision import enable_fp32_execution


def load_model(path):
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval().requires_grad_(False)
    return enable_fp32_execution(model)


def capture_static(args):
    model = load_model(args.model)
    records = read_json(args.output / 'manifest.json')['records']
    started = perf_counter()
    for index, record in enumerate(records):
        path = args.output / 'static' / f"{record['key']}.npz"
        if path.exists():
            continue
        prompt, response = inputs(record)
        answer = response['answer_ids']
        cache, states, normalized = prefill(model, prompt, answer)
        observed, alternatives = confidence(model, normalized, answer, len(prompt), response['special_ids'])
        with torch.no_grad():
            embedding = model.model.embed_tokens(torch.tensor(answer, device=model.device))
        offsets = np.asarray(response['offsets'])
        valid = (offsets[:, 1] > offsets[:, 0]) & ~np.isin(answer, response['special_ids'])
        np.savez_compressed(path, states=states.to(torch.float16).cpu().numpy(),
            embedding=embedding.to(torch.float16).cpu().numpy(), confidence=observed.numpy(),
            alternative=alternatives.numpy(), token_ids=answer, valid=valid)
        del cache, states, normalized, embedding
        print(dict(stage='static', done=index+1, total=len(records), id=record['key'],
                   tokens=len(answer), seconds=round(perf_counter()-started, 2)), flush=True)
    write_json(args.output / 'static_complete.json', dict(status='complete', answers=len(records),
        seconds=perf_counter()-started, cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', required=True, choices=('all', 'prepare', 'static', 'fit', 'responses', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    parser.add_argument('--model', type=Path, default=Path('/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct'))
    parser.add_argument('--fit-sources', type=int, default=12)
    parser.add_argument('--dev-sources', type=int, default=8)
    parser.add_argument('--test-sources', type=int, default=0)
    parser.add_argument('--query-batch', type=int, default=16)
    parser.add_argument('--sketch-rank', type=int, default=8)
    args = parser.parse_args(argv)
    if args.phase == 'all':
        # Separate phases keep large model allocations out of CPU probe fitting.
        import subprocess
        for stage in ('prepare', 'static', 'fit', 'responses', 'evaluate'):
            arguments = ['--phase', stage]
            for name, value in vars(args).items():
                if name != 'phase':
                    arguments.extend(['--' + name.replace('_', '-'), str(value)])
            subprocess.run([sys.executable, '-m', 'experiments.decision_risk_flow.run', *arguments], check=True)
        subprocess.run([sys.executable, '-m', 'experiments.decision_risk_flow.verify',
                        '--output', str(args.output)], check=True)
    elif args.phase == 'prepare':
        if args.test_sources:
            parser.error('This pilot evaluates the eight regression cases only; use --test-sources 0.')
        records = prepare(args.packs, args.output, args.fit_sources, args.dev_sources, args.test_sources)
        write_json(args.output / 'protocol.json', dict(model=str(args.model), checkpoints=[16,24,32],
            method='native_fisher_window_topology_v1', folds=5, source_annotations=False,
            candidate='actual token versus baseline strongest non-actual token',
            risk_derivative='not implemented in this pilot', query_batch=args.query_batch,
            gate='post-softmax, no renormalization', layers='all 32, two bands', window=16,
            sketch_rank=args.sketch_rank, graph='cosine > .5; top 16 energy nodes per structural group',
            groups=['prompt', 'recent16', 'remote', 'special', 'mlp'],
            kernel_supervision=False, readout_supervision=True, regularization_grid=[.01, .1, 1.],
            selection='source-equal development BCE',
            thresholds='task-specific dev normal token / normal-answer max 95th percentile; strict greater-than',
            execution='FP32 forward/input gradients; unchanged frozen BF16 weights',
            native_replay_tolerance=dict(max_logit_absolute=.005, raw_state_relative_l2=.0005),
            main_readout='source-OOF ordinary probes + confidence + native choice + Fisher kernel + graph',
            probe_training=dict(folds=5, models=['rank8', 'mlp9'], epochs=30,
                optimizer='AdamW lr .001 weight_decay .1', selection='dev BCE every 5 epochs'),
            scope=f'{len(records)} complete answers; eight exposed source-excluded regression cases',
            warning='fixed-past-KV current-query derivatives; sketch-limited rank; observer replay'))
        print('Prepared', len(records), 'answers; labels not used for selection.', flush=True)
    elif args.phase == 'static':
        capture_static(args)
    elif args.phase == 'fit':
        from .train import fit
        fit(args)
    elif args.phase == 'responses':
        from .responses import capture_responses
        capture_responses(args)
    else:
        from .evaluate import evaluate
        evaluate(args)


if __name__ == '__main__':
    main()
