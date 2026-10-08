"""Measure original/two cyclic source orders for all eight historical QA answers.

Fresh directory only. This entrypoint never imports a label evaluator, fits a
detector or changes the source+route/graph default. Run only after plan review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from experiments.decision_risk_flow.run import load_model
from .source_order import (CONDITIONS, compile_orders, complete_readout,
                           order_fields, verify_readout)


ROSTER = Path('outputs/conditional_adoption_20261008/complete_roster/INPUTS.json')
REFERENCE = Path('outputs/conditional_adoption_20261008/complete_capture')
CACHE = Path('outputs/native_support_ragtruth_all/source_first_v1')
OUTPUT = Path('outputs/source_order_equivariance_20261009')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_order_equivariance_20261009/EXPERIMENT_PLAN.md')
MODEL = Path('/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
EXPECTED = [('12219', '14353', 328, 269), ('12216', '14353', 328, 114),
            ('12297', '14366', 276, 156), ('12294', '14366', 276, 53),
            ('15604', '15521', 282, 111), ('15600', '15521', 282, 93),
            ('11907', '14300', 427, 206), ('11904', '14300', 427, 137)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def prepare_inputs(output):
    """Only prompt/source identity and already pure response/token caches."""
    document = json.loads(ROSTER.read_text())
    if Path(document['model']).resolve() != MODEL.resolve():
        raise ValueError('Frozen observer model differs')
    actual_roster = [(r['id'], r['source_id'], r['prompt_length'], r['token_count'])
                     for r in document['records']]
    if actual_roster != EXPECTED:
        raise ValueError('The whole frozen eight-answer roster differs')
    source_info = Path(document['dataset']) / 'source_info.jsonl'
    wanted = {row[1] for row in EXPECTED}
    prompts = {}
    for line in source_info.open():
        source = json.loads(line)
        identity = str(source['source_id'])
        if identity in wanted:
            if identity in prompts:
                raise ValueError('Duplicate source identity')
            prompts[identity] = source['prompt']
    if set(prompts) != wanted:
        raise ValueError('Four original source prompts not present')
    manifest_path = CACHE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    indexed = {r['id']: r for r in manifest['records']}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    dependencies = {str(path.resolve()): file_hash(path) for path in (ROSTER, source_info, manifest_path, PLAN)}
    truth = Path(document['dataset']) / 'response.jsonl'
    # Byte prehash only: the capture never parses natural evaluation annotations.
    dependencies[str(truth.resolve())] = file_hash(truth)
    maps, references = {}, {}
    for record in document['records']:
        identity = record['id']
        original = indexed[identity]
        if original['source_id'] != record['source_id']:
            raise ValueError('Source-first identity differs')
        source_path = CACHE / original['source_file']
        response_path = CACHE / original['directory'] / 'response.json'
        source = json.loads(source_path.read_text())
        response = json.loads(response_path.read_text())
        if set(response) != {'answer_ids', 'offsets', 'text', 'token_text', 'special_ids', 'units'}:
            raise ValueError('Response cache must contain the known pure fields only')
        if response != record['response'] or source['prompt_with_source'] != record['prompt'] or source['source_mask'] != record['source_mask']:
            raise ValueError('Source-first/roster token identity differs')
        if len(response['answer_ids']) != record['token_count'] or len(record['prompt']) != record['prompt_length']:
            raise ValueError('Hidden answer truncation or prompt-length change')
        if record['source_id'] not in maps:
            maps[record['source_id']] = compile_orders(tokenizer, prompts[record['source_id']],
                record['prompt'], record['source_mask'])
        else:
            if maps[record['source_id']]['variants']['native']['prompt_ids'] != record['prompt']:
                raise ValueError('Same-source answers have different original prompts')
        reference_path = REFERENCE / identity / 'arrays.npz'
        with np.load(reference_path) as saved:
            reference = {name: saved[name].copy() for name in
                ('actual_id', 'rival_id', 'query_position', 'actual_logp', 'margin')}
        if not np.array_equal(reference['actual_id'], response['answer_ids']) or not np.array_equal(
                reference['query_position'], np.arange(record['prompt_length'] - 1, record['prompt_length'] + record['token_count'] - 1)):
            raise ValueError('Native reference target/position differs')
        references[identity] = reference
        for path in (source_path, response_path, reference_path):
            dependencies[str(path.resolve())] = file_hash(path)
    write_json(output / 'INPUTS.json', document)
    write_json(output / 'PROMPT_MAPS.json', maps)
    return document, maps, references, dependencies


def bind_execution(output, dependencies):
    """Bind the files actually imported by the focused collector and tests."""
    names = [Path(__file__), Path(__file__).with_name('source_order.py'),
        Path(__file__).with_name('test_source_order.py'), Path('experiments/decision_risk_flow/run.py'),
        Path('experiments/decision_risk_flow/precision.py'),
        Path('experiments/probabilistic_detection/data.py')]
    for path in names:
        destination = output / 'code_snapshot' / path.resolve().relative_to(Path.cwd().resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        dependencies[str(path.resolve())] = file_hash(path)
    weights = {}
    for path in sorted(MODEL.iterdir()):
        if path.is_file() and (path.suffix in ('.json', '.safetensors', '.bin') or path.name.endswith('.model')):
            weights[str(path.resolve())] = file_hash(path)
    if not any(path.endswith(('.safetensors', '.bin')) for path in weights):
        raise ValueError('No actual frozen weight byte identities found')
    write_json(output / 'DEPENDENCIES.json', dependencies)
    write_json(output / 'MODEL_FILES.json', dict(hash_method='fresh full-file SHA256 before load', files=weights))
    return weights


def runtime(model):
    properties = torch.cuda.get_device_properties(model.device)
    return dict(python_executable=sys.executable, python_version=platform.python_version(),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        cuda_version=torch.version.cuda, device=str(model.device), gpu_name=properties.name,
        gpu_total_bytes=properties.total_memory, cpu_threads=torch.get_num_threads(),
        model_config=model.config.to_dict(), weight_dtype=str(next(model.parameters()).dtype),
        parameter_gradients_enabled=any(p.requires_grad for p in model.parameters()), training=model.training,
        attention_implementation=model.config._attn_implementation,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        sdpa_flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_math_enabled=torch.backends.cuda.math_sdp_enabled(),
        sdpa_mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
        arithmetic='existing frozen BF16 weight / FP32 native execution loader',
        historical_runtime_attestation=False)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    forwards, attempts, projections, events = 0, 0, 0, []
    try:
        document, maps, references, dependencies = prepare_inputs(output)
        weights = bind_execution(output, dependencies)
        write_json(output / 'PROTOCOL.json', dict(schema='source_order_equivariance_v1',
            plan=str(PLAN.resolve()), plan_sha256=dependencies[str(PLAN.resolve())],
            conditions=CONDITIONS, targets=1139, answers=8, native_before_all_cycles=True,
            native_tolerance=.001, identity_tolerance=1e-6, rival_policy='freeze original native highest nonactual',
            complete_forwards_planned=32, lm_head_chunk_rows=16, new_label_fits=0,
            annotation_parsed=False, entropy_risk=False, primary='margin_range',
            descriptive_only=True, natural_generator_causality=False, fusion_or_graph_changes=False))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        model = load_model(document['model'])
        if len(model.model.layers) != 32 or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen full-model observer contract differs')
        write_json(output / 'MODEL_ENV.json', runtime(model))
        native, reproduction, identity, frozen = {}, {}, {}, {}
        for record in document['records']:
            name = record['id']
            directory = output / 'capture' / name
            directory.mkdir(parents=True)
            prompt = maps[record['source_id']]['variants']['native']['prompt_ids']
            answer = record['response']['answer_ids']
            attempts += 1
            measured, audit = complete_readout(model, prompt, answer)
            forwards += 1
            projections += audit['lm_head_projection_calls']
            events.append(dict(id=name, condition='native', **audit))
            np.savez(directory / 'native.npz', **measured)
            reproduction[name] = verify_readout(measured, references[name], .001)
            attempts += 1
            sham, sham_audit = complete_readout(model, prompt, answer)
            forwards += 1
            projections += sham_audit['lm_head_projection_calls']
            events.append(dict(id=name, condition='identity_replay', **sham_audit))
            np.savez(directory / 'identity.npz', **sham)
            identity[name] = verify_readout(sham, measured, 1e-6)
            native[name] = measured
            frozen[name] = dict(actual_ids=measured['actual_id'].tolist(), rival_ids=measured['rival_id'].tolist(),
                native_path=str((directory / 'native.npz').resolve()), native_sha256=file_hash(directory / 'native.npz'))
            print(json.dumps(dict(stage='native_identity', id=name, forwards=forwards,
                native_max_error=reproduction[name]['max_absolute_errors'], identity_max_error=identity[name]['max_absolute_errors'])), flush=True)
        write_json(output / 'NATIVE_REPRODUCTION.json', reproduction)
        write_json(output / 'IDENTITY_AUDIT.json', identity)
        write_json(output / 'NATIVE_TARGETS_FROZEN.json', frozen)
        frozen_hash = file_hash(output / 'NATIVE_TARGETS_FROZEN.json')
        for record in document['records']:
            name = record['id']
            original = native[name]
            conditions = [original]
            for condition in CONDITIONS[1:]:
                if file_hash(output / 'NATIVE_TARGETS_FROZEN.json') != frozen_hash:
                    raise ValueError('Frozen original native targets changed before cycles')
                prompt = maps[record['source_id']]['variants'][condition]['prompt_ids']
                attempts += 1
                measured, audit = complete_readout(model, prompt, record['response']['answer_ids'], original['rival_id'])
                forwards += 1
                projections += audit['lm_head_projection_calls']
                events.append(dict(id=name, condition=condition, **audit))
                for key in ('actual_id', 'rival_id', 'query_position'):
                    if not np.array_equal(measured[key], original[key]):
                        raise ValueError(f'Cyclic condition changed the frozen target: {key}')
                conditions.append(measured)
            logp = np.stack([condition['actual_logp'] for condition in conditions])
            margin = np.stack([condition['margin'] for condition in conditions])
            directory = output / 'capture' / name
            np.savez(directory / 'arrays.npz', condition=np.asarray(CONDITIONS),
                actual_id=original['actual_id'], rival_id=original['rival_id'],
                query_position=np.stack([condition['query_position'] for condition in conditions]),
                actual_logp=logp, margin=margin, **order_fields(logp, margin))
            print(json.dumps(dict(stage='two_cycles', id=name, forwards=forwards)), flush=True)
        if forwards != 32 or file_hash(output / 'NATIVE_TARGETS_FROZEN.json') != frozen_hash:
            raise ValueError('Full eight-answer measurement not complete')
        # Re-read consumed non-model files: do not claim a mixed-identity capture.
        for path, expected in dependencies.items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f'Consumed dependency changed during capture: {path}')
        write_json(output / 'FORWARD_LEDGER.json', dict(events=events, full_model_forwards=forwards,
            full_readout_attempts=attempts, lm_head_projection_calls=projections, backward_calls=0,
            unfinished_readout_attempts=attempts - forwards))
        complete = dict(status='DONE', answers=8, targets=1139, full_model_forwards=forwards,
            lm_head_projection_calls=projections, backward_calls=0, natural_label_fits=0,
            annotation_parsed=False, seconds=time.perf_counter() - started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), primary='margin_range',
            descriptive_only=True, original_rival_freeze_sha256=frozen_hash)
        write_json(output / 'CAPTURE_COMPLETE.json', complete)
        artifacts = {str(path.relative_to(output)): file_hash(path) for path in sorted(output.rglob('*')) if path.is_file()}
        write_json(output / 'FREEZE.json', dict(status='ALL_MEASUREMENTS_FROZEN_BEFORE_LABEL_DIAGNOSTIC',
            artifacts=artifacts, dependencies=dependencies, model_file_preload_hashes=weights,
            model_bytes_after_capture_rehashed=False))
        print(json.dumps(complete), flush=True)
    except BaseException as error:
        write_json(output / 'CAPTURE_FAILED.json', dict(status='FAILED_NO_LABEL_EVALUATION',
            error_type=type(error).__name__, message=str(error), seconds=time.perf_counter() - started,
            completed_full_model_forwards=forwards, lm_head_projection_calls=projections,
            full_readout_attempts=attempts, unfinished_readout_attempts=attempts - forwards,
            annotation_parsed=False, events=events))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('capture',), default='capture')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == '__main__':
    main()
