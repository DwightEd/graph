"""Post-freeze CPU diagnostics for the fixed eight-answer source-order challenge.

No natural labels are opened until every capture/dependency/identity gate has
passed. Order sensitivity is descriptive, not a factual probability or a new
source+route/graph detector. Matched false alarms are explicitly oracle only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels, valid_tokens
from experiments.probabilistic_detection.evaluation import pairwise_within
from .source_route_refine_eval import (PACK_FIELDS, answer_masks, alarm_counts,
                                       threshold_for_budget, transitions,
                                       verify_alignment)


OUTPUT = Path('outputs/source_order_equivariance_20261009')
BASELINE = Path('outputs/source_route_state_20261009_v2')
REFERENCE = Path('outputs/source_route_refine_20261009/REFINE_REFERENCE.json')
CACHE = Path('outputs/native_support_ragtruth_all/source_first_v1')
AUX = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/'
           'source_order_equivariance_20261009/EVALUATOR_AUX.json')
PRE_CAPTURE_AUX_SHA256 = '6a81bcf36ae9e1e1c29217944b98be813b150796c175e2c8253a95822e9584db'
PRE_CAPTURE_SOURCE_SHA256 = 'efc56ff3078636cfd2bf556111dc5ea04a8c592b6a1c63c89f688d999d4091e2'
CONDITIONS = ('native', 'cycle_fwd', 'cycle_rev')
FIELDS = ('margin_range', 'actual_logp_range', 'signed_margin', 'signed_actual_logp')
PRIMARY = 'margin_range'
EXPECTED = [('12219', '14353', 328, 269), ('12216', '14353', 328, 114),
            ('12297', '14366', 276, 156), ('12294', '14366', 276, 53),
            ('15604', '15521', 282, 111), ('15600', '15521', 282, 93),
            ('11907', '14300', 427, 206), ('11904', '14300', 427, 137)]
KEY_TARGETS = {('12219', 223): 'not', ('12297', 106): '2',
               ('15604', 96): 'private', ('15604', 103): 'normal'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_hashes(root, hashes):
    """Hash bytes only, including GT dependencies; never parse annotations here."""
    for name, expected in hashes.items():
        path = Path(name) if root is None else root / name
        if root is not None and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Artifact path escapes capture: {name}')
        if file_hash(path) != expected:
            raise ValueError(f'Frozen bytes changed: {path}')


def descriptors(logp, margin):
    """Independent reconstruction of precisely the four predeclared scalars."""
    return dict(margin_range=np.ptp(margin, axis=0), actual_logp_range=np.ptp(logp, axis=0),
        signed_margin=margin[1:].mean(axis=0) - margin[0],
        signed_actual_logp=logp[1:].mean(axis=0) - logp[0],
        delta_margin=margin[1:] - margin[0], delta_actual_logp=logp[1:] - logp[0])


def evaluation_attachment(directory, freeze_path, model_files):
    """Bind post-capture guard repair without rewriting the original capture freeze.

    The original auxiliary file was archived before this guard-only edit. It
    describes a pre-capture source snapshot, not the code eventually executed.
    """
    path = directory / 'EVALUATOR_ATTACHMENT.json'
    attachment = json.loads(path.read_text())
    if attachment['timing'] != 'after_complete_capture_before_label_diagnostics':
        raise ValueError('Explicit post-capture evaluator attachment is required')
    if attachment['capture_freeze_sha256'] != file_hash(freeze_path):
        raise ValueError('Post-capture attachment names another capture freeze')
    if attachment['original_aux_sha256'] != PRE_CAPTURE_AUX_SHA256 or file_hash(AUX) != PRE_CAPTURE_AUX_SHA256:
        raise ValueError('Original pre-capture auxiliary bytes must remain unchanged')
    if not attachment['metadata_guard_patch_only'] or attachment['pre_capture_executed_code_attestation']:
        raise ValueError('Guard repair must not claim pre-capture execution attestation')
    if tuple(attachment['ranked_fields']) != (*FIELDS, 'old_native') or attachment['primary'] != PRIMARY:
        raise ValueError('Post-capture guard repair must preserve fixed scientific definitions')
    if attachment['model_preload_hashes'] != model_files:
        raise ValueError('Attachment changed the capture model-file identity')
    live_source = str(Path(__file__).resolve())
    if live_source not in attachment['files']:
        raise ValueError('Executed evaluator source is not bound')
    checked_hashes(None, attachment['files'])
    aux = json.loads(AUX.read_text())
    if aux['files'][live_source] != PRE_CAPTURE_SOURCE_SHA256:
        raise ValueError('Original source snapshot identity differs')
    if file_hash(Path(aux['evaluator_snapshot'])) != PRE_CAPTURE_SOURCE_SHA256:
        raise ValueError('Original pre-capture evaluator snapshot changed')
    # The live script is intentionally repaired; its original hash remains on
    # the archived snapshot. Every other original auxiliary dependency is exact.
    checked_hashes(None, {name: digest for name, digest in aux['files'].items() if name != live_source})
    archives = attachment['pre_capture_archives']
    if set(archives.values()) != {PRE_CAPTURE_AUX_SHA256, PRE_CAPTURE_SOURCE_SHA256}:
        raise ValueError('Original auxiliary/source archives are required')
    checked_hashes(None, archives)
    return file_hash(path)


def read_frozen(directory):
    """Complete freeze, model bytes, execution ledger, pure inputs and old identity."""
    directory = Path(directory)
    if (directory / 'CAPTURE_FAILED.json').exists():
        raise ValueError('Failed capture is not eligible for label diagnostics')
    freeze_path = directory / 'FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    if freeze['status'] != 'ALL_MEASUREMENTS_FROZEN_BEFORE_LABEL_DIAGNOSTIC':
        raise ValueError('Whole capture freeze is required')
    required = {'INPUTS.json', 'PROMPT_MAPS.json', 'PROTOCOL.json', 'DEPENDENCIES.json',
        'MODEL_FILES.json', 'MODEL_ENV.json', 'CAPTURE_COMPLETE.json', 'FORWARD_LEDGER.json',
        'NATIVE_REPRODUCTION.json', 'IDENTITY_AUDIT.json', 'NATIVE_TARGETS_FROZEN.json'}
    required.update(f'capture/{identity}/{name}.npz' for identity, *_ in EXPECTED
                    for name in ('native', 'identity', 'arrays'))
    if not required.issubset(freeze['artifacts']):
        raise ValueError('Freeze omits a required whole-capture artifact')
    checked_hashes(directory, freeze['artifacts'])
    dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
    if dependencies != freeze['dependencies']:
        raise ValueError('Freeze and dependency ledger differ')
    checked_hashes(None, dependencies)
    model_files = json.loads((directory / 'MODEL_FILES.json').read_text())['files']
    if model_files != freeze['model_file_preload_hashes']:
        raise ValueError('Freeze and freshly hashed model-file ledger differ')
    attachment_hash = evaluation_attachment(directory, freeze_path, model_files)
    checked_hashes(None, model_files)
    complete = json.loads((directory / 'CAPTURE_COMPLETE.json').read_text())
    protocol = json.loads((directory / 'PROTOCOL.json').read_text())
    if (complete['status'], complete['answers'], complete['targets'],
            complete['full_model_forwards'], complete['backward_calls'],
            complete['natural_label_fits'], complete['annotation_parsed'], complete['primary']) != (
            'DONE', 8, 1139, 32, 0, 0, False, PRIMARY):
        raise ValueError('All-eight/32-forward/no-label contract differs')
    if tuple(protocol['conditions']) != CONDITIONS or protocol['primary'] != PRIMARY:
        raise ValueError('Fixed measurement conditions/primary changed')
    ledger = json.loads((directory / 'FORWARD_LEDGER.json').read_text())
    expected_events = [(identity, condition) for identity, *_ in EXPECTED
                       for condition in ('native', 'identity_replay')]
    expected_events += [(identity, condition) for identity, *_ in EXPECTED
                        for condition in CONDITIONS[1:]]
    if [(e['id'], e['condition']) for e in ledger['events']] != expected_events:
        raise ValueError('Original native/identity must finish before all cyclic measurements')
    if ledger['full_model_forwards'] != 32 or ledger['full_readout_attempts'] != 32 or ledger['backward_calls'] != 0:
        raise ValueError('Incomplete or retried full-capture ledger')
    if any(e['layer_visits'] != [1] * 32 or not e['fresh_full_prefill'] or e['cached_KV_reused']
           for e in ledger['events']):
        raise ValueError('Every condition must recompute all 32 native layers')
    for filename, tolerance in (('NATIVE_REPRODUCTION.json', .001), ('IDENTITY_AUDIT.json', 1e-6)):
        audit = json.loads((directory / filename).read_text())
        if set(audit) != {row[0] for row in EXPECTED}:
            raise ValueError('Identity audit does not cover every answer')
        if any(not row['passed'] or not row['identity_and_rivals_exact'] or
               max(row['max_absolute_errors'].values()) > tolerance for row in audit.values()):
            raise ValueError(f'Identity numeric gate failed: {filename}')
    targets_frozen = json.loads((directory / 'NATIVE_TARGETS_FROZEN.json').read_text())
    if file_hash(directory / 'NATIVE_TARGETS_FROZEN.json') != complete['original_rival_freeze_sha256']:
        raise ValueError('Original competitor freeze changed')
    old_freeze = json.loads((BASELINE / 'FREEZE.json').read_text())
    checked_hashes(BASELINE, old_freeze['hashes'])
    with np.load(BASELINE / 'pack.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(PACK_FIELDS):
            raise ValueError('Old identity pack must not contain labels')
        pack = {key: saved[key].copy() for key in PACK_FIELDS}
    with np.load(BASELINE / 'scores.npz', allow_pickle=False) as saved:
        scores = {'old_native': saved['old_native'].copy()}
    metadata = json.loads((BASELINE / 'metadata.json').read_text())
    old_cut = float(json.loads(REFERENCE.read_text())['thresholds']['old_native'])
    verify_alignment(pack, scores, metadata, {'old_native': old_cut})
    inputs = json.loads((directory / 'INPUTS.json').read_text())
    if [(r['id'], r['source_id'], r['prompt_length'], r['token_count'])
            for r in inputs['records']] != EXPECTED:
        raise ValueError('Original whole-answer roster differs')
    if [r['id'] for r in metadata['records']] != [r[0] for r in EXPECTED]:
        raise ValueError('Old pack/metadata answer order differs')
    maps = json.loads((directory / 'PROMPT_MAPS.json').read_text())
    captures, responses, batches = {}, {}, {name: [] for name in FIELDS}
    for frozen_record, record, (identity, source, prompt_length, count) in zip(
            inputs['records'], metadata['records'], EXPECTED):
        path = (CACHE / record['directory'] / 'response.json').resolve()
        if str(path) not in dependencies or record['source_id'] != source or record['tokens'] != count:
            raise ValueError(f'{identity}: response-cache/source identity not bound')
        response = json.loads(path.read_text())
        if set(response) != {'answer_ids', 'offsets', 'text', 'token_text', 'special_ids', 'units'} or response != frozen_record['response']:
            raise ValueError(f'{identity}: original pure response changed')
        ids = np.asarray(response['answer_ids'])
        region = slice(record['packed_start'], record['packed_stop'])
        targets = pack['target'][region]
        if not np.array_equal(targets, np.flatnonzero(valid_tokens(response))) or not np.array_equal(
                ids[targets], pack['token_id'][region]):
            raise ValueError(f'{identity}: packed valid targets/original token IDs differ')
        variants = maps[source]['variants']
        original_prompt = np.asarray(frozen_record['prompt'])
        if not np.array_equal(variants['native']['prompt_ids'], original_prompt):
            raise ValueError(f'{identity}: native prompt IDs differ')
        for condition in CONDITIONS:
            variant = variants[condition]
            permutation = np.asarray(variant['new_to_old'])
            inverse = np.asarray(variant['old_to_new'])
            if len(permutation) != prompt_length or not np.array_equal(np.sort(permutation), np.arange(prompt_length)):
                raise ValueError(f'{identity}: nonbijective prompt map')
            if not np.array_equal(permutation[inverse], np.arange(prompt_length)) or not np.array_equal(
                    variant['prompt_ids'], original_prompt[permutation]):
                raise ValueError(f'{identity}: prompt content or inverse map differs')
        with np.load(directory / 'capture' / identity / 'arrays.npz', allow_pickle=False) as saved:
            measured = {key: saved[key].copy() for key in saved.files}
        if tuple(measured['condition']) != CONDITIONS or not np.array_equal(measured['actual_id'], ids):
            raise ValueError(f'{identity}: condition/actual token identities differ')
        positions = np.arange(prompt_length - 1, prompt_length + count - 1)
        if not np.array_equal(measured['query_position'], np.tile(positions, (3, 1))):
            raise ValueError(f'{identity}: predictor must be P+t-1 in every condition')
        for key in ('actual_logp', 'margin'):
            if measured[key].shape != (3, count) or not np.isfinite(measured[key]).all():
                raise ValueError(f'{identity}: incomplete/nonfinite whole-target output')
        if np.any(measured['rival_id'] == ids) or not np.array_equal(
                measured['rival_id'], targets_frozen[identity]['rival_ids']):
            raise ValueError(f'{identity}: original competitor changed')
        for filename, tolerance in (('native', 0.), ('identity', 1e-6)):
            with np.load(directory / 'capture' / identity / f'{filename}.npz', allow_pickle=False) as saved:
                for key in ('actual_id', 'rival_id'):
                    np.testing.assert_array_equal(saved[key], measured[key])
                np.testing.assert_array_equal(saved['query_position'], positions)
                for key in ('actual_logp', 'margin'):
                    np.testing.assert_allclose(saved[key], measured[key][0], rtol=0, atol=tolerance)
        reconstructed = descriptors(measured['actual_logp'], measured['margin'])
        for name, values in reconstructed.items():
            np.testing.assert_array_equal(values, measured[name])
        for name in FIELDS:
            batches[name].append(measured[name][targets])
        captures[identity], responses[identity] = measured, response
    scores.update({name: np.concatenate(rows) for name, rows in batches.items()})
    verify_alignment(pack, scores, metadata, {name: old_cut for name in scores})
    return pack, scores, metadata, captures, responses, old_cut, dict(
        capture_freeze_sha256=file_hash(freeze_path), evaluator_aux_sha256=file_hash(AUX),
        evaluator_attachment_sha256=attachment_hash, evaluator_guard_repaired_after_capture=True,
        pre_capture_executed_evaluator_attestation=False,
        all_capture_dependencies_verified=True, model_preload_hashes_reverified=True,
        old_baseline_freeze_sha256=file_hash(BASELINE / 'FREEZE.json'),
        whole_answer_targets=1139, evaluation_valid_tokens=len(pack['target']), labels_opened=False)


def summaries(values):
    return dict(tokens=len(values), mean=float(np.mean(values)) if len(values) else None,
        median=float(np.median(values)) if len(values) else None,
        maximum=float(np.max(values)) if len(values) else None)


def rank_report(pack, values, clean, first, answers, cutoff):
    healthy = clean | first
    onset = (pack['labels'] == 0) | pack['onsets']
    return dict(all_tokens=ranking(pack['labels'], values),
        within_answer_auroc=pairwise_within(pack['labels'], values, pack['answer_index']),
        strict_first_error=ranking(first[healthy].astype(int), values[healthy]),
        onsets_vs_normal=ranking(pack['onsets'][onset].astype(int), values[onset]),
        matched_normal_token_budget=dict(alarm_counts(pack, values, cutoff, clean, first, answers),
            label_assisted=True, oracle_diagnostic=True, allowed_fp=57, baseline='old_native',
            calibrated_normal_fpr=False, independent_deployment_threshold=False,
            ties='strict cutoff may use fewer than 57 false positives'))


def token_rows(pack, scores, records, captures, responses, thresholds):
    for record in records:
        identity = record['id']
        measured, response = captures[identity], responses[identity]
        for index in range(record['packed_start'], record['packed_stop']):
            target = int(pack['target'][index])
            row = dict(id=identity, source_id=record['source_id'], generator=record['generator'],
                target=target, token_id=int(pack['token_id'][index]), word=response['token_text'][target],
                label=int(pack['labels'][index]), onset=int(pack['onsets'][index]),
                first_error=int(pack['firsts'][index]), rival_id=int(measured['rival_id'][target]),
                query_position=int(measured['query_position'][0, target]))
            for slot, condition in enumerate(CONDITIONS):
                row[condition + '__actual_logp'] = float(measured['actual_logp'][slot, target])
                row[condition + '__margin'] = float(measured['margin'][slot, target])
                if slot:
                    row[condition + '__delta_actual_logp'] = float(measured['delta_actual_logp'][slot - 1, target])
                    row[condition + '__delta_margin'] = float(measured['delta_margin'][slot - 1, target])
            for name, values in scores.items():
                row[name] = float(values[index])
                row[name + '__oracle_alarm'] = int(values[index] > thresholds[name])
            yield row


def evaluate(directory):
    directory = Path(directory)
    names = ('ORDER_DIAGNOSTICS.json', 'ORDER_TOKENS.csv', 'ORDER_KEY_TOKENS.json',
             'ORDER_DIAGNOSTICS_FREEZE.json')
    if any((directory / name).exists() for name in names):
        raise FileExistsError('Diagnostic outputs exist; refuse overwrite')
    pack, scores, metadata, captures, responses, old_cut, binding = read_frozen(directory)
    # This is the first annotation access, after ALL eight capture gates above.
    pack.update(evaluation_labels(CACHE, pack, metadata))
    clean, first, answers = answer_masks(pack, metadata['records'])
    baseline = alarm_counts(pack, scores['old_native'], old_cut, clean, first, answers)
    if baseline['fp'] != 57:
        raise ValueError('Old fixed baseline no longer has the predeclared 57 normal-token FP')
    thresholds = {name: threshold_for_budget(values[pack['labels'] == 0], 57)
                  for name, values in scores.items()}
    methods = {name: rank_report(pack, values, clean, first, answers, thresholds[name])
               for name, values in scores.items()}
    per_answer = []
    for record in metadata['records']:
        region = slice(record['packed_start'], record['packed_stop'])
        labels = pack['labels'][region]
        descriptor = {}
        for name, values in scores.items():
            selected = values[region]
            descriptor[name] = dict(ranking=ranking(labels, selected), all=summaries(selected),
                normal=summaries(selected[labels == 0]), error=summaries(selected[labels == 1]),
                oracle_tp=int(((selected > thresholds[name]) & (labels == 1)).sum()),
                oracle_fp=int(((selected > thresholds[name]) & (labels == 0)).sum()))
        measured = captures[record['id']]
        targets = pack['target'][region]
        per_answer.append(dict(id=record['id'], source_id=record['source_id'], generator=record['generator'],
            original_targets=len(responses[record['id']]['answer_ids']), valid_tokens=len(labels),
            positives=int(labels.sum()), descriptors=descriptor,
            condition_readouts={condition: {key: summaries(measured[key][slot, targets])
                for key in ('actual_logp', 'margin')} for slot, condition in enumerate(CONDITIONS)},
            signed_cycle_responses={condition: {key: summaries(measured[key][slot - 1, targets])
                for key in ('delta_actual_logp', 'delta_margin')}
                for slot, condition in enumerate(CONDITIONS) if slot}))
    report = dict(primary=PRIMARY, secondary='actual_logp_range', ranked_fields=[*FIELDS, 'old_native'],
        method_selection=False, direction_flipped=False, new_model_or_risk_fit=False,
        fusion_or_graph_changed=False, descriptive_order_sensitivity_only=True,
        no_new_fit_threshold=True, historical_exposure=True, original_generator_internal_states=False,
        confounds=['document order changes RoPE position and causal document-prefix states together',
                  'same-source answers come from different generators and have different forced prefixes',
                  'normal cross-document steps/topic transitions may be order sensitive',
                  'within-document wrong subject/scope binding may remain stable'],
        counts=dict(answers=8, sources=4, full_targets=1139, valid_tokens=len(pack['labels']),
            positives=int(pack['labels'].sum()), normal_tokens=int((pack['labels'] == 0).sum()),
            first_errors=int(first.sum()), onsets=int(pack['onsets'].sum())),
        old_baseline_fit95_reference=baseline, methods=methods, per_answer=per_answer,
        baseline_to_primary_at_oracle57=transitions(pack['labels'], scores['old_native'], scores[PRIMARY],
            thresholds['old_native'], thresholds[PRIMARY]), binding=dict(binding, labels_opened=True),
        limitation='Four exposed sources are a mechanism pilot, not complete QA DEV/test or AUROC>=.8 validation')
    rows = list(token_rows(pack, scores, metadata['records'], captures, responses, thresholds))
    with (directory / names[1]).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [dict(row, selection='fixed historical target' if (row['id'], row['target']) in KEY_TARGETS
                     else 'all tokens with fold/corner spelling prefix') for row in rows
                if (row['id'], row['target']) in KEY_TARGETS or
                row['word'].strip().lower().startswith(('fold', 'corner'))]
    if not set(KEY_TARGETS).issubset({(row['id'], row['target']) for row in selected}):
        raise ValueError('A prespecified key target is absent from valid evaluation rows')
    for name, value in ((names[2], selected), (names[0], report)):
        with (directory / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    diagnostic_hashes = {name: file_hash(directory / name) for name in names[:3]}
    with (directory / names[3]).open('x', encoding='utf-8') as stream:
        json.dump(dict(capture_freeze_sha256=binding['capture_freeze_sha256'],
            evaluator_aux_sha256=binding['evaluator_aux_sha256'],
            evaluator_attachment_sha256=binding['evaluator_attachment_sha256'],
            artifacts=diagnostic_hashes), stream, indent=2)
        stream.write('\n')
    return report


def plot(directory):
    """Separate post-evaluation eight-panel artifact, all original target locations."""
    directory = Path(directory)
    png, pdf = directory / 'ORDER_TOKEN_PANELS.png', directory / 'ORDER_TOKEN_PANELS.pdf'
    if png.exists() or pdf.exists():
        raise FileExistsError('Plot exists; refuse overwrite')
    freeze = json.loads((directory / 'ORDER_DIAGNOSTICS_FREEZE.json').read_text())
    checked_hashes(directory, freeze['artifacts'])
    if file_hash(directory / 'FREEZE.json') != freeze['capture_freeze_sha256']:
        raise ValueError('Capture freeze changed after diagnostics')
    if file_hash(directory / 'EVALUATOR_ATTACHMENT.json') != freeze['evaluator_attachment_sha256']:
        raise ValueError('Post-capture evaluator attachment changed after diagnostics')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return dict(status='unavailable_existing_matplotlib_missing', installed=False)
    report = json.loads((directory / 'ORDER_DIAGNOSTICS.json').read_text())
    with (directory / 'ORDER_TOKENS.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(4, 2, figsize=(15, 12), sharey=True, constrained_layout=True)
    cutoff = report['methods'][PRIMARY]['matched_normal_token_budget']['threshold']
    for axis, (identity, source, _, count) in zip(axes.flat, EXPECTED):
        selected = [row for row in rows if row['id'] == identity]
        x = np.asarray([int(row['target']) for row in selected])
        y = np.asarray([float(row[PRIMARY]) for row in selected])
        error = np.asarray([int(row['label']) for row in selected], dtype=bool)
        axis.plot(x, y, color='#225b88', linewidth=.9, label='fixed margin range')
        axis.scatter(x[error], y[error], color='#bd3b3b', s=9, label='official error token', zorder=3)
        for target in x[error]:
            axis.axvspan(target - .5, target + .5, color='#bd3b3b', alpha=.10, linewidth=0)
        axis.axhline(cutoff, color='#777777', linestyle='--', linewidth=.8, label='57-FP oracle cutoff')
        axis.set_xlim(-1, count)
        axis.set_title(f"source {source} / response {identity} / {selected[0]['generator']} / error {error.sum()}", fontsize=9)
        axis.set_xlabel('original answer target t')
        axis.set_ylabel('order sensitivity, logit-margin range')
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Four same-source pairs; different generators/prefixes; exposed pilot, descriptive only', fontsize=12)
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)
    return dict(status='created', png=str(png), pdf=str(pdf), png_sha256=file_hash(png),
                pdf_sha256=file_hash(pdf), plotted_targets=len(rows), all_eight_answers=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('evaluate', 'plot'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.output) if args.stage == 'evaluate' else plot(args.output)
    if args.stage == 'evaluate':
        result = dict(primary=PRIMARY, counts=result['counts'], ranking=result['methods'][PRIMARY]['all_tokens'])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
