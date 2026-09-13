"""Independent, read-only artifact witness; never opens annotation/gold files."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_split(text_sha):
    return 'validation' if int(digest([20260913, 'source-split', text_sha])[:16], 16) % 5 == 0 else 'train'


def checked_manifest(path, manifest, expected):
    assert set(manifest['artifacts']) == expected
    for name, checksum in manifest['artifacts'].items():
        assert sha(path / name) == checksum, name


def code_check(path, settings):
    for name, checksum in settings['code_sha256'].items():
        assert sha(ROOT / name) == checksum == sha(path / 'executed_code' / name), name


def reconstruction_check():
    path = ROOT / 'outputs/grounded_graph_reconstruction_v1_20260913'
    settings, manifest = read(path / 'settings.json'), read(path / 'manifest.json')
    examples, summary = read(path / 'examples.json'), read(path / 'summary.json')
    assert manifest['status'] == 'complete'
    assert sha(path / 'settings.json') == manifest['settings_sha256']
    checked_manifest(path, manifest, {'examples.json', 'summary.json'})
    code_check(path, settings)
    assert sha(settings['input_path']) == settings['input_sha256']
    assert settings['labels_read'] is False and settings['protocol']['generated_templates'] is False
    sources = {}
    with Path(settings['input_path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            assert not any(k in row for k in ('labels', 'hallucination_spans', 'gold'))
            if row['official_split'] != 'train':
                continue
            sid = str(row['source_id'])
            text = row['prompt'][slice(*row['source_span'])]
            text_sha = hashlib.sha256(text.encode()).hexdigest()
            if sid in sources:
                assert sources[sid]['text'] == text and sources[sid]['task'] == row['task']
            if sid not in sources or int(row['id']) < int(sources[sid]['row']['id']):
                sources[sid] = dict(source_id=sid, task=row['task'], text=text, text_sha=text_sha, split=source_split(text_sha), row=row)
    chosen, census = [], {}
    for task in ('QA', 'Summary', 'Data2txt'):
        for split, count in (('train', 64), ('validation', 16)):
            candidates = sorted((x for x in sources.values() if x['task'] == task and x['split'] == split),
                key=lambda x: (digest([20260913, 'source-order', x['text_sha']]), x['source_id']))
            unique = {}
            for x in candidates:
                unique.setdefault(x['text_sha'], x)
            selected = list(unique.values())[:count]
            assert len(selected) == count
            chosen.extend(selected)
            census[task + ':' + split] = dict(eligible_source_ids=len(candidates), unique_texts=len(unique), selected=count)
    assert census == settings['selection_census'] == summary['census']
    assert [x['source_id'] for x in chosen] == [x['source_id'] for x in examples]
    assert set(settings['source_artifacts']) == {'sources/' + x['source_id'] + '.json' for x in chosen}
    inventory = Path(settings['inventory_path'])
    assert sha(inventory / 'manifest.json') == settings['inventory_manifest_sha256']
    inv_settings = read(inventory / 'settings.json')
    assert sha(inventory / 'settings.json') == settings['inventory_settings_sha256']
    assert digest(inv_settings) == settings['inventory_settings_digest']
    pointer_count = 0
    for original, example in zip(chosen, examples):
        name = 'sources/' + example['source_id'] + '.json'
        assert sha(inventory / name) == settings['source_artifacts'][name]
        wrapped = read(inventory / name)
        inv = wrapped['data']
        assert inv['sha256'] == digest({k: v for k, v in inv.items() if k != 'sha256'})
        assert wrapped['settings_object_digest'] == digest(inv_settings)
        assert example['text'] == inv['text'] == original['text']
        assert example['source_text_sha256'] == original['text_sha']
        assert example['split'] == original['split']
        prompt = example['prompt_record']
        assert prompt['prompt'] == original['row']['prompt']
        assert prompt['prompt_token_ids'] == original['row']['token_ids'][:original['row']['prompt_length']]
        owners = {x['id']: x for x in (inv['fields'] if inv['task'] == 'Data2txt' else inv['contexts'])}
        words = list(re.finditer(r"\w+(?:['’\-]\w+)*", example['text']))
        cutoff = words[8].start() if len(words) > 8 else len(example['text'])
        assert 0 < len(example['targets']) <= 64
        seen = set()
        for target in example['targets']:
            a, b = target['target_span']
            owner = owners[target['source_owner_id']]
            assert target['target_span'] == target['source_span']
            assert a >= cutoff and owner['raw_span'][0] <= a < b <= owner['raw_span'][1]
            assert example['text'][a:b] == target['raw_value']
            assert owner.get('value_status') != 'unknown'
            assert (a, b) not in seen
            seen.add((a, b))
        pointer_count += len(example['targets'])
    train = {x['source_text_sha256'] for x in examples if x['split'] == 'train'}
    val = {x['source_text_sha256'] for x in examples if x['split'] == 'validation'}
    assert len(train) == 192 and len(val) == 48 and not train & val
    assert summary['sources'] == len(examples) == 240
    assert summary['pointers'] == pointer_count == 9220 and summary['sources_without_pointer'] == 0
    return dict(summary=summary, manifest_sha256=sha(path / 'manifest.json'), code_files=len(settings['code_sha256']),
        selected_roster_rederived_from_frozen_input=True, unique_train_sha=len(train), unique_validation_sha=len(val), source_sha_overlap=0)


def feature_check(name, tokenizer):
    path = ROOT / 'outputs' / name
    settings, prepared = read(path / 'settings.json'), read(path / 'prepare_manifest.json')
    manifest, summary, entries = read(path / 'manifest.json'), read(path / 'summary.json'), read(path / 'entries.json')
    code_check(path, settings)
    assert prepared['status'] == 'prepared' and manifest['status'] == 'complete'
    assert prepared['settings_sha256'] == manifest['settings_sha256'] == sha(path / 'settings.json')
    assert manifest['prepare_manifest_sha256'] == sha(path / 'prepare_manifest.json')
    assert len({e['id'] for e in entries}) == len(entries)
    checked_manifest(path, prepared, {'entries.json'} | {e['packet_file'] for e in entries})
    expected = {'summary.json'} | {f"features/{e['id']}{suffix}" for e in entries if e['status'] == 'available' for suffix in ('.json', '.npz')}
    checked_manifest(path, manifest, expected)
    assert {str(p.relative_to(path)) for p in (path / 'features').iterdir()} == expected - {'summary.json'}
    assert summary['examples'] == len(entries)
    assert summary['counts'] == dict(Counter(e['status'] for e in entries))
    assert summary['tokens'] == sum(e['tokens'] for e in entries)
    inventory = Path(settings['inventory_path'])
    assert sha(inventory / 'manifest.json') == settings['inventory_manifest_sha256']
    assert sha(inventory / 'settings.json') == settings['inventory_settings_sha256']
    parent = settings['parent']
    natural = parent['kind'] == 'natural_response'
    if natural:
        assert sha(parent['path']) == parent['input_sha256']
        original_rows = [json.loads(line) for line in Path(parent['path']).open()]
        assert [str(x['id']) for x in original_rows] == [e['id'] for e in entries]
        original_rows = {str(x['id']): x for x in original_rows}
    else:
        assert parent['kind'] == 'source_reconstruction'
        assert sha(Path(parent['path']) / 'manifest.json') == parent['manifest_sha256']
        originals = read(Path(parent['path']) / 'examples.json')
        assert [x['id'] for x in originals] == [e['id'] for e in entries]
        originals = {x['id']: x for x in originals}
    total_nodes = total_edges = total_bytes = total_words = total_forwards = 0
    unavailable_nodes = 0
    source_rows, query_rows, all_nodes, all_edges = [], [], [], []
    kinds = settings['protocol']['node_kinds']
    edge_kinds = settings['protocol']['edge_kinds']
    assert len(kinds) == 9 and len(edge_kinds) == 4
    for e in entries:
        packet = read(path / e['packet_file'])
        assert packet['sha256'] == e['packet_sha256'] == digest({k: v for k, v in packet.items() if k != 'sha256'})
        assert packet['source_id'] == e['source_id']
        source_name = f"sources/{e['source_id']}.json"
        assert sha(inventory / source_name) == settings['source_artifacts'][source_name]
        inv = read(inventory / source_name)['data']
        assert inv['sha256'] == digest({k: v for k, v in inv.items() if k != 'sha256'})
        assert packet['inventory_sha256'] == inv['sha256']
        prompt = packet['prompt_record']
        assert prompt['prompt'][slice(*prompt['source_span'])] == inv['text']
        assert hashlib.sha256(inv['text'].encode()).hexdigest() == e['source_text_sha256']
        p = tokenizer(prompt['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        response = tokenizer(packet['response_text'], add_special_tokens=False, return_offsets_mapping=True)
        pids = [tokenizer.bos_token_id] + p['input_ids']
        assert pids == prompt['prompt_token_ids'] and len(pids) == prompt['prompt_length']
        assert packet['input_ids'] == (pids + response['input_ids'])[:-1]
        assert packet['target_token_ids'] == response['input_ids']
        assert packet['response_offsets'] == [list(x) for x in response['offset_mapping']]
        assert packet['target_positions'] == list(range(len(pids), len(pids) + len(response['input_ids'])))
        assert packet['query_positions'] == [i - 1 for i in packet['target_positions']]
        if natural:
            row = original_rows[e['id']]
            assert packet['input_ids'] + packet['target_token_ids'][-1:] == row['token_ids']
            assert packet['response_offsets'] == row['offsets']
            assert packet['response_text'] == row['response']
            assert e['row_sha256'] == digest(row)
            assert packet['weak_targets'] == [] and packet['pointer_supervision'] == []
        a, b = prompt['source_span']
        source_offsets = [[max(0, left-a), min(b-a, right-a)] if left < b and right > a else [0, 0]
            for left, right in [[0, 0], *p['offset_mapping']]]
        assert packet['source_offsets'] == source_offsets
        inv_nodes = inv['fields'] + inv['contexts'] + inv['components'] + inv['bundles']
        nodes = packet['nodes']
        assert [n['id'] for n in nodes] == [n['id'] for n in inv_nodes]
        positions = {n['id']: i for i, n in enumerate(nodes)}
        if not natural:
            original = originals[e['id']]
            assert packet['response_text'] == original['text']
            assert packet['prompt_record'] == original['prompt_record']
            assert e['split'] == original['split'] and e['task'] == original['task']
            expected_weak, pointer = [], {}
            for target in original['targets']:
                left, right = target['target_span']
                indices = [i for i, (lo, hi) in enumerate(packet['response_offsets']) if lo < right and hi > left and hi > lo]
                owner = positions[target['source_owner_id']]
                assert nodes[owner]['available'] and indices
                expected_weak.append(dict(target_span=target['target_span'], source_owner_id=target['source_owner_id'],
                    token_indices=indices, first_token=indices[0], owner_node_index=owner))
                pointer.setdefault(indices[0], set()).add(owner)
            assert packet['weak_targets'] == expected_weak
            assert packet['pointer_supervision'] == [dict(query_index=i, owner_node_indices=sorted(v)) for i, v in sorted(pointer.items())]
        for node, original in zip(nodes, inv_nodes):
            assert node['kind'] == original['kind'] and node['type'] == kinds.index(original['kind'])
            assert node['raw_span'] == original.get('raw_span')
            assert node['member_ids'] == original.get('member_ids', [])
            if 'member_ids' in original:
                expected_tokens = set().union(*(set(nodes[positions[m]]['prompt_token_indices']) for m in original['member_ids']))
            else:
                left, right = original['raw_span']
                expected_tokens = {i for i, (lo, hi) in enumerate(source_offsets) if lo < right and hi > left and hi > lo}
            assert node['prompt_token_indices'] == sorted(expected_tokens)
            assert node['available'] == bool(expected_tokens)
            assert all(i < len(pids) for i in expected_tokens)
        edges = []
        for edge in inv['edges']:
            kind = edge_kinds.index(edge['kind'])
            edges += [[positions[edge['from']], positions[edge['to']], kind], [positions[edge['to']], positions[edge['from']], kind+2]]
        assert packet['edges'] == edges
        assert packet['node_kind_vocabulary'] == kinds and packet['edge_kind_vocabulary'] == edge_kinds
        assert e['nodes'] == len(nodes) and e['tokens'] == len(response['input_ids']) and e['input_tokens'] == len(packet['input_ids'])
        expected_status = 'available' if len(packet['input_ids']) <= settings['protocol']['max_tokens'] and any(n['available'] for n in nodes) else 'unavailable_length_or_source'
        assert e['status'] == expected_status
        total_nodes += len(nodes)
        total_edges += len(edges)
        total_words += len(re.findall(r'\S+', packet['response_text']))
        unavailable_nodes += sum(not n['available'] for n in nodes)
        all_nodes.append(len(nodes)); all_edges.append(len(edges))
        if e['status'] != 'available':
            continue
        receipt = read(path / 'features' / f"{e['id']}.json")
        assert receipt['sha256'] == digest({k: v for k, v in receipt.items() if k != 'sha256'})
        assert receipt['packet_sha256'] == packet['sha256'] and receipt['input_ids_sha256'] == digest(packet['input_ids'])
        assert receipt['representation'] == packet['representation'] == settings['protocol']['representation']
        assert receipt['model_identity'] == dict(model_files=settings['model_files'],
            tokenizer_files=[f for f in settings['model_files'] if 'token' in f['name'] or f['name'] == 'special_tokens_map.json'],
            capture_code_sha256=settings['code_sha256']['next_iteration/grounded_graph_features.py'])
        assert receipt['actual_observer_forwards'] == receipt['final_norm_hook_executions'] == 1
        assert receipt['observer_dtype'] == 'torch.bfloat16' and receipt['attention'] == 'sdpa'
        assert receipt['labels_read'] is False and packet['labels_read'] is False
        with np.load(path / 'features' / f"{e['id']}.npz", allow_pickle=False) as arrays:
            assert set(arrays.files) == {'source', 'query'}
            for array_name, rows in (('source', len(nodes)), ('query', len(response['input_ids']))):
                array = arrays[array_name]
                assert array.dtype == np.float32 and array.shape == (rows, 4096)
                assert list(array.shape) == receipt[array_name + '_array_shape']
                assert hashlib.sha256(array.tobytes()).hexdigest() == receipt[array_name + '_array_sha256']
                assert np.isfinite(array).all()
                total_bytes += array.nbytes
                if array_name == 'source':
                    unavailable = [i for i, n in enumerate(nodes) if not n['available']]
                    assert (array[unavailable] == 0).all()
        total_forwards += receipt['actual_observer_forwards']
        source_rows.append(len(nodes)); query_rows.append(len(response['input_ids']))
    assert total_forwards == summary['actual_forwards'] == sum(e['status'] == 'available' for e in entries)
    return dict(summary=summary, manifest_sha256=sha(path / 'manifest.json'), prepare_manifest_sha256=sha(path / 'prepare_manifest.json'),
        code_files=len(settings['code_sha256']), prepare_artifacts=len(prepared['artifacts']), capture_artifacts=len(manifest['artifacts']),
        sources=len({e['source_id'] for e in entries}), task_split_census=dict(Counter(e['task'] + ':' + e['split'] for e in entries)),
        source_nodes=total_nodes, directed_edges=total_edges, unavailable_nodes=unavailable_nodes, all_nonwhitespace_words=total_words,
        source_row_range=[min(source_rows), max(source_rows)], query_row_range=[min(query_rows), max(query_rows)],
        array_bytes=total_bytes, all_arrays_float32_finite_sha_verified=True, prompt_only_source_pool_and_exact_node_edge_order_verified=True,
        exact_original_tokens_offsets_and_pretoken_queries_verified=True, actual_forwards=total_forwards)


def main():
    result = {'reconstruction': reconstruction_check()}
    feature_root = ROOT / 'outputs/grounded_graph_features_v1_20260913'
    settings = read(feature_root / 'settings.json')
    model = Path(settings['model_path'])
    actual_files = [dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p))
        for p in sorted(model.iterdir()) if p.is_file()]
    assert actual_files == settings['model_files']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for name in ('grounded_graph_features_v1_20260913', 'grounded_graph_natural_features_v1_20260913'):
        assert read(ROOT / 'outputs' / name / 'settings.json')['model_files'] == actual_files
        result[name] = feature_check(name, tokenizer)
    reconstruction_entries = read(feature_root / 'entries.json')
    natural_entries = read(ROOT / 'outputs/grounded_graph_natural_features_v1_20260913/entries.json')
    natural_shas = {e['source_text_sha256'] for e in natural_entries}
    result['natural_source_overlap_with_selected_reconstruction'] = [
        {k: e[k] for k in ('source_id', 'task', 'split', 'source_text_sha256')}
        for e in reconstruction_entries if e['source_text_sha256'] in natural_shas]
    result['model_files_verified'] = len(actual_files)
    result['labels_read'] = False
    result['model_forwards_in_independent_audit'] = 0
    result['status'] = 'pass'
    destination = ROOT / 'refine-logs/grounded_graph_feature_independent_audit_20260913.json'
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
