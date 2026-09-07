"""All tasks/splits through real tiny-Llama capture; labels stay post-hoc."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.attention_rhythm_run import parser, run
from experiments.reanchor_flow.tests.test_scan_dataset import _capture


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '|' + messages[1]['content'] + '|'

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(range(1, len(text)+1)),
                    offset_mapping=[(i, i+1) for i in range(len(text))])

    def decode(self, ids):
        return f't{ids[0]}'


def test_all_tasks_resume_offline_analysis_and_cache_override(tmp_path, monkeypatch):
    transformers = pytest.importorskip('transformers')
    import experiments.reanchor_flow.scan_dataset as scan_module

    scans, output, relocated = tmp_path / 'scans', tmp_path / 'out', tmp_path / 'relocated'
    scans.mkdir()
    prompt = "{'x': 1}"
    p = len(prompt) + 2
    rows = []
    tasks = ('QA', 'Summary', 'Data2txt')
    for split in ('train', 'test'):
        records = {f'{split}-{task}': (f'source-{split}-{task}', task) for task in tasks}
        root = _capture(scans, records=records, overrides={
            'response_start': p, 'sequence_length': p+4, 'route_row_position': np.arange(p-1, p+3),
            'token_ids': np.arange(1, p+5)})
        if split == 'train':
            root.rename(scans / split)
        root = scans / split
        manifest = json.loads((root / 'run_manifest.json').read_text())
        manifest['config']['split'] = split
        (root / 'run_manifest.json').write_text(json.dumps(manifest))
        for _, (source, task) in records.items():
            rows.append(dict(source_id=source, task_type=task, prompt=prompt,
                             source_info={'passages': prompt} if task == 'QA' else prompt))
    source_file = tmp_path / 'source.jsonl'
    source_file.write_text('\n'.join(map(json.dumps, rows)))
    cfg = transformers.LlamaConfig(vocab_size=61, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(19)
    model = transformers.LlamaForCausalLM(cfg).eval()
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: CharacterTokenizer())
    model_loads, label_loads = [], []
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained',
                        lambda *a, **kw: model_loads.append(True) or model)

    class LabelStore:
        def __init__(self, dataset, dataset_root=None):
            assert Path(dataset_root) == relocated / dataset.split
            assert len(json.loads((output / 'index.json').read_text())['samples']) == 6
            assert len(list(output.rglob('*.audit.npz'))) == 6

        def load(self, scan):
            label_loads.append(scan.sample_id)
            return np.array([0, 0, 1, -1])

    monkeypatch.setattr(scan_module, 'ScanLabelStore', LabelStore)
    argv = ['--split', 'all', '--task', 'all', '--samples-per-task', '0', '--scans', str(scans),
            '--cache', str(relocated), '--source-info', str(source_file), '--output', str(output),
            '--plots-per-task', '0', '--bootstrap', '2', '--device', 'cpu', '--dtype', 'float32',
            '--future-lo', '1', '--future-hi', '2', '--evaluate']
    report = run(parser().parse_args(argv))
    assert len(model_loads) == 1 and len(label_loads) == 6
    assert len(report['groups']) == 8
    assert report['groups']['test/ALL']['response_tokens'] == 12
    assert report['groups']['test/ALL']['hallucinated_tokens'] == 3
    assert not report['detection_metrics_run']
    assert (output / 'gallery.html').exists()
    with np.load(output / 'test' / 'QA' / 'test-QA.npz') as stored:
        assert stored['waad'].shape == (2, 4, 5)
        assert not any('label' in key for key in stored.files if key != 'labels_used_for_capture')

    def forbidden(*args, **kwargs):
        raise AssertionError('completed capture/analysis must not load LLM weights or labels again')

    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', forbidden)
    monkeypatch.setattr(scan_module, 'ScanLabelStore', forbidden)
    manifest = run(parser().parse_args([*argv, '--phase', 'capture']))
    assert all(entry['resumed'] for entry in manifest['samples'])
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', forbidden)
    report = run(parser().parse_args(['--phase', 'analyze', '--output', str(output), '--evaluate', '--bootstrap', '0']))
    assert report['groups']['train/ALL']['hallucinated_tokens'] == 3
