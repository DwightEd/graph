"""裸缓存恢复字符对齐；只用本地测试tokenizer，不加载模型或修改原数据。"""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.unsupervised_token_graph.evaluation_data import (
    EvaluationBinding, resolve_tokenizer, verified_offsets,
)
from experiments.unsupervised_token_graph.span_audit.inputs import (
    AuditInputs, default_observer_tokenizer,
)
from experiments.unsupervised_token_graph.span_audit.run import main
from experiments.unsupervised_token_graph.span_audit.tests.test_audit import (
    CharacterTokenizer, write_dataset,
)


@pytest.fixture(autouse=True)
def no_environment_tokenizer(monkeypatch):
    monkeypatch.delenv('TOKENIZER', raising=False)


def deployed_layout(root):
    cache, dataset = write_dataset(root / 'fixture', 'canonical', metadata=False)
    target = root / 'user/data/RAGTruth/attention/llama31_8b/train'
    target.parent.mkdir(parents=True)
    shutil.move(str(cache), str(target))
    model = root / 'user/models/Meta-Llama-3.1-8B-Instruct'
    return target, dataset, model


def local_tokenizer_loader(monkeypatch, tokenizer=None):
    loaded = []
    tokenizer = tokenizer or CharacterTokenizer()

    class LocalAutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, use_fast, local_files_only):
            assert use_fast and local_files_only
            assert Path(path).is_dir()
            loaded.append(str(path))
            return tokenizer

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=LocalAutoTokenizer))
    return loaded


@pytest.mark.parametrize('ending', ['train', 'test', 'train/10011.npz'])
def test_known_layout_locates_original_observer_not_generator(tmp_path, ending):
    model = tmp_path / 'user/models/Meta-Llama-3.1-8B-Instruct'
    model.mkdir(parents=True)
    cache = tmp_path / 'user/data/RAGTruth/attention/llama31_8b' / ending
    assert default_observer_tokenizer(cache) == str(model)


def test_other_layout_and_missing_model_are_not_guessed(tmp_path):
    cache = tmp_path / 'user/data/RAGTruth/attention/llama31_8b/train'
    assert default_observer_tokenizer(cache) is None
    model = tmp_path / 'user/models/Meta-Llama-3.1-8B-Instruct'
    model.mkdir(parents=True)
    other = tmp_path / 'user/data/RAGTruth/attention/llama2/train'
    assert default_observer_tokenizer(other) is None
    assert resolve_tokenizer({}, {'model': str(model), 'generator': str(model)}) is None


def test_resolution_prefers_explicit_then_environment_then_manifest(tmp_path, monkeypatch):
    cache, _, fallback = deployed_layout(tmp_path)
    paths = [tmp_path / name for name in ('explicit', 'environment', 'declared')]
    for path in paths + [fallback]:
        path.mkdir(parents=True)
    (cache.parent / 'manifest.json').write_text(json.dumps({'tokenizer_path': str(paths[2])}))
    settings = {'cache': '/', 'tokenizer_fallback': str(fallback)}
    record = {'cache': str(cache / '10011.npz')}
    monkeypatch.setenv('TOKENIZER', str(paths[1]))
    assert resolve_tokenizer(settings, record, str(paths[0])) == str(paths[0])
    assert resolve_tokenizer(settings, record) == str(paths[1])
    monkeypatch.delenv('TOKENIZER')
    assert resolve_tokenizer(settings, record) == str(paths[2])
    (cache.parent / 'manifest.json').unlink()
    assert resolve_tokenizer(settings, record) == str(fallback)


def test_existing_index_settings_are_passed_to_alignment(tmp_path, monkeypatch):
    cache, dataset, fallback = deployed_layout(tmp_path)
    fallback.mkdir(parents=True)
    index_dir = tmp_path / 'existing_population'
    index_dir.mkdir()
    index = index_dir / 'inputs.jsonl'
    index.write_text(json.dumps({'id': '1', 'source_id': 'source1', 'model': 'llama-2-7b-chat'}))
    observer = tmp_path / 'explicit_observer'
    observer.mkdir()
    (index_dir / 'settings.json').write_text(json.dumps({'model': str(observer)}))
    loaded = local_tokenizer_loader(monkeypatch)
    inputs = AuditInputs(cache, dataset, index_path=index)
    answer = inputs.load_answer('1')
    assert loaded == [str(observer)]
    assert answer.generator == 'llama-2-7b-chat'
    assert answer.offsets.shape == (52, 2)


def test_failed_run_resumes_unchanged_settings_and_inputs(tmp_path, monkeypatch, capsys):
    cache, dataset, model = deployed_layout(tmp_path)
    output = tmp_path / 'audit'
    args = ['--cache', str(cache), '--dataset', str(dataset), '--output', str(output),
            '--window', '3', '--bootstrap', '0']
    cache_file = cache / 'attention_1.npz'
    before_cache = cache_file.read_bytes()
    before_annotations = (dataset / 'response.jsonl').read_bytes()

    with pytest.raises(ValueError, match='missing offsets') as error:
        main(args)
    assert 'Scores are saved' not in str(error.value)
    assert not list((output / 'samples').glob('*.npz'))
    before_settings = (output / 'settings.json').read_bytes()

    model.mkdir(parents=True)
    loaded = local_tokenizer_loader(monkeypatch)
    main(args + ['--resume'])
    assert loaded == [str(model)]
    assert str(model) in capsys.readouterr().out
    assert (output / 'settings.json').read_bytes() == before_settings
    assert cache_file.read_bytes() == before_cache
    assert (dataset / 'response.jsonl').read_bytes() == before_annotations
    assert json.loads((output / 'summary.json').read_text())['matched_pairs'] == 1

    saved = (output / 'samples/1.npz').read_bytes()
    main(args + ['--resume'])
    assert loaded == [str(model)]
    assert (output / 'samples/1.npz').read_bytes() == saved


def test_environment_fixes_custom_cache_without_changing_cli_settings(tmp_path, monkeypatch):
    cache, dataset = write_dataset(tmp_path, 'canonical', metadata=False)
    model = tmp_path / 'observer'
    model.mkdir()
    monkeypatch.setenv('TOKENIZER', str(model))
    loaded = local_tokenizer_loader(monkeypatch)
    output = tmp_path / 'out'
    main(['--cache', str(cache), '--dataset', str(dataset), '--output', str(output),
          '--window', '3', '--bootstrap', '0'])
    assert loaded == [str(model)]
    assert json.loads((output / 'settings.json').read_text())['tokenizer'] is None


class WrongTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        result['input_ids'] = [token + 1 for token in result['input_ids']]
        return result

    def decode(self, tokens, **kwargs):
        return ''.join(chr(token) for token in tokens)


def test_found_directory_never_bypasses_token_id_verification(tmp_path, monkeypatch):
    cache, dataset, model = deployed_layout(tmp_path)
    model.mkdir(parents=True)
    local_tokenizer_loader(monkeypatch, WrongTokenizer())
    inputs = AuditInputs(cache, dataset)
    before = (cache / 'attention_1.npz').read_bytes()
    with pytest.raises(ValueError, match='saved token_ids do not match'):
        inputs.load_answer('1')
    assert (cache / 'attention_1.npz').read_bytes() == before


def test_unicode_offsets_and_special_suffix_keep_exact_alignment():
    tokenizer = CharacterTokenizer()
    tokenizer.all_special_ids = [999999]
    text = '甲 café🙂'
    ids = np.r_[1000, [ord(char) for char in text], 999999]
    offsets = verified_offsets(tokenizer, ids, 1, text)
    np.testing.assert_array_equal(offsets[:-1], [[i, i + 1] for i in range(len(text))])
    np.testing.assert_array_equal(offsets[-1], [len(text), len(text)])
