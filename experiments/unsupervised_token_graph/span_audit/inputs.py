"""只在输入边界核验身份与token对齐；标签在这里明确用于机制研究。"""

import json
from pathlib import Path

import numpy as np

from ..cache_index import CacheIndex, identity_fields
from ..channels import iter_channels
from ..data import ResponseCache
from ..evaluation_data import EvaluationBinding, read_sources
from .units import Answer, marked_spans, span_mask


def read_annotations(path):
    with Path(path).open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return {str(record['id']): record for record in records}


def group_cache_files(cache, index):
    """支持每答一文件以及有原始ID的逐head文件；不按目录顺序猜身份。"""
    cache = Path(cache)
    paths = [cache] if cache.is_file() else sorted(cache.rglob('*.npz'))
    groups = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            identity, _ = index.resolve(path, identity_fields(archive))
        response_id = str(identity.get('id', path.stem.removeprefix('attention_')))
        groups.setdefault(response_id, []).append(path)
    return groups


def saved_entropy(paths, feature_root, response_id, token_ids, response_length):
    """熵必须已存且逐预测对齐；没有就返回缺测，报告不能声称控制了熵。"""
    path = paths[0]
    if feature_root is not None:
        path = Path(feature_root) / f'{response_id}.npz'
    with np.load(path, allow_pickle=False) as archive:
        if feature_root is not None:
            if not np.array_equal(archive['token_ids'], token_ids):
                raise ValueError(f'{response_id}: entropy cache tokenization differs')
        if 'entropy' not in archive:
            return np.full(response_length, np.nan)
        entropy = archive['entropy'].astype(float)
    if entropy.shape != (response_length,):
        raise ValueError(f'{response_id}: entropy must be aligned [response_length]')
    return entropy


def default_observer_tokenizer(cache):
    """本项目已知的 llama31_8b 目录布局；仍须逐样本核验完整 token ID。"""
    layout = ('data', 'RAGTruth', 'attention', 'llama31_8b')
    for directory in (cache, *cache.parents):
        if directory.parts[-4:] == layout:
            model = directory.parents[3] / 'models' / 'Meta-Llama-3.1-8B-Instruct'
            if model.is_dir():
                return str(model)
    return None


class AuditInputs:
    """复用已有reader，集中放置元数据、tokenizer和文件读取。"""

    def __init__(self, cache, dataset, index_path=None, tokenizer=None, feature_root=None,
                 include_labels=True):
        cache = Path(cache).resolve()
        directory = cache.parent if cache.is_file() else cache
        self.index = CacheIndex(directory, index_path)
        self.reader = ResponseCache(index=self.index)
        self.groups = group_cache_files(cache, self.index)
        self.annotations = read_annotations(Path(dataset) / 'response.jsonl')
        self.sources, _ = read_sources(Path(dataset) / 'response.jsonl')
        alignment_settings = dict(
            cache='/',
            index_files=self.index.inputs,
            tokenizer_fallback=default_observer_tokenizer(cache),
        )
        self.binding = EvaluationBinding(alignment_settings, tokenizer)
        self.reported_tokenizer = None
        self.feature_root = feature_root
        self.include_labels = include_labels

    def selected_ids(self, split, tasks):
        for response_id in self.groups:
            annotation = self.annotations[response_id]
            source = self.sources[str(annotation['source_id'])]
            task = source.get('task_type', source.get('task'))
            if annotation['split'] == split and (not tasks or task in tasks):
                yield response_id

    def load_answer(self, response_id):
        """先读身份与文字；没有匹配片段时不解压全部attention。"""
        paths = self.groups[response_id]
        with np.load(paths[0], allow_pickle=False) as archive:
            identity, _ = self.index.resolve(paths[0], identity_fields(archive))
        token_ids = np.asarray(identity['token_ids'])
        prompt_length = int(identity['prompt_length'])
        arrays = dict(token_ids=token_ids, prompt_length=prompt_length)
        if 'offsets' in identity:
            arrays['offsets'] = np.asarray(identity['offsets'])
        metadata = {key: value for key, value in identity.items()
                    if key not in ('token_ids', 'offsets', 'prompt_length')}
        metadata.update(id=response_id, file=paths[0].name, cache=str(paths[0].resolve()))
        annotation = self.annotations[response_id]
        metadata, offsets = self.binding.bind(metadata, annotation, arrays, self.sources)
        self.report_alignment(metadata)
        return self.make_answer(metadata, offsets, annotation, token_ids, prompt_length, paths)

    def report_alignment(self, metadata):
        tokenizer = metadata.get('verified_tokenizer')
        if tokenizer and tokenizer != self.reported_tokenizer:
            print(f'Token alignment verified with local tokenizer: {tokenizer}', flush=True)
            self.reported_tokenizer = tokenizer

    def make_answer(self, metadata, offsets, annotation, token_ids, prompt_length, paths):
        spans = marked_spans(offsets, annotation['labels']) if self.include_labels else []
        response_length = len(token_ids) - prompt_length
        entropy = saved_entropy(paths, self.feature_root, metadata['id'],
                                token_ids, response_length)
        return Answer(
            response_id=metadata['id'], source_id=metadata['source_id'],
            task=metadata['task'], generator=metadata['generator'], split=metadata['split'],
            text=annotation['response'], token_ids=token_ids,
            prompt_length=prompt_length, offsets=offsets,
            error_mask=span_mask(response_length, spans) if self.include_labels else None, spans=spans,
            entropy=entropy, cache_paths=paths,
        )

    def channels(self, answer, layers=None, heads=None):
        seen = set()
        for path in answer.cache_paths:
            record = self.reader.load(path)
            if not np.array_equal(record.token_ids, answer.token_ids):
                raise ValueError('Per-head files do not share the same original tokens')
            for channel in iter_channels(record, layers, heads):
                identity = (channel.layer, channel.head)
                if identity in seen:
                    raise ValueError(f'Duplicate physical channel: {identity}')
                seen.add(identity)
                yield channel
