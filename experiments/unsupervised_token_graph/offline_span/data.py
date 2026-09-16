"""复用现有 NPZ/索引；样本、训练图中没有幻觉标签。"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ..cache_index import CacheIndex, identity_fields
from ..data import ResponseCache
from .metadata import find_response_file, merge_metadata, needs_metadata, read_metadata


@dataclass
class Sample:
    response_id: str
    source_id: str
    task: str
    generator: str
    split: str
    cache_files: list[str]
    token_ids: np.ndarray
    prompt_length: int
    offsets: np.ndarray
    response_sha256: str = ""
    metadata_file: str = ""

    @property
    def response_length(self):
        return len(self.token_ids) - self.prompt_length

    def metadata(self):
        values = asdict(self)
        del values['token_ids']
        del values['offsets']
        return values


@dataclass
class TokenGraph:
    sample: Sample
    channels: np.ndarray                 # [channel, (layer, head)]
    edges: np.ndarray                    # [edge, (channel, source_token, query_token)]
    weights: np.ndarray                  # retained attention; never locally normalized
    masses: np.ndarray                   # [channel, prediction, prompt/history/missing/pruned]
    coverage: np.ndarray                 # [response_prediction]
    source_groups: np.ndarray            # [prompt_token]; -1 means outside the supplied source mask
    node_features: np.ndarray            # [all_tokens, hidden_dim], or [all_tokens, 0]
    entropy: np.ndarray                  # [response_prediction], NaN when unavailable
    feature_mode: str


@dataclass
class SpanView:
    start: int
    end: int
    evidence_nodes: np.ndarray
    selector_nodes: np.ndarray
    response_nodes: np.ndarray
    later_nodes: np.ndarray
    evidence_group: int


@dataclass
class MatchedPair:
    observed: SpanView
    reconnected: SpanView
    control_kind: str                     # loss bookkeeping ONLY; never a model input


@dataclass
class DetectionResult:
    response_id: str
    token_scores: np.ndarray
    span_bounds: np.ndarray
    span_scores: np.ndarray
    covered_tokens: np.ndarray


def write_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def load_samples(cache_root, index_path=None, split=None, tasks=(), generators=(),
                 dataset=None, source_info=None):
    """先读缓存/索引，缺失身份再按回答编号关联原始 RAGTruth 元数据。"""
    root = Path(cache_root)
    directory = root.parent if root.is_file() else root
    index = CacheIndex(directory, index_path)
    paths = [root] if root.is_file() else sorted(root.rglob('*.npz'))
    samples = {}
    metadata = None
    metadata_file = None

    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            native = identity_fields(archive)
        identity, _ = index.resolve(path, native)
        sample_id = identity.get('id', path.stem.removeprefix('attention_'))
        used_metadata = ""
        if needs_metadata(identity):
            if metadata is None:
                metadata_file = find_response_file(root, dataset)
                metadata = {}
                if metadata_file is not None:
                    metadata = read_metadata(metadata_file, source_info)
            if str(sample_id) in metadata:
                identity = merge_metadata(identity, sample_id, metadata)
                used_metadata = str(metadata_file)

        sample_split = identity.get('split') or split
        task = identity.get('task', 'unknown')
        generator = identity.get('generator', 'unknown')
        if tasks and task not in tasks:
            continue
        if generators and generator not in generators:
            continue
        if split and sample_split != split:
            raise ValueError(f'{path}: cached split disagrees with --split={split}')
        if 'token_ids' not in identity:
            raise ValueError(f'{path}: token_ids missing; point --index to the EXISTING inputs/records index')

        token_ids = np.asarray(identity['token_ids'], dtype=np.int64)
        prompt_length = int(identity['prompt_length'])
        offsets = np.asarray(identity.get('offsets', []), dtype=np.int64).reshape(-1, 2)
        sample = Sample(str(sample_id), identity.get('source_id', ''), task, generator,
                        sample_split or '', [str(path.resolve())], token_ids,
                        prompt_length, offsets, identity.get('response_sha256', ''), used_metadata)
        if sample_id in samples:
            previous = samples[sample_id]
            if (previous.source_id != sample.source_id or previous.prompt_length != prompt_length
                    or not np.array_equal(previous.token_ids, token_ids)):
                raise ValueError(f'{sample_id}: per-head files are not the same answer')
            previous.cache_files.extend(sample.cache_files)
        else:
            samples[sample_id] = sample

    if not samples:
        raise ValueError(f'No matching attention NPZs in {root}')
    return list(samples.values()), index


def load_observations(sample, index, feature_mode='tokens', hidden_layer=None, feature_root=None):
    """现有 reader 解码 attention；可选 hidden 必须来自明确字段和同一 tokenization。"""
    reader = ResponseCache(index=index)
    records = [reader.load(path) for path in sample.cache_files]
    count = len(sample.token_ids)
    observations = dict(records=records, node_features=np.empty((count, 0), np.float32),
                        entropy=np.full(sample.response_length, np.nan, np.float32),
                        source_groups=None, source_mask=None, hidden_fields=[])
    feature_paths = list(sample.cache_files[:1])
    if feature_root is not None:
        feature_paths = [str(Path(feature_root) / f'{sample.response_id}.npz')]

    with np.load(feature_paths[0], allow_pickle=False) as archive:
        observations['hidden_fields'] = [
            key for key in ('hidden', 'hidden_states') if key in archive.files
        ]
        if feature_root is not None:
            if not np.array_equal(archive['token_ids'], sample.token_ids):
                raise ValueError(f'{sample.response_id}: feature token IDs differ from attention')
        for name in ('source_groups', 'source_mask'):
            if name in archive:
                observations[name] = archive[name].copy()
        entropy_key = next((key for key in ('entropy', 'next_token_entropy') if key in archive), None)
        if entropy_key is not None:
            entropy = archive[entropy_key]
            if entropy.shape != (sample.response_length,):
                raise ValueError('entropy must be [response_predictions], aligned before each output token')
            observations['entropy'] = entropy.astype(np.float32)
        if feature_mode == 'hidden':
            hidden_key = next((key for key in ('hidden_states', 'hidden') if key in archive), None)
            if hidden_key is None:
                raise ValueError('hidden mode needs saved hidden/hidden_states; tokens mode is explicit, not a silent fallback')
            hidden = archive[hidden_key]
            if hidden.ndim == 3:
                if hidden_layer is None:
                    raise ValueError('hidden [layers,tokens,dim] requires --hidden-layer (saved array index)')
                hidden = hidden[hidden_layer]
            if hidden.ndim != 2 or hidden.shape[0] != count or not np.isfinite(hidden).all():
                raise ValueError('hidden must cover ALL prompt+response token positions, [N,D]')
            observations['node_features'] = hidden.astype(np.float32)
    return observations


def save_graph(graph, path):
    partial = Path(path).with_suffix('.partial.npz')
    np.savez_compressed(partial, record_json=json.dumps(graph.sample.metadata(), ensure_ascii=False),
                        token_ids=graph.sample.token_ids, offsets=graph.sample.offsets,
                        channels=graph.channels, edges=graph.edges, weights=graph.weights,
                        masses=graph.masses, coverage=graph.coverage, source_groups=graph.source_groups,
                        node_features=graph.node_features, entropy=graph.entropy, feature_mode=graph.feature_mode)
    partial.replace(path)


def load_graph(path):
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive['record_json']))
        sample = Sample(**record, token_ids=archive['token_ids'], offsets=archive['offsets'])
        return TokenGraph(sample, archive['channels'], archive['edges'], archive['weights'],
                          archive['masses'], archive['coverage'], archive['source_groups'],
                          archive['node_features'], archive['entropy'], str(archive['feature_mode']))
