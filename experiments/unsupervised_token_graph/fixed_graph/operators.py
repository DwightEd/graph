"""固定的逐头图变换；不训练，不改原模型，不把图连乘称为因果推理。"""

import numpy as np
from scipy import sparse


FEATURES = (
    'prompt_mass', 'local_mass', 'remote_mass', 'attention_entropy',
    'local_entropy', 'prompt_entropy', 'local_peak', 'copy_mass', 'prompt_peak',
)
VARIANTS = ('nodes', 'graph', 'shuffled', 'one_hop', 'no_future')
BLOCKS = ('current', 'local', 'remote', 'innovation',
          'local_local', 'local_remote', 'remote_local', 'later')


def normalized_entropy(weights, possible_count):
    """概率在当前来源子集内归一化；同时保留该子集原质量。"""
    mass = weights.sum()
    if mass <= 0 or len(weights) < 2:
        return 0.
    probabilities = weights[weights > 0] / mass
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(max(2, possible_count)))


def describe_row(keys, weights, token_ids, prompt_length, target, local_window):
    """target是回答词编号；来源是原输入位置。logits熵不从attention伪造。"""
    prompt = keys < prompt_length
    history = ~prompt
    distance = prompt_length + target - keys
    local = history & (distance <= local_window)
    remote = history & ~local
    copied = history & (token_ids[keys] == token_ids[prompt_length + target])
    local_weights = weights[local]
    local_mass = local_weights.sum()
    peak = local_weights.max() / local_mass if local_mass > 0 else 0.
    return np.array([
        weights[prompt].sum(), local_mass, weights[remote].sum(),
        normalized_entropy(weights, prompt_length + target),
        normalized_entropy(local_weights, min(local_window, target)),
        normalized_entropy(weights[prompt], prompt_length), peak, weights[copied].sum(),
        weights[prompt].max() / weights[prompt].sum() if weights[prompt].sum() > 0 else 0.,
    ], dtype=np.float32)


def channel_arrays(channel, sample, local_window):
    """每个原head只处理一次：属性按预测位置对齐，边保留真实权重。"""
    length = sample.response_length
    features = np.zeros((length, len(FEATURES)), np.float32)
    covered = np.zeros(length, bool)
    masses = np.full(length, np.nan, np.float32)
    targets, sources, values = [], [], []
    for row, query in enumerate(channel.queries):
        target = int(query + 1 - sample.prompt_length)
        if not 0 <= target < length:
            continue
        keys, weights = channel.row(row)
        features[target] = describe_row(keys, weights, sample.token_ids,
                                        sample.prompt_length, target, local_window)
        covered[target] = weights.sum() > 0
        masses[target] = weights.sum()
        history = keys >= sample.prompt_length
        targets.extend([target] * int(history.sum()))
        sources.extend((keys[history] - sample.prompt_length).tolist())
        values.extend(weights[history].tolist())
    history = sparse.csr_matrix((values, (targets, sources)), shape=(length, length))
    return features, covered, masses, history


def shuffle_history(history, response_ids, covered, local_window, random):
    """同query、距离带、复制和可观测类别内置换全部端点，包含零权重位置。"""
    targets, previous = np.tril_indices(history.shape[0], k=-1)
    if not len(previous):
        return history.copy(), 0.
    lag = targets - previous
    bands = np.ceil(np.log2(lag)).astype(int)
    copied = response_ids[previous] == response_ids[targets]
    strata = 8 * bands + 4 * (lag <= local_window) + 2 * copied + covered[previous]
    groups = targets * (int(strata.max()) + 1) + strata

    original_order = np.argsort(groups, kind='stable')
    random_order = np.lexsort((random.random(len(groups)), groups))
    destinations = np.empty(len(previous), dtype=int)
    destinations[original_order] = previous[random_order]

    rows = np.repeat(np.arange(history.shape[0]), np.diff(history.indptr))
    slots = rows * (rows - 1) // 2 + history.indices
    changed_columns = destinations[slots]
    shuffled = sparse.csr_matrix((history.data.copy(), (rows, changed_columns)), shape=history.shape)
    difference = shuffled - history
    changed_mass = float(np.abs(difference.data).sum() / 2)
    return shuffled, changed_mass


def typed_operators(history, covered, local_window):
    """只传已有属性；缺测邻居的质量另报，不重新放大剩余历史权重。"""
    rows = np.repeat(np.arange(history.shape[0]), np.diff(history.indptr))
    columns = history.indices
    usable = covered[rows] & covered[columns]
    local = (rows - columns <= local_window) & usable
    remote = (rows - columns > local_window) & usable
    matrices = []
    for mask in (local, remote):
        matrices.append(sparse.csr_matrix(
            (history.data[mask], (rows[mask], columns[mask])), shape=history.shape))
    dropped = float(history.data[~usable].sum())
    return matrices[0], matrices[1], dropped


def graph_blocks(features, local, remote):
    """关系顺序分别保存；后文分支只是离线分析，不新增反向原生边。"""
    inherited_local = local @ features
    inherited_remote = remote @ features
    history = local + remote
    readers = history.T.tocsr()
    mass = np.asarray(readers.sum(axis=1)).ravel()
    inverse_mass = np.divide(1., mass, out=np.zeros_like(mass), where=mass > 0)
    later = sparse.diags(inverse_mass) @ readers @ features
    return (
        features, inherited_local, inherited_remote,
        features - inherited_local - inherited_remote,
        local @ inherited_local, local @ inherited_remote,
        remote @ inherited_local, np.asarray(later),
    )


def add_projection(output, blocks, identity, seed):
    """固定双哈希CountSketch压缩拼接向量；不同head不先求平均。"""
    for block_index, values in blocks:
        random = np.random.default_rng([seed, *map(int, identity), block_index])
        for _ in range(2):
            buckets = random.integers(output.shape[1], size=values.shape[1])
            signs = random.choice([-1., 1.], size=values.shape[1]) / np.sqrt(2.)
            np.add.at(output.T, buckets, (values * signs).T)


def embed_channel(features, covered, history, response_ids, identity, settings):
    local, remote, dropped = typed_operators(history, covered, settings['local_window'])
    blocks = graph_blocks(features, local, remote)
    modes = {'nodes': (0,), 'graph': tuple(range(8)),
             'one_hop': (0, 1, 2, 3, 7), 'no_future': tuple(range(7))}
    outputs = {}
    for name, selected in modes.items():
        output = np.zeros((len(features), settings['dimensions']), np.float32)
        add_projection(output, [(i, blocks[i]) for i in selected], identity, settings['seed'])
        outputs[name] = output
    random = np.random.default_rng([settings['seed'], *map(int, identity), 999])
    shuffled, moved = shuffle_history(history, response_ids, covered, settings['local_window'], random)
    null_local, null_remote, _ = typed_operators(shuffled, covered, settings['local_window'])
    null_blocks = graph_blocks(features, null_local, null_remote)
    output = np.zeros_like(outputs['graph'])
    add_projection(output, list(enumerate(null_blocks)), identity, settings['seed'])
    outputs['shuffled'] = output
    return outputs, (moved, float(history.data.sum()), dropped)
