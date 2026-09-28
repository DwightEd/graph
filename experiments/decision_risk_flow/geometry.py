"""Fixed kernel summaries and signed window-message graph diagnostics."""
import numpy as np
from scipy.sparse.csgraph import connected_components

from .kernel import GROUPS


def effective_rank(vectors):
    values = np.linalg.eigvalsh(vectors.T @ vectors).clip(0)
    if values.sum() < 1e-20:
        return 0.
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities.clip(1e-30)))))


def graph_statistics(vectors, groups):
    """A functional similarity graph, not directed computational dependencies."""
    energy = np.sum(vectors * vectors, -1)
    selected = []
    for group in range(len(GROUPS)):
        indices = np.flatnonzero((groups == group) & (energy > 1e-16))
        selected.extend(indices[np.argsort(-energy[indices], kind='stable')[:16]])
    selected = np.asarray(selected, dtype=int)
    if not len(selected):
        return np.zeros(9), selected
    units = vectors[selected] / np.sqrt(energy[selected, None])
    cosine = np.clip(units @ units.T, -1, 1)
    np.fill_diagonal(cosine, 0)
    adjacency = np.maximum(cosine - .5, 0)
    degree = adjacency.sum(1)
    _, components = connected_components(adjacency, directed=False)
    sizes = np.bincount(components)
    prompt = groups[selected] == 0
    history = np.isin(groups[selected], [1, 2])
    cross = adjacency[np.ix_(prompt, history)].sum()
    denominator = max(len(selected) * (len(selected) - 1), 1)
    values = [len(selected), energy[selected].sum() / max(energy.sum(), 1e-20),
        len(sizes) / len(selected), sizes.max() / len(selected),
        np.sum(cosine < -.5) / denominator, np.sum(cosine > .5) / denominator,
        cross / max(degree[prompt].sum(), 1e-20),
        cross / max(degree[history].sum(), 1e-20),
        degree.std() / max(degree.mean(), 1e-20)]
    return np.asarray(values), selected


GRAPH_NAMES = ('nodes', 'energy_coverage', 'component_fraction', 'largest_component',
               'negative_edges', 'positive_edges', 'prompt_cut', 'history_cut', 'degree_cv')


def describe(choice, sketch, attention, windows):
    """All group energies use every window; graph truncation has explicit coverage."""
    features = {}
    group_ids = np.r_[np.repeat(np.arange(4), windows), 4]
    for band in range(2):
        summed = []
        for index, group in enumerate(GROUPS):
            mask = group_ids == index
            vector = sketch[band, mask].astype(np.float64)
            values = choice[band, mask]
            prefix = f'b{band}_{group}'
            features['attention_' + prefix] = float(attention[band, mask].sum())
            features['choice_positive_' + prefix] = float(np.linalg.norm(np.maximum(values, 0)))
            features['choice_negative_' + prefix] = float(np.linalg.norm(np.minimum(values, 0)))
            features['choice_net_' + prefix] = float(values.sum())
            features['kernel_energy_' + prefix] = float(np.sum(vector * vector))
            features['kernel_coherent_' + prefix] = float(np.sum(vector.sum(0) ** 2))
            features['kernel_rank_' + prefix] = effective_rank(vector)
            summed.append(vector.sum(0))
        for left, right in ((0, 1), (0, 2), (0, 4), (1, 4), (2, 4)):
            denominator = np.linalg.norm(summed[left]) * np.linalg.norm(summed[right])
            features[f'kernel_alignment_b{band}_{GROUPS[left]}_{GROUPS[right]}'] = float(
                np.dot(summed[left], summed[right]) / max(denominator, 1e-20))
    vectors = sketch.reshape(-1, sketch.shape[-1]).astype(np.float64)
    groups = np.tile(group_ids, 2)
    geometry, selected = graph_statistics(vectors, groups)
    features.update({'graph_' + key: float(value) for key, value in zip(GRAPH_NAMES, geometry)})
    # Preserve each node's energy; destroy its direction with fixed independent signs.
    rng = np.random.default_rng(1729)
    shuffled = vectors * rng.choice([-1., 1.], size=vectors.shape)
    geometry, _ = graph_statistics(shuffled, groups)
    features.update({'shuffled_' + key: float(value) for key, value in zip(GRAPH_NAMES, geometry)})
    return features, selected
