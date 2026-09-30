"""Token-wise evidence contrasts; no neighbouring-token risk aggregation."""

import numpy as np

from experiments.native_support.unified.calibration import transform


METHODS = ('token_pair', 'odds_pair', 'odds_full')
PRIMARY = 'odds_full'
PROBABILITY_RESOLUTION = np.finfo(np.float32).eps


def log_odds(logp):
    """Regularize the complementary mass at the cached FP32 resolution.

    Cached log probabilities can round to zero. This is a resolution-limited
    contrast, not exact log odds when the complementary mass is unresolved.
    """
    logp = np.minimum(np.asarray(logp, dtype=float), 0.)
    complement = np.maximum(-np.expm1(logp), PROBABILITY_RESOLUTION)
    return logp - np.log(complement)


def contrasts(present_full, absent_full, present_local, absent_local, route):
    full = log_odds(absent_full) - log_odds(present_full)
    local = log_odds(absent_local) - log_odds(present_local)
    return dict(token_pair=.5 * (absent_full - present_full + absent_local - present_local),
                odds_pair=.5 * (full + local), odds_full=full, route=route)


def pack_contrasts(pack, metadata):
    observed = dict(zip(metadata['observations'], pack['observations'].T))
    context = dict(zip(metadata['context'], pack['context'].T))
    present_full = observed['with_full_logp']
    present_local = observed['with_local_logp']
    # Recover the original scalar measurements from the lossless pack layout.
    absent_full = present_full + observed['full_deviation'] + context['full_unit']
    absent_local = present_local + observed['local_deviation'] + context['local_unit']
    return contrasts(present_full, absent_full, present_local, absent_local, observed['route'])


def score_contrasts(values, scales):
    # Preserve the historical 3:1 source/route weight, without its window or
    # unit-mean broadcast. Ranks are reference percentiles, not error chances.
    route_rank = transform(values['route'], scales['route'])
    return {name: .75 * transform(values[name], scales[name]) + .25 * route_rank
            for name in METHODS}


GRAPH_METHODS = ('node_ridge', 'node_iforest', 'graph_ridge', 'temporal_ridge',
                 'shuffled_17', 'shuffled_29', 'shuffled_43')
HIDDEN_WIDTH = 32
MASS_METHODS = ('graph_mass', 'temporal_mass', 'shuffled_mass_17',
                'shuffled_mass_29', 'shuffled_mass_43')


def history_weights(root_effect, prompt_length):
    """Signed total root responses, not direct neural edges for path rollout."""
    count = len(root_effect)
    weights = np.zeros((count, count), dtype=float)
    weights[:, :count - 1] = root_effect[:, prompt_length:]
    return np.tril(weights, k=-1)


def matched_history(weights, token_ids, seed=None):
    """Preserve each receiver's signed weights within lag/repeated-ID strata.

    None returns the conditional expectation. Randomization preserves row
    strata and weight multisets, not each sender's outgoing degree.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros_like(weights)
    for target in range(1, len(weights)):
        parents = np.arange(target)
        lag = np.floor(np.log2(target - parents)).astype(int)
        repeated = token_ids[:target] == token_ids[target]
        strata = 2 * lag + repeated
        for group in np.unique(strata):
            indices = parents[strata == group]
            values = weights[target, indices]
            result[target, indices] = values.mean() if seed is None else rng.permutation(values)
    return result


def signed_context(weights, attributes, expected=False, token_ids=None, preserve_mass=False):
    """One-hop feature contexts; no recursive multiplication of total effects."""
    contexts = []
    for channel in (np.maximum(weights, 0), np.maximum(-weights, 0)):
        if expected:
            channel = matched_history(channel, token_ids)
        if not preserve_mass:
            mass = channel.sum(-1, keepdims=True)
            channel = np.divide(channel, mass, out=np.zeros_like(channel), where=mass > 0)
        contexts.append(channel @ attributes)
    return np.column_stack(contexts)


def graph_attributes(row, trace, hidden):
    """Fixed random hidden projection plus six native, label-free attributes."""
    rng = np.random.default_rng(17)
    projection = rng.standard_normal((hidden.shape[1], HIDDEN_WIDTH)) / np.sqrt(HIDDEN_WIDTH)
    projected = hidden.astype(float) @ projection
    source = np.asarray(row['source']['source_mask'], dtype=bool)
    effect = trace['root_effect'].astype(float)
    total = np.maximum(np.abs(effect).sum(-1), 1e-30)
    source_effect = effect[:, :len(source)][:, source]
    scalars = np.column_stack((-trace['logp'], trace['entropy'], -trace['margin'],
        np.log1p(np.linalg.norm(trace['root_norm'].astype(float), axis=1)),
        np.maximum(source_effect, 0).sum(-1) / total,
        np.maximum(-source_effect, 0).sum(-1) / total))
    return np.column_stack((projected, scalars))


def graph_covariates(row, weights):
    """Same position, lexical-format and signed-mass controls in every model."""
    count = len(weights)
    position = np.arange(count)
    texts = row['response']['token_text']
    lexical = np.asarray([[any(c.isalpha() for c in word), any(c.isdigit() for c in word),
        not any(c.isalnum() for c in word), len(word)] for word in texts], dtype=float)
    task = np.tile([row['task'] == name for name in ('QA', 'Summary', 'Data2txt')], (count, 1))
    return np.column_stack((position / max(count - 1, 1), np.log1p(position),
        np.full(count, np.log1p(count)), np.full(count, np.log1p(len(row['prompt']))),
        lexical, task, np.log1p(np.maximum(weights, 0).sum(-1)),
        np.log1p(np.maximum(-weights, 0).sum(-1))))


def graph_designs(row, trace, hidden, preserve_mass=False):
    attributes = graph_attributes(row, trace, hidden)
    weights = history_weights(trace['root_effect'], len(row['prompt']))
    controls = graph_covariates(row, weights)
    token_ids = np.asarray(row['response']['answer_ids'])
    suffix = 'mass' if preserve_mass else 'ridge'
    designs = dict(node_ridge=controls)
    designs['graph_' + suffix] = np.column_stack(
        (controls, signed_context(weights, attributes, preserve_mass=preserve_mass)))
    designs['temporal_' + suffix] = np.column_stack((controls,
        signed_context(weights, attributes, expected=True, token_ids=token_ids,
                       preserve_mass=preserve_mass)))
    for seed in (17, 29, 43):
        shuffled = matched_history(weights, token_ids, seed)
        name = f'shuffled_mass_{seed}' if preserve_mass else f'shuffled_{seed}'
        designs[name] = np.column_stack((controls,
            signed_context(shuffled, attributes, preserve_mass=preserve_mass)))
    return attributes, designs, weights


def weighted_standardize(train, evaluation, weights):
    mean = np.average(train, axis=0, weights=weights)
    scale = np.sqrt(np.average((train - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-8, scale, 1.)
    return (train - mean) / scale, (evaluation - mean) / scale


def graph_reconstruction(train, evaluation, method):
    """Fit one source-excluded model. No gold labels or score direction search."""
    from sklearn.ensemble import IsolationForest

    weights = np.concatenate([np.full(row['valid'].sum(), 1 / row['valid'].sum()) for row in train])
    weights /= weights.sum()
    target = np.concatenate([row['attributes'][row['valid']] for row in train])
    target, observed = weighted_standardize(target, evaluation['attributes'], weights)
    if method == 'node_iforest':
        # Source-balanced resampling avoids giving long answers more fit weight.
        rng = np.random.default_rng(17)
        indices = rng.choice(len(target), len(target), p=weights)
        model = IsolationForest(n_estimators=100, max_samples=min(256, len(target)),
                                random_state=17, n_jobs=1).fit(target[indices])
        return -model.score_samples(observed)
    predictors = np.concatenate([row['designs'][method][row['valid']] for row in train])
    predictors, query = weighted_standardize(predictors, evaluation['designs'][method], weights)
    predictors = np.column_stack((np.ones(len(predictors)), predictors))
    query = np.column_stack((np.ones(len(query)), query))
    penalty = np.eye(predictors.shape[1])
    penalty[0, 0] = 0.
    coefficients = np.linalg.solve(predictors.T @ (weights[:, None] * predictors) + penalty,
                                   predictors.T @ (weights[:, None] * target))
    error = (observed - query @ coefficients) ** 2
    return .5 * error[:, :HIDDEN_WIDTH].mean(-1) + .5 * error[:, HIDDEN_WIDTH:].mean(-1)


def crossfit_graph(records, method):
    """Outer source exclusion plus inner held-source calibration predictions."""
    from experiments.native_support.unified.calibration import fit_distribution

    result, folds = {}, {}
    cache = {}
    for held in records:
        training = [row for row in records if row['source_id'] != held['source_id']]
        calibration, sources = [], []
        for inner in training:
            fitted = [row for row in training if row['source_id'] != inner['source_id']]
            identity = (tuple(row['key'] for row in fitted), inner['key'])
            if identity not in cache:
                cache[identity] = graph_reconstruction(fitted, inner, method)
            values = cache[identity][inner['valid']]
            calibration.append(values)
            sources.extend([inner['source_id']] * len(values))
        reference = fit_distribution(np.concatenate(calibration), sources)
        raw = graph_reconstruction(training, held, method)
        result[held['key']] = dict(raw=raw, score=transform(raw, reference))
        folds[held['key']] = dict(held_source=held['source_id'],
            fit_sources=[row['source_id'] for row in training],
            calibration_sources=sorted(set(sources)),
            values=reference['values'].tolist(), cumulative=reference['cumulative'].tolist())
    return result, folds
