"""Small explanation models predict CHARM logits, not hallucination labels."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = ("context", "context_entropy", "channel_marginals", "with_agreement")


def training_rows(samples, ids, tokens_per_answer=64):
    """Deterministic uniform subsampling, without examining any token labels."""
    features, logits, weights = [], [], []
    for sample in samples:
        if sample["id"] not in ids:
            continue
        count = len(sample["logits"])
        positions = np.unique(np.linspace(0, count - 1, min(count, tokens_per_answer), dtype=int))
        features.append(sample["features"][positions])
        logits.append(sample["logits"][positions])
        weights.append(np.full(len(positions), 1 / len(positions)))
    return np.concatenate(features), np.concatenate(logits), np.concatenate(weights)


def select_columns(features, logits, candidates, maximum=64):
    """Select on fit logits only. This does NOT control all 1024 head entropies."""
    values = features[:, candidates].astype(float)
    values -= values.mean(axis=0)
    centered = logits - logits.mean()
    scale = np.sqrt((values ** 2).sum(axis=0) * (centered ** 2).sum())
    correlation = abs(values.T @ centered) / np.maximum(scale, 1e-12)
    required = np.flatnonzero(np.asarray(candidates) < 9)
    optional = np.flatnonzero(np.asarray(candidates) >= 9)
    ranked = optional[np.argsort(-correlation[optional], kind="stable")]
    selected = np.r_[required, ranked[:max(0, maximum - len(required))]]
    return np.asarray(candidates)[selected], correlation


def fit_explanations(samples, partitions, columns, maximum=64):
    """Choose linear/nonlinear surrogates by held-out selection logit MSE only."""
    fit_x, fit_y, fit_weight = training_rows(samples, set(partitions["fit"]))
    sel_x, sel_y, sel_weight = training_rows(samples, set(partitions["select"]))
    models, summaries = {}, {}
    for name in FEATURE_SETS:
        selected, correlation = select_columns(fit_x, fit_y, columns[name], maximum)
        candidates = {
            "ridge_1": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
            "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100.)),
            "small_tree": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=10, early_stopping=False, random_state=0)}
        errors = {}
        for key, model in candidates.items():
            if key.startswith("ridge"):
                model.fit(fit_x[:, selected], fit_y, ridge__sample_weight=fit_weight)
            else:
                # HGB's regularization/leaf counts are easier to interpret at mean weight=1.
                model.fit(fit_x[:, selected], fit_y, sample_weight=fit_weight / fit_weight.mean())
            predicted = model.predict(sel_x[:, selected])
            errors[key] = float(mean_squared_error(sel_y, predicted, sample_weight=sel_weight))
        choice = min(errors, key=errors.get)
        models[name] = (selected, candidates[choice])
        summaries[name] = dict(selected_columns=selected.tolist(), candidate_columns=len(columns[name]),
            maximum_features=maximum, chosen_model=choice, selection_mse=errors,
            fit_tokens=len(fit_y), selection_tokens=len(sel_y))
    return models, summaries


def evaluate_explanations(models, samples):
    """Residual AUROC is an exploratory score audit, not a newly tuned detector."""
    features = np.concatenate([sample["features"] for sample in samples])
    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["gold"] for sample in samples]).astype(bool)
    first = []
    weights = []
    for sample in samples:
        mask = np.zeros(len(sample["gold"]), bool)
        positions = np.flatnonzero(sample["gold"])
        if len(positions):
            mask[positions[0]] = True
        first.append(mask)
        weights.append(np.full(len(mask), 1 / len(mask)))
    first = np.concatenate(first)
    weights = np.concatenate(weights)
    result = {}
    for name, (columns, model) in models.items():
        predicted = model.predict(features[:, columns])
        residual = logits - predicted
        chosen = first | ~labels
        has_classes = first.any() and (~labels).any()
        result[name] = dict(mse=float(mean_squared_error(logits, predicted, sample_weight=weights)),
            r2=float(r2_score(logits, predicted, sample_weight=weights)),
            score_spearman=float(spearmanr(logits, predicted).statistic) if np.ptp(predicted) and np.ptp(logits) else None,
            first_error_surrogate_auroc=float(roc_auc_score(first[chosen], predicted[chosen])) if has_classes else None,
            first_error_residual_auroc=float(roc_auc_score(first[chosen], residual[chosen])) if has_classes else None)
    return result


def screen_head_pairs(samples, partitions, layers, heads, limit=2):
    """Preselect a small number of same-layer pairs using FIT logit association only."""
    features, logits, _ = training_rows(samples, set(partitions["fit"]))
    entropy_columns = list(range(9, 9 + layers * heads))
    _, association = select_columns(features, logits, entropy_columns, maximum=layers * heads)
    pairs = []
    if heads < 2:
        return pairs
    for layer in range(layers):
        ranked = np.argsort(-association[layer * heads:(layer + 1) * heads], kind="stable")[:2]
        first, second = map(int, ranked)
        pairs.append(dict(layer=layer, first_head=first, second_head=second,
            fit_association=float(association[layer * heads + first] + association[layer * heads + second])))
    return sorted(pairs, key=lambda pair: (-pair["fit_association"], pair["layer"]))[:limit]
