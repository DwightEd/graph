"""Frozen supervised probes, fitted only on other sources."""

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..evaluate import ranking


def make_model(name):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if name == "logistic":
        return make_pipeline(imputer, StandardScaler(), LogisticRegression(
            C=1.0, solver="liblinear", max_iter=2000, class_weight="balanced", random_state=37))
    return make_pipeline(imputer, HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=7, max_depth=3, min_samples_leaf=20,
        l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=37))


def validate_reference(dataset, reference):
    target_sources = {r["source_id"] for r in dataset["records"]}
    training_sources = {r["source_id"] for r in reference["records"]}
    if target_sources & training_sources:
        raise ValueError("Training reference and evaluation sources must be disjoint")
    fields = ("model", "rank", "seed", "dtype", "roles", "channels")
    if any(dataset["protocol"][name] != reference["protocol"][name] for name in fields):
        raise ValueError("Training and evaluation capture protocols differ")
    if dataset["schema"] != reference["schema"]:
        raise ValueError("Training and evaluation feature identities differ")


def source_folds(records, reference=None):
    if reference is not None:
        return [("reference", reference, list(range(len(records))))]
    sources = sorted({record["source_id"] for record in records})
    if len(sources) < 2:
        raise ValueError("Need at least two sources or a disjoint --reference")
    return [(f"{index:04d}", [r for r in records if r["source_id"] != source],
             [i for i, r in enumerate(records) if r["source_id"] == source])
            for index, source in enumerate(sources)]


def training_arrays(records, view):
    values = np.concatenate([r["views"][view][r["valid"]] for r in records])
    labels = np.concatenate([r["labels"][r["valid"]] for r in records])
    sources = np.concatenate([np.repeat(r["source_id"], r["valid"].sum()) for r in records])
    counts = {source: (sources == source).sum() for source in np.unique(sources)}
    weights = np.array([1 / counts[source] for source in sources])
    weights /= weights.mean()
    return values, labels, weights


def fit_predict(training, records, targets, view, model_name, directory):
    values, labels, weights = training_arrays(training, view)
    detail = dict(training_tokens=len(labels), training_positives=int(labels.sum()))
    if len(np.unique(labels)) != 2:
        return None, {**detail, "status": "unavailable_training_requires_both_classes"}
    model = make_model(model_name)
    estimator = model.steps[-1][0]
    model.fit(values, labels, **{f"{estimator}__sample_weight": weights})
    predictions = {i: model.predict_proba(records[i]["views"][view])[:, 1] for i in targets}
    detail.update(status="trained", train_resubstitution=ranking(labels, model.predict_proba(values)[:, 1]),
                  convergence_iterations=np.asarray(model.steps[-1][1].n_iter_).tolist())
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / f"{view}_{model_name}.joblib")
    return predictions, detail


def train_readouts(dataset, destination, model_names, reference=None):
    records = dataset["records"]
    if reference is not None:
        validate_reference(dataset, reference)
    folds = source_folds(records, None if reference is None else reference["records"])
    methods = [f"{view}_{model}" for view in dataset["schema"] for model in model_names]
    predictions = [{method: np.full(len(r["labels"]), np.nan) for method in methods} for r in records]
    reports = []
    for fold, training, targets in tqdm(folds, desc="source-held-out folds"):
        report = dict(fold=fold, training_sources=sorted({r["source_id"] for r in training}),
                      evaluation_sources=sorted({records[i]["source_id"] for i in targets}), methods={})
        for view in dataset["schema"]:
            for model in model_names:
                method = f"{view}_{model}"
                values, detail = fit_predict(training, records, targets, view, model, destination / "models" / fold)
                report["methods"][method] = detail
                if values is not None:
                    for index, scores in values.items():
                        predictions[index][method] = scores
        reports.append(report)
    return predictions, reports
