"""Freeze source-disjoint, label-free reference moments before target inference."""

import numpy as np
from state_audit.storage import read_json
from tqdm import tqdm

from .filter_features import cached_features
from .optimize import DIRECTORY as FEATURE_DIRECTORY
from .optimize import methods_for
from .routes import ROUTE_SCORES
from .source_regions import compile_regions, region_mask

COVARIANCE_RIDGE = 1e-6


def load_features(output, settings):
    existing = output / "route_comparison_v2"
    directory = existing if (existing / "source_regions.json").is_file() else output / FEATURE_DIRECTORY
    regions = compile_regions(settings, directory)
    baseline, _ = methods_for(regions)
    attention = "attention_displacement" if regions["status"] == "available" else "prompt_attention_displacement"
    records = []
    for index, response in enumerate(tqdm(settings["responses"], desc="joint state inputs")):
        features = cached_features(response, output / "responses" / f"{index:04d}",
                                   output / FEATURE_DIRECTORY / "features" / f"{index:04d}.npz",
                                   region_mask(regions, response),
                                   fields=(*ROUTE_SCORES, "entropy", "ledger_error", "target", "query", "token_id"))
        observations = np.column_stack((features[baseline], features[attention], np.log1p(features["entropy"])))
        records.append({"response": response, "features": features, "observations": observations})
    return records, regions, baseline, attention


def reference_moments(records, excluded_source):
    selected = [r for r in records if r["response"]["source_id"] != excluded_source]
    sources = sorted({r["response"]["source_id"] for r in selected})
    if not sources:
        raise ValueError("joint state requires an independent reference source; provide --reference-output")
    means, second_moments = [], []
    for source in sources:
        values = np.concatenate([r["observations"] for r in selected if r["response"]["source_id"] == source])
        means.append(values.mean(0))
        second_moments.append(values.T @ values / len(values))
    mean = np.mean(means, axis=0)
    covariance = np.mean(second_moments, axis=0) - np.outer(mean, mean)
    covariance += COVARIANCE_RIDGE * max(float(np.trace(covariance) / len(mean)), 1e-12) * np.eye(len(mean))
    return {"mean": mean.tolist(), "covariance": covariance.tolist(), "source_ids": sources,
            "response_ids": [r["response"]["id"] for r in selected],
            "tokens": sum(len(r["observations"]) for r in selected), "source_weighting": "equal"}


def load_reference(reference_output, output, settings, records, baseline):
    if reference_output is None:
        return records, "source_held_out_input_cohort"
    if reference_output.resolve() == output.resolve():
        raise ValueError("--reference-output must be a separate cache; omit it for source-held-out pilot mode")
    reference_settings = read_json(reference_output / "settings.json")
    if reference_settings["model"] != settings["model"]:
        raise ValueError("reference and target observer models differ")
    for field in ("task", "generator"):
        if reference_settings.get("cohort", {}).get(field) != settings.get("cohort", {}).get(field):
            raise ValueError(f"reference and target {field} differ")
    reference, _, reference_baseline, _ = load_features(reference_output, reference_settings)
    if reference_baseline != baseline:
        raise ValueError("reference and target source definitions differ")
    return reference, "external_reference_with_overlapping_sources_excluded"
