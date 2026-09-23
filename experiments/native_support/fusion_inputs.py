"""Use compact saved scores and one external reference; never read target labels."""

from pathlib import Path

import numpy as np
from state_audit.storage import read_json
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .state_inputs import load_features, reference_moments
from .state_model import DIRECTORY as STATE_DIRECTORY
from .switching import switching_filter

TOKEN_FIELDS = ("target", "query", "token_id", "ledger_error")


def load_state_scores(directory, response, methods):
    # Do not decompress the T×T run posterior for a scalar readout.
    with np.load(directory / "scores.npz", allow_pickle=False) as saved:
        scores = {name: saved[name] for name in (*methods, *TOKEN_FIELDS)}
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    if (not np.array_equal(scores["token_id"], response["token_ids"][prompt:])
            or not np.array_equal(scores["target"], np.arange(count))
            or not np.array_equal(scores["query"], np.arange(prompt - 1, prompt + count - 1))):
        raise ValueError(f"{response['id']}: saved state prediction alignment differs from input")
    return scores


def external_reference(settings, protocol, reference_output):
    saved_path = protocol["reference_output"]
    if saved_path is None:
        raise ValueError("fuse requires v4 scores made with --reference-output; run --stage model with an independent reference cache")
    reference_output = reference_output or Path(saved_path)
    reference_settings = read_json(reference_output / "settings.json")
    if reference_settings["model"] != settings["model"]:
        raise ValueError("reference and target observer models differ")
    for field in ("task", "generator"):
        if reference_settings.get("cohort", {}).get(field) != settings.get("cohort", {}).get(field):
            raise ValueError(f"reference and target {field} differ")
    target_sources = {r["source_id"] for r in settings["responses"]}
    reference_sources = {r["source_id"] for r in reference_settings["responses"]}
    if target_sources & reference_sources:
        raise ValueError("fuse requires entirely source-disjoint reference and target caches")
    records, _, baseline, attention = load_features(reference_output, reference_settings)
    if baseline != protocol["raw_baseline"]:
        raise ValueError("reference and target source definitions differ")
    prior = reference_moments(records, excluded_source=None)
    for source in target_sources:
        if prior != protocol["priors_by_source"][source]:
            raise ValueError("reference differs from the prior used by saved v4 scores; rerun --stage model with this reference")
    return reference_output, records, prior, baseline, attention, reference_settings.get("cohort", {"selection": "input_manifest"})


def reference_scores(records, prior, baseline, attention, window):
    collected, source_ids, response_ids = [], [], []
    mean = np.asarray(prior["mean"])[:1]
    covariance = np.asarray(prior["covariance"])[:1, :1]
    with threadpool_limits(limits=1):
        for record in tqdm(records, desc="reference route states"):
            values = record["observations"][:, :1]
            state = switching_filter(values, mean, covariance, window)
            features = record["features"]
            collected.append({baseline: features[baseline], attention: features[attention],
                              "entropy": features["entropy"], "route_state": state["state_mean"][:, 0]})
            source_ids.extend([record["response"]["source_id"]] * len(values))
            response_ids.extend([record["response"]["id"]] * len(values))
    joined = {name: np.concatenate([row[name] for row in collected]) for name in collected[0]}
    return joined, np.asarray(source_ids), np.asarray(response_ids)


def load_fusion_inputs(output, reference_output, window):
    settings = read_json(output / "settings.json")
    source = output / STATE_DIRECTORY / f"w{window}"
    protocol = read_json(source / "scoring_protocol.json")
    reference_output, records, prior, baseline, attention, cohort = external_reference(settings, protocol, reference_output)
    reference, sources, responses = reference_scores(records, prior, baseline, attention, window)
    return {"settings": settings, "state_directory": source, "state_protocol": protocol,
            "reference_output": reference_output, "reference_scores": reference,
            "reference_sources": sources, "reference_responses": responses,
            "reference_cohort": cohort, "baseline": baseline, "attention": attention}
