"""Read-only cache audit. Write CSV/NPZ/JSON data; never capture, fit or rescore."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from state_audit.storage import read_arrays, read_json, write_csv, write_json
from tqdm import tqdm

from .dynamics_audit_features import (
    accumulate_features,
    export_vectors,
    feature_rows,
    projection_rows,
    representation_schema,
    token_vectors,
)
from .dynamics_audit_rank import ranking_tables
from .dynamics_core import mode_posteriors
from .dynamics_normals import normal_tables
from .evaluate import annotation_targets

DIRECTORY = "state_dynamics"


def check_arrays(response, observation, saved, annotation, capture, methods):
    count = len(response["token_ids"]) - response["prompt_length"]
    expected = {"target": np.arange(count), "query": np.arange(count) + response["prompt_length"] - 1,
                "token_id": np.asarray(response["token_ids"][response["prompt_length"]:])}
    checks = {f"{origin}_{name}": bool(np.array_equal(arrays[name], value))
              for origin, arrays in (("observation", observation), ("score", saved))
              for name, value in expected.items()}
    checks["annotation_token_ids"] = np.array_equal(annotation["token_ids"], expected["token_id"])
    checks["annotation_source"] = annotation.get("source_id", response["source_id"]) == response["source_id"]
    checks["future_count"] = np.array_equal(observation["future_query_count"], np.maximum(count - np.arange(count) - 2, 0))
    layers, heads = observation["profile"].shape[1:3]
    shapes = {"state": (count, capture["rank"]), "source_input": (count, layers * heads * capture["rank"]),
              "history": (count, layers * heads * capture["rank"]),
              "ffn": (count, layers * (capture["rank"] + capture["choices"])),
              "profile": (count, layers, heads, len(observation["profile_channels"]))}
    for name, shape in shapes.items():
        checks[f"shape_{name}"] = observation[name].shape == shape
    # Unavailable evidence-only baselines are intentionally NaN in prompt-only caches.
    used = {*shapes, "entropy", "surprisal", "candidate_tail_mass", "future_query_count"}
    used.update(name for name in methods if name in observation)
    numeric = {f"observation_{key}": observation[key] for key in sorted(used)}
    numeric.update({f"score_{name}": saved[name] for name in (*methods, "mode_emission", "mode_posterior")})
    checks.update({f"finite_{key}": bool(np.isfinite(value).all()) for key, value in numeric.items()})
    checks["posterior_simplex"] = bool(np.allclose(saved["mode_posterior"].sum(1), 1, atol=1e-6, rtol=0)
                                      and ((saved["mode_posterior"] >= 0) & (saved["mode_posterior"] <= 1)).all())
    checks["risk_equals_H"] = np.array_equal(saved["state_dynamics"], saved["mode_posterior"][:, 1])
    checks["risk_alias"] = np.array_equal(saved["risk"], saved["state_dynamics"])
    checks.update({f"saved_{name}": np.array_equal(saved[name], observation[name])
                   for name in methods if name in observation})
    return [{"response_id": response["id"], "check": name, "passed": bool(value)} for name, value in checks.items()]


def replay_modes(saved, parameters):
    # Use the same float32 softmax as infer(), avoiding artificial replay drift.
    transition = torch.as_tensor(parameters["mode_logits"], dtype=torch.float32).softmax(-1).numpy()
    initial = torch.as_tensor(parameters["initial_logits"], dtype=torch.float32).softmax(-1).numpy()
    return mode_posteriors(saved["mode_emission"], transition, initial)


def audit_tokens(response, observation, saved, annotation, posterior, methods):
    count = len(saved["target"])
    labels, onsets, first, valid = annotation_targets(annotation, count, response["id"])
    text = response["token_text"][response["prompt_length"]:]
    profile_means = observation["profile"].mean(axis=(1, 2))
    result = []
    for target in np.flatnonzero(valid):
        phase = "normal"
        if labels[target]:
            phase = "first_error" if first[target] else "other_onset" if onsets[target] else "continuation"
        log_modes = posterior["log_smoothed"][target]
        result.append({"response_id": response["id"], "source_id": response["source_id"], "target": int(target),
            "query": int(saved["query"][target]), "token_id": int(saved["token_id"][target]), "token": text[target],
            "context": "".join(text[max(0, target - 8):min(count, target + 9)]),
            "label": int(labels[target]), "is_span_onset": bool(onsets[target]),
            "is_answer_first_error": bool(first[target]), "phase": phase,
            "half": "front" if target < count / 2 else "back", "response_length": count,
            **{name: float(saved[name][target]) for name in methods},
            "state_log_odds": float(log_modes[1] - log_modes[0]),
            "emission_log_ratio": float(saved["mode_emission"][target, 1] - saved["mode_emission"][target, 0]),
            "future_query_count": int(observation["future_query_count"][target]),
            **{f"mean_head_{name}": float(profile_means[target, index])
               for index, name in enumerate(observation["profile_channels"])}})
    return result, labels, valid


def event_rows(tokens, radius=8):
    lookup = {(row["response_id"], row["target"]): row for row in tokens}
    result = []
    for event in tokens:
        if not (event["is_span_onset"] or event["is_answer_first_error"]):
            continue
        for offset in range(-radius, radius + 1):
            key = (event["response_id"], event["target"] + offset)
            if key in lookup:
                result.append({"event_target": event["target"], "offset": offset,
                               "event_is_first": event["is_answer_first_error"], **lookup[key]})
    return result


def write_archive(destination, tables, count):
    paths = [destination / f"{name}.csv" for name in tables]
    paths.extend(destination / name for name in ("summary.json", "model.json", "representation_schema.json"))
    paths.extend(destination / "representations" / f"{index:04d}.npz" for index in range(count))
    with ZipFile(destination.parent / "audit_data.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(destination.parent))


def recheck_evaluation(metrics, previous):
    result = []
    for row in metrics:
        method, scope = row["method"], row["scope"]
        if method not in previous["methods"] or scope not in previous["methods"][method]:
            continue
        old = previous["methods"][method][scope]
        for name in ("tokens", "positives", "negatives", "auroc", "ap"):
            current, saved = row[name], old[name]
            matches = current == saved if current is None or saved is None else np.isclose(current, saved, rtol=0, atol=1e-12)
            result.append({"method": method, "scope": scope, "metric": name, "saved": saved,
                           "recomputed": current, "matches": bool(matches)})
    return result


def save_tables(destination, tables):
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ["response_id", "target"]
        write_csv(destination / f"{name}.csv", rows, fields)


def model_data(parameters, complete, coordinates):
    transition = torch.as_tensor(parameters["mode_logits"], dtype=torch.float32).softmax(-1).numpy()
    factor = np.tril(parameters["noise_factor"])
    noise = factor @ factor.transpose(0, 2, 1) + .03 * np.eye(factor.shape[-1])
    return {"training": complete, "transition": transition.tolist(),
            "expected_mode_duration": (1 / (1 - transition.diagonal())).tolist(),
            "noise_eigenvalues": np.linalg.eigvalsh(noise).tolist(),
            "input_variance_multiplier": float(np.exp(np.clip(parameters["log_input_variance"], -7, 3))),
            "parameters": {name: {"shape": list(value.shape), "norm": float(np.linalg.norm(value))}
                           for name, value in parameters.items()},
            "projection": {name: {"shape": list(coordinates[name]["components"].shape),
                                   "train_retained_variance": float(coordinates[name]["retained_variance"])}
                           for name in ("source_input", "history", "profile")}}


def audit_answer(index, response, annotation, source, destination, model, tables, moments):
    observation = read_arrays(source / "capture" / f"{index:04d}" / "observations.npz")
    saved = read_arrays(source / "responses" / f"{index:04d}" / "scores.npz")
    checks = check_arrays(response, observation, saved, annotation, model["capture"], model["methods"])
    tables["checks"].extend(checks)
    if not all(row["passed"] for row in checks):
        save_tables(destination, {"checks": tables["checks"]})
        write_json(destination / "summary.json", {"status": "failed", "response_id": response["id"],
                                                   "failed_checks": [row["check"] for row in checks if not row["passed"]]})
        raise ValueError(f"{response['id']}: alignment/numeric audit failed; see {destination / 'checks.csv'}")
    posterior = replay_modes(saved, model["parameters"])
    error = float(np.max(np.abs(posterior["smoothed"] - saved["mode_posterior"])))
    tables["posterior_replay"].append({"response_id": response["id"], "max_abs_error": error,
                                     "matches": error <= 1e-6})
    rows, labels, valid = audit_tokens(response, observation, saved, annotation, posterior, model["methods"])
    tables["tokens"].extend(rows)
    tables["coverage"].append({"response_id": response["id"], "source_id": response["source_id"],
                               "representation_file": f"representations/{index:04d}.npz",
                               "captured": len(labels), "evaluated": int(valid.sum()), "excluded": int((~valid).sum())})
    vectors = token_vectors(observation, model["coordinates"])
    export_vectors(destination / "representations" / f"{index:04d}.npz", observation, vectors, labels, valid, posterior)
    accumulate_features(moments, observation, vectors, labels, valid, saved["state_dynamics"])
    tables["projection"].extend(projection_rows(response["id"], observation, model["coordinates"], labels, valid,
                                               saved["state_dynamics"]))
    if index == 0:
        write_json(destination / "representation_schema.json", representation_schema(observation, vectors, model["capture"]))


def load_model_data(output, reference, settings, protocol):
    from .dynamics import load_coordinates, verify_disjoint

    reference_settings = read_json(reference / "settings.json")
    verify_disjoint(settings, reference_settings)
    capture = read_json(output / DIRECTORY / "capture_settings.json")
    reference_model = reference / DIRECTORY / "model"
    trained = read_json(reference_model / "fit_settings.json")["capture"]
    for current, declared in ((settings, capture), (reference_settings, trained)):
        if current["responses"] != declared["responses"] or current["model"] != declared["model"]:
            raise ValueError("Audit settings differ from the original capture manifest")
    for field in ("model", "rank", "choices", "seed", "block_tokens", "dtype", "source_scope", "special_ids"):
        if capture[field] != trained[field]:
            raise ValueError(f"Audit reference/capture mismatch: {field}")
    identities = [row["id"] for row in settings["responses"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate response IDs in scoring settings")
    return {"capture": capture, "methods": tuple(protocol["methods"]),
            "coordinates": load_coordinates(reference_model / "coordinates.npz"),
            "parameters": read_arrays(reference_model / "parameters.npz"),
            "complete": read_json(reference_model / "complete.json")}


def audit(output, reference=None, annotations=None):
    source = output / DIRECTORY
    destination = source / "audit"
    write_json(destination / "summary.json", {"status": "running"})
    (source / "audit_data.zip").unlink(missing_ok=True)
    protocol = read_json(source / "scoring_protocol.json")
    reference = reference or Path(protocol["reference_output"])
    annotations = annotations or output / "annotations.json"
    settings, labels_by_id = read_json(output / "settings.json"), read_json(annotations)
    model = load_model_data(output, reference, settings, protocol)
    tables = {name: [] for name in ("tokens", "checks", "coverage", "posterior_replay", "projection")}
    moments = {}
    for index, response in enumerate(tqdm(settings["responses"], desc="cached dynamics audit")):
        audit_answer(index, response, labels_by_id[response["id"]], source, destination, model, tables, moments)
    tables.update(ranking_tables(tables["tokens"], (*model["methods"], "state_log_odds")))
    tables["features"] = feature_rows(moments)
    tables["event_windows"] = event_rows(tables["tokens"])
    tables.update(normal_tables(tables["tokens"]))
    tables["evaluation_recheck"] = recheck_evaluation(tables["metrics"], read_json(source / "evaluation.json"))
    save_tables(destination, tables)
    write_json(destination / "model.json", model_data(model["parameters"], model["complete"], model["coordinates"]))
    summary = audit_summary(tables, model, output, reference, annotations)
    write_json(destination / "summary.json", summary)
    write_archive(destination, tables, len(settings["responses"]))
    return summary


def audit_summary(tables, model, output, reference, annotations):
    replay_ok = all(row["matches"] for row in tables["posterior_replay"])
    evaluation_ok = all(row["matches"] for row in tables["evaluation_recheck"])
    return {"status": "verified" if replay_ok and evaluation_ok else "mismatch",
            "purpose": "cached_dynamics_data_audit", "model_forward": False, "model_fitting": False,
            "score_files_modified": False, "labels_used": "diagnostics_only", "classifier_trained": False,
            "responses": len(tables["coverage"]), "evaluated_tokens": len(tables["tokens"]),
            "errors": sum(row["label"] for row in tables["tokens"]),
            "alignment_checks": len(tables["checks"]), "posterior_replay_matches": replay_ok,
            "saved_evaluation_matches": evaluation_ok, "reference_output": str(reference),
            "annotations": str(annotations), "archive": str(output / DIRECTORY / "audit_data.zip"),
            "diagnostic_readout": "state_log_odds replays cached emissions; only finite-precision ties can change ranks",
            "primary_readout_unchanged": "state_dynamics", "mode_H_is_truth_label": False,
            "representation_widths": {"z": len(model["coordinates"]["state_mean"]),
                                      **{name: len(model["coordinates"][name]["components"])
                                         for name in ("source_input", "history", "profile")}},
            "all_error": [row for row in tables["metrics"] if row["scope"] == "all_error"],
            "rows_by_table": {name: len(rows) for name, rows in tables.items()}}
