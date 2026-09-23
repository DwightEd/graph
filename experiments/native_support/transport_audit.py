"""Label-assisted audit of saved transport scores and states; no model calls."""

from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.stats import rankdata
from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)
from tqdm import tqdm

from .evaluate import annotation_targets, evaluation_records
from .transport_audit_bootstrap import paired_bootstrap
from .transport_audit_rank import (
    CANDIDATE,
    CONTROLS,
    RANK_SCOPES,
    audit_scopes,
    ranking_tables,
)
from .transport_audit_state import edge_rows, state_groups, state_measurements


def aligned_answer(response, annotation, directory, methods):
    scores = read_arrays(directory / "scores.npz")
    state = read_arrays(directory / "state.npz")
    count = len(response["token_ids"]) - response["prompt_length"]
    expected = {"target": np.arange(count), "query": response["prompt_length"] - 1 + np.arange(count),
                "token_id": np.asarray(response["token_ids"][response["prompt_length"]:])}
    for name, value in expected.items():
        if not np.array_equal(scores[name], value):
            raise ValueError(f'{response["id"]}: saved {name} does not match settings')
    labels, onsets, first, valid = annotation_targets(annotation, count, response["id"])
    if not all(np.isfinite(scores[name][valid]).all() for name in methods):
        raise ValueError(f'{response["id"]}: audit requires finite scores on the common evaluation tokens')
    if state["edge_weight"].shape != (count, count) or state["observed_budget"].shape != state["inferred_budget"].shape:
        raise ValueError(f'{response["id"]}: saved graph/budget shape mismatch')
    source_count = len(read_json(directory / "sources.json")["blocks"]) + 1
    return scores, state, (labels, onsets, first, valid), source_count


def token_rows(response, scores, annotation, measured, methods):
    labels, onsets, first, valid = annotation
    text = response["token_text"][response["prompt_length"]:]
    result = []
    for target in np.flatnonzero(valid):
        phase = "normal"
        if labels[target]:
            phase = "first_error" if first[target] else "other_onset" if onsets[target] else "continuation"
        delta = float(scores[CANDIDATE][target] - scores["raw_route"][target])
        result.append({"response_id": response["id"], "source_id": response["source_id"], "target": int(target),
            "query": int(scores["query"][target]), "token_id": int(scores["token_id"][target]), "token": text[target],
            "context": "".join(text[max(0, target - 8):target + 9]), "label": int(labels[target]),
            "is_span_onset": bool(onsets[target]), "is_answer_first_error": bool(first[target]),
            "phase": phase, "half": "front" if target < len(text) / 2 else "back", "response_length": len(text),
            **{name: float(scores[name][target]) for name in methods},
            "delta_raw_route": delta, "abs_delta_raw_route": abs(delta),
            "delta_offline_mean": float(scores[CANDIDATE][target] - scores["route_offline_mean"][target]),
            **{name: float(value[target]) for name, value in measured.items()}})
    return result


def state_checks(response, scores, state, measured, strength):
    degree = measured["degree"]
    comparisons = {
        "transport_budget_readout": (scores[CANDIDATE], measured["state_budget_readout"], 1e-10),
        "raw_budget_regrouping": (scores["raw_route"], measured["raw_budget_readout"], 1e-5),
        "saved_degree": (state["degree"], degree, 1e-6),
        "retention_coefficient": (state["retention_weight"], strength * degree / (1 + strength * degree), 1e-10),
        "role_state_equation": (measured["relative_role_residual"], np.zeros(len(degree)), 1e-7),
    }
    result = []
    for name, (left, right, tolerance) in comparisons.items():
        error = float(np.max(np.abs(left - right)))
        result.append({"response_id": response["id"], "check": name, "tolerance": tolerance,
                       "max_abs_error": error if np.isfinite(error) else None,
                       "passed": bool(np.isfinite(error) and error <= tolerance)})
    return result


def correlation(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def similarity_rows(tokens):
    result = []
    for scope, (selected, _) in audit_scopes(tokens).items():
        if scope not in RANK_SCOPES or not selected.any():
            continue
        candidate = np.asarray([row[CANDIDATE] for row in tokens])[selected]
        for control in CONTROLS:
            baseline = np.asarray([row[control] for row in tokens])[selected]
            change = np.abs(candidate - baseline)
            result.append({"scope": scope, "control": control, "tokens": len(candidate),
                "pearson": correlation(candidate, baseline),
                "spearman": correlation(rankdata(candidate), rankdata(baseline)),
                "mean_abs_change": float(change.mean()), "median_abs_change": float(np.median(change)),
                "q90_abs_change": float(np.quantile(change, .9)), "max_abs_change": float(change.max())})
    return result


def evaluation_checks(metrics, previous):
    result = []
    for row in metrics:
        if row["response_id"] != "ALL" or row["scope"] not in previous["methods"][row["method"]]:
            continue
        old = previous["methods"][row["method"]][row["scope"]]
        for name in ("tokens", "positives", "negatives", "auroc", "ap"):
            current, saved = row[name], old[name]
            matches = current == saved if current is None or saved is None else np.isclose(current, saved, atol=1e-12, rtol=0)
            result.append({"method": row["method"], "scope": row["scope"], "metric": name,
                           "saved": saved, "recomputed": current, "matches": bool(matches)})
    return result


def export_answer(destination, index, scores, state, annotation, measured, methods):
    labels, onsets, first, valid = annotation
    write_arrays(destination / "responses" / f"{index:04d}.npz",
        target=scores["target"], query=scores["query"], token_id=scores["token_id"],
        labels=labels, onsets=onsets, firsts=first, valid=valid,
        scores=np.column_stack([scores[name] for name in methods]), methods=np.asarray(methods),
        edge_weight=state["edge_weight"], reuse_weight=state["reuse_weight"], **measured)


def collect_answers(settings, annotations, source, destination, protocol):
    methods = list(protocol["methods"])
    tables = {name: [] for name in ("tokens", "graph_edges", "state_checks")}
    for index, response in enumerate(tqdm(settings["responses"], desc="audit saved transport")):
        directory = source / "responses" / f"{index:04d}"
        scores, state, annotation, source_count = aligned_answer(response, annotations[response["id"]], directory, methods)
        labels, _, _, valid = annotation
        measured = state_measurements(state, source_count, protocol["strength"], labels, valid)
        tables["tokens"].extend(token_rows(response, scores, annotation, measured, methods))
        tables["graph_edges"].extend({"response_id": response["id"], "source_id": response["source_id"], **row}
                                     for row in edge_rows(state, labels, valid))
        tables["state_checks"].extend(state_checks(response, scores, state, measured, protocol["strength"]))
        export_answer(destination, index, scores, state, annotation, measured, methods)
    return tables


def save_audit(destination, tables, summary, count):
    for name, rows in tables.items():
        write_csv(destination / f"{name}.csv", rows, list(rows[0]) if rows else ["response_id", "target"])
    write_json(destination / "summary.json", summary)
    paths = [destination / f"{name}.csv" for name in tables]
    paths.extend(destination / name for name in ("summary.json", "bootstrap.npz"))
    paths.extend(destination / "responses" / f"{index:04d}.npz" for index in range(count))
    with ZipFile(destination.parent / "audit_data.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(destination.parent))


def audit(output, annotations=None, bootstrap_replicates=1000, seed=37):
    source = output / "source_transport"
    destination = source / "audit"
    annotations = annotations or output / "annotations.json"
    settings = read_json(output / "settings.json")
    protocol = read_json(source / "scoring_protocol.json")
    # Reuse the evaluation boundary's token-ID, source-ID and valid-token checks.
    evaluation_records(output, annotations, source, tuple(protocol["methods"]))
    tables = collect_answers(settings, read_json(annotations), source, destination, protocol)
    tables.update(ranking_tables(tables["tokens"], protocol["methods"]))
    tables["state_groups"] = state_groups(tables["tokens"])
    tables["score_similarity"] = similarity_rows(tables["tokens"])
    tables["evaluation_checks"] = evaluation_checks(tables["metrics"], read_json(source / "evaluation.json"))
    tables["bootstrap"], draws = paired_bootstrap(tables["tokens"], bootstrap_replicates, seed)
    write_arrays(destination / "bootstrap.npz", **draws)
    summary = audit_summary(settings, protocol, annotations, tables, bootstrap_replicates, seed)
    save_audit(destination, tables, summary, len(settings["responses"]))
    return summary


def audit_summary(settings, protocol, annotations, tables, replicates, seed):
    tokens = tables["tokens"]
    return {"purpose": "label_assisted_saved_transport_audit", "responses": len(settings["responses"]),
        "tokens": len(tokens), "positives": sum(row["label"] for row in tokens),
        "first_errors_outside_onsets": sum(row["is_answer_first_error"] and not row["is_span_onset"] for row in tokens),
        "annotations_path": str(annotations), "labels_used_for_audit": True, "scores_changed": False,
        "model_forward": False, "graph_recomputed": False, "state_solved_again": False,
        "state_checks_passed": all(row["passed"] for row in tables["state_checks"]),
        "saved_evaluation_matches": all(row["matches"] for row in tables["evaluation_checks"]),
        "half_definition": "answer_token_position; not error_span_halves",
        "ap_credit": "exact complete-tie threshold precision credit; associative accounting, not causal attribution",
        "budget_changes": "fractional inclusion change under random boundary tie order; not calibrated alarms",
        "graph_label_alignment": "receiver target t; donor query target s OR actual answer key s-1; -1 is unknown",
        "bootstrap": {"unit": "source", "paired": True, "replicates": replicates, "seed": seed,
                      "interval": "exploratory percentile 95%; no method or parameter selection"},
        "cohort": settings.get("cohort", {}), "scoring_protocol": protocol,
        "tables": {name: len(rows) for name, rows in tables.items()}, "archive": "source_transport/audit_data.zip"}
