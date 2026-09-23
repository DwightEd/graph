"""Post-scoring diagnostics; annotation access is confined to evaluation."""

import numpy as np
from scipy.stats import spearmanr
from state_audit.storage import write_json

from .evaluate import evaluation_records


def rank_audit(destination, annotations, methods):
    records = evaluation_records(destination, annotations, destination, tuple(methods))
    labels = np.concatenate([record["labels"] for record in records])
    scores = np.concatenate([record["scores"] for record in records])
    count = int(np.ceil(len(labels) / 10))
    result = {"top_count": count, "tie_order": "stable_input_response_then_target", "methods": {}}
    for column, method in enumerate(methods):
        finite = np.flatnonzero(np.isfinite(scores[:, column]))
        order = finite[np.argsort(-scores[finite, column], kind="stable")[:count]]
        correlations = {record["id"]: float(spearmanr(record["target"], record["scores"][:, column],
                                                     nan_policy="omit").statistic) for record in records}
        correlations = {key: value if np.isfinite(value) else None for key, value in correlations.items()}
        result["methods"][method] = {"ranked_tokens": len(order), "unscored_tokens": len(labels) - len(finite),
                                     "errors_in_top_decile": int(labels[order].sum()),
                                     "position_spearman_by_answer": correlations}
    write_json(destination / "ranking_audit.json", result)


def state_audit(destination, annotations):
    columns = ("source_read_mass", "source_read_concentration", "source_read_entropy",
               "source_support_terminal", "local_opposition", "history_carry",
               "source_support_state", "lineage_opposition", "unresolved_state",
               "unobserved_alternative_mass", "expected_path_nodes", "state_entropy")
    records = evaluation_records(destination, annotations, destination, columns)
    result = {"scope": "descriptive_by_answer_not_matched_controls", "by_answer": {}}
    for record in records:
        groups = {}
        for label, name in ((0, "unmarked"), (1, "error")):
            selected = record["scores"][record["labels"] == label]
            groups[name] = {"tokens": len(selected), "observations": {}}
            if len(selected):
                for column, field in enumerate(columns):
                    values = selected[:, column]
                    groups[name]["observations"][field] = {
                        "mean": float(values.mean()), "median": float(np.median(values)),
                        "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
                    }
        result["by_answer"][record["id"]] = groups
    write_json(destination / "state_audit.json", result)
