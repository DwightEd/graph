"""Token-conditioned finite-effect diagnostics, with pooled/cross-answer gains separated."""

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..evidence_contrast.unit_report import evaluate_units
from .token_representation import COMPARISONS, METHODS, TOKEN_METHODS


def method_names(protocol):
    tokens, methods, comparisons = list(TOKEN_METHODS), list(METHODS), list(COMPARISONS)
    if protocol["unit_carrier_control"]:
        tokens.append("unit_carrier_v1")
        methods.extend(("unit_carrier_v1", "unit_carrier_v1_unit_mean"))
        comparisons.extend(("token_source_selected" + suffix, "unit_carrier_v1" + suffix)
                           for suffix in ("", "_unit_mean"))
    return tokens, methods, comparisons


def ranking_scope(evaluation):
    rows = {}
    for method, phases in evaluation.get("methods", {}).items():
        metric = phases["all_error"]
        within = metric["within_answer"]
        total = metric["positives"] * (metric["tokens"] - metric["positives"])
        pairs = within["positive_negative_pairs"]
        same_auc = within["pair_weighted_auroc"]
        same_correct = pairs * same_auc if pairs else 0
        cross_auc = (total * metric["auroc"] - same_correct) / (total - pairs) if total > pairs else None
        rows[method] = dict(pooled_auroc=metric["auroc"], within_answer_auroc=same_auc,
            within_answer_pairs=pairs, cross_answer_auroc=cross_auc, cross_answer_pairs=total - pairs)
    return dict(note="Pooled gains can arise entirely between answers; no independent-token significance claim", methods=rows)


def edge_rows(saved, response):
    pieces = response["token_text"][response["prompt_length"]:]
    effect = saved["keep"][:, None, :].astype(float) - saved["single"]
    rows = []
    for index, (layer, head, receiver, key) in enumerate(saved["edges"]):
        rows.append(dict(response_id=response["id"], target=int(saved["target"]),
            target_text=pieces[int(saved["target"])], foil_id=int(saved["foil_id"]),
            foil_text=str(saved["foil_text"]),
            layer=int(layer), head=int(head), receiver=int(receiver), key=int(key),
            key_text=pieces[key], receiver_text=pieces[receiver],
            current_receiver=bool(receiver == saved["target"] - 1),
            gradient_with_source=float(saved["approximation"][index, 0]),
            gradient_without_source=float(saved["approximation"][index, 1]),
            margin_delete_with_source=float(effect[0, index, 1]),
            margin_delete_without_source=float(effect[1, index, 1]),
            route_mass_with_source=float(saved["route_mass"][index, 0]),
            route_mass_without_source=float(saved["route_mass"][index, 1]),
            message_energy_with_source=float(saved["message_energy"][index, 0]),
            message_energy_without_source=float(saved["message_energy"][index, 1])))
    return rows


def diagnostics(output, settings):
    rows, edges = [], []
    for index, response in enumerate(settings["responses"]):
        directory = output / "responses" / f"{index:04d}"
        count = len(response["token_ids"]) - response["prompt_length"]
        for target in range(count):
            saved = read_arrays(directory / f"token_{target:06d}.npz")
            effect = saved["keep"][:, None, :].astype(float) - saved["single"]
            joint = saved["keep"].astype(float) - saved["joint"]
            gradient_error = abs(effect[:, :, 1].T - saved["approximation"])
            rows.append(dict(response_id=response["id"], target=target, selected_edges=len(saved["edges"]),
                candidate_edges=int(saved["candidate_count"]),
                earlier_receiver_edges=int((saved["edges"][:, 2] < target - 1).sum()),
                sham_overlap=int(np.all(saved["edges"] == saved["sham_edges"], axis=1).sum()),
                gradient_finite_mae=float(gradient_error.mean()) if gradient_error.size else None,
                reconstruction_max_error=float(saved["reconstruction_error"].max()) if saved["reconstruction_measured"] else None,
                joint_minus_sum_single_margin_max=float(abs(joint[:, 1] - effect[:, :, 1].sum(1)).max())))
            edges.extend(edge_rows(saved, response))
    write_csv(output / "token_diagnostics.csv", rows, list(rows[0]))
    write_csv(output / "selected_edges.csv", edges, list(edges[0]) if edges else ["target"])


def evaluate(output, settings, protocol):
    diagnostics(output, settings)
    tokens, methods, comparisons = method_names(protocol)
    result = evaluate_units(output, settings, methods, tokens, comparisons)
    write_json(output / "ranking_scope.json", ranking_scope(result))
    return result
