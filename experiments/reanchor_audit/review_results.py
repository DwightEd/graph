"""Audit saved results on CPU, including candidate/source misalignment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .graph import analyze_graph
from .selection import ranked_edges


def read_json(path):
    return json.loads(path.read_text())


def readout_tables(directory, identity):
    baseline = read_json(directory / "baseline.json")
    worlds = {path.stem: read_json(path) for path in (directory / "worlds").glob("*.json")}
    metrics = ["sequence_margin", "next_margin", "tail_margin", "mean_margin"]
    world_rows = [dict(identity, world=name, next_top1_changed=record["next_top1"] != baseline["next_top1"],
        **{key: record[key] for key in metrics}) for name, record in worlds.items()]
    effects = pd.read_csv(directory / "effects.csv")
    rows = []
    for effect in effects.to_dict("records"):
        record = worlds[effect["unit"] + f"_{effect['dose']:g}"]
        delta = {"delta_" + key: baseline[key] - record[key] for key in metrics}
        np.testing.assert_allclose(delta["delta_sequence_margin"], effect["support"], atol=1e-12)
        rows.append(dict(identity, **effect, **delta))
    base_row = dict(identity, **{key: baseline[key] for key in metrics},
        first_candidate_tokens=baseline["correct_tokens"], second_candidate_tokens=baseline["wrong_tokens"],
        branch_first_disagreement_max=max(abs(baseline[key]) for key in
            ("correct_branch_first_error", "wrong_branch_first_error")))
    return rows, world_rows, base_row


def previous_mass_bounds(graph, layer, head, receiver, source):
    positions = graph["source"][layer, head, receiver - 1]
    mass = graph["attention"][layer, head, receiver - 1]
    retained = positions == source
    if retained.any():
        value = float(mass[retained].sum())
        return value, value
    # An omitted earlier key has mass at most the smallest saved top-k entry.
    return 0., float(mass.min())


def candidate_rows(graph, context, plan, window, identity):
    rows = []
    text = context["token_text"]
    special = np.array([piece.startswith("<|") and piece.endswith("|>") for piece in text])
    for event in plan:
        entry = event["entry"]
        layer, head, receiver = [entry[key] for key in ("layer", "head", "receiver")]
        source = entry["sources"][0]
        indices = graph["source"][layer, head, receiver]
        mass = graph["attention"][layer, head, receiver]
        current = float(mass[indices == source].sum())
        lower, upper = previous_mass_bounds(graph, layer, head, receiver, source)
        cutoff = max(receiver - window, context["prompt_length"])
        roles = {"retained_" + role + "_mass_lower_bound": float(mass[np.isin(indices, positions)].sum())
                 for role, positions in context["roles"].items()}
        rows.append(dict(identity, event=event["event"], selection=event["selected"]["selection"],
            layer=layer, head=head, receiver=receiver, source=source, source_text=text[source],
            receiver_text=text[receiver], distance_to_decision=len(text) - 1 - receiver,
            source_special=bool(special[source]), source_in_lookback_set=bool(source < cutoff and not special[source]),
            selected_mass=current, selected_mass_change_lower=current - upper,
            selected_mass_change_upper=current - lower, lookback_gain=event["selected"]["lookback_gain"],
            retained_mass=float(mass.sum()), **roles))
    return rows, special


def root_rows(flow, identity):
    rows = []
    for capacity in ("attention", "message_norm"):
        for seed in ("potential", "uniform"):
            roots = flow[capacity + "_" + seed + "_nodes"][0]
            rows.append(dict(identity, capacity=capacity, seed=seed, position_zero_root_mass=float(roots[0]),
                other_root_mass=float(roots[1:].sum()), maximal_root_position=int(roots.argmax())))
    return rows


def ranking_rows(graph, flow, context, special, window, identity):
    """Sensitivity diagnostics only; none of these new candidates has causal results."""
    edges = ranked_edges(graph, flow)
    eligible = edges[(edges.receiver < len(context["prefix_ids"]) - 1)
                     & (edges.lookback_gain > 0) & (edges.target_flow > 0)].copy()
    non_special = ~special[eligible.source]
    far = eligible.source < np.maximum(eligible.receiver - window, context["prompt_length"])
    rows = []
    for metric in ("target_flow", "norm_flow", "uniform_root_flow"):
        for name, mask in (("original_pool", np.ones(len(eligible), dtype=bool)),
                           ("non_special", non_special), ("non_special_far", non_special & far)):
            top = eligible.loc[mask].sort_values(metric, ascending=False)
            top = top.drop_duplicates(["layer", "head", "receiver"]).head(5)
            for rank, item in enumerate(top.to_dict("records"), 1):
                rows.append(dict(identity, metric=metric, pool=name, rank=rank, **item,
                    source_text=context["token_text"][item["source"]],
                    receiver_text=context["token_text"][item["receiver"]], causal_measurement="not_measured_for_this_ranking"))
    return rows


def review_case(directory, settings):
    context = read_json(directory / "context.json")
    identity = {key: context[key] for key in ("case_id", "source_id", "side")}
    with np.load(directory / "routes.npz", allow_pickle=False) as saved:
        graph = {name: saved[name] for name in saved.files}
    flow = analyze_graph(graph)
    plan = read_json(directory / "plan.json")
    candidates, special = candidate_rows(graph, context, plan, settings["window"], identity)
    effects, worlds, baseline = readout_tables(directory, identity)
    random = [dict(identity, **row) for row in pd.read_csv(directory / "random_controls.csv").to_dict("records")]
    return dict(candidates=candidates, roots=root_rows(flow, identity), effects=effects, worlds=worlds,
        baselines=[baseline], random=random,
        ranking_sensitivity=ranking_rows(graph, flow, context, special, settings["window"], identity))


def summarize(tables):
    primary = tables["candidates"].query("selection == 'lookback_target_flow'")
    effects = tables["effects"]
    dose = effects.pivot(index=["case_id", "side", "event", "source_group"], columns="dose", values="support")
    full = effects[effects.dose == 1.]
    checks = effects.filter(regex="error$")
    return dict(sources=int(effects.source_id.nunique()), primary_candidates=len(primary),
        primary_special_sources=int(primary.source_special.sum()),
        primary_sources_outside_lookback_set=int((~primary.source_in_lookback_set).sum()),
        primary_selected_source_decreased=int((primary.selected_mass_change_upper < 0).sum()),
        finite_effect_rows=int(np.isfinite(effects.support).sum()),
        max_control_error=float(checks.max().max()),
        full_dose_rows=len(full), full_dose_abs_effect_above_005=int((full.support.abs() > .05).sum()),
        dose_same_sign=int((np.sign(dose[.25]) == np.sign(dose[1.])).sum()),
        dose_opposite_sign=int((dose[.25] * dose[1.] < 0).sum()),
        dose_exactly_one_zero=int(((dose[.25] == 0) ^ (dose[1.] == 0)).sum()),
        saved_worlds=len(tables["worlds"]), next_top1_changes=int(tables["worlds"].next_top1_changed.sum()),
        effect_reference_nats=.05, reference_is_significance_test=False,
        purpose="cached_mechanism_review_not_detector_evaluation")


def review_saved_results(input_path, output):
    settings = read_json(input_path / "audit_config.json")
    collections = {}
    for directory in tqdm(sorted((input_path / "cases").glob("*/*")), desc="cached route review"):
        for name, rows in review_case(directory, settings).items():
            collections.setdefault(name, []).extend(rows)
    tables = {name: pd.DataFrame(rows) for name, rows in collections.items()}
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    summary = summarize(tables)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review_saved_results(args.input, args.output), ensure_ascii=False))
