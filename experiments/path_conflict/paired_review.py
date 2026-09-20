"""CPU review of saved paired CSVs; retain head identity and readout units."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paired_exports import adaptation_coverage
from .paired_report import classify_interactions, control_counts


def arithmetic_checks(tables):
    effects, interactions = tables["effects"], tables["interactions"]
    adaptation, persistence = tables["adaptation"], tables["persistence"]
    residuals = dict(
        single=effects.full - effects.cut - effects.support,
        joint=interactions.full - interactions.without_both - interactions.joint_support,
        interaction=interactions.full - interactions.without_left - interactions.without_right
                    + interactions.without_both - interactions.interaction,
        left_conditional=interactions.without_right - interactions.without_both - interactions.left_conditional,
        right_conditional=interactions.without_left - interactions.without_both - interactions.right_conditional,
        restoration=adaptation.restored - adaptation.removed - adaptation.restoration_gain,
        persistence=persistence.full - persistence.changed - persistence.support)
    return {name: dict(rows=len(values), finite_rows=int(np.isfinite(values).sum()),
                      maximum_absolute_error=float(values.abs().max()))
            for name, values in residuals.items()}


def persistence_comparison(effects, persistence):
    keys = ["case_id", "source_id", "side", "phase", "layer", "head", "selection", "position"]
    selected = effects.source_group.eq("head_total") & effects.dose.eq(1) & effects.control.isna()
    current = effects.loc[selected, keys + ["support", "numeric_ok"]].rename(
        columns=dict(support="current_support", numeric_ok="current_numeric_ok"))
    carried = persistence[keys + ["full", "support", "numeric_ok", "token_path"]].rename(
        columns=dict(support="carried_support", numeric_ok="carried_numeric_ok"))
    return carried.merge(current, on=keys, how="left", validate="one_to_one")


def phase_readouts(source, effects):
    keys = ["case_id", "side", "phase"]
    rows = []
    for identity, frame in effects.groupby(keys):
        directory = source / "pairs" / identity[0] / identity[1] / identity[2]
        context = json.loads((directory / "context.json").read_text())
        row = frame.iloc[0]
        rows.append(dict(zip(keys, identity), position=row.position, readout=row.readout,
            baseline=row.full, baseline_unique_values=frame.full.nunique(),
            first_candidate_tokens=len(context["candidates"][0]),
            second_candidate_tokens=len(context["candidates"][1]),
            baseline_record_available=(directory / "baseline.json").exists()
                or (directory / "baseline.npz").exists()))
    return pd.DataFrame(rows)


def save_tables(source, output, tables, minimum_effect):
    effects = tables["effects"]
    onset = effects[effects.phase.eq("onset")].drop(columns="sources")
    onset.to_csv(output / "onset_effects.csv", index=False)
    interactions = classify_interactions(tables["interactions"], minimum_effect)
    interactions[interactions.phase.eq("onset")].to_csv(output / "onset_four_worlds.csv", index=False)
    adaptation = tables["adaptation"].copy()
    adaptation["distance_before"] = (adaptation.removed - adaptation.full).abs()
    adaptation["distance_after"] = (adaptation.restored - adaptation.full).abs()
    adaptation["overshoots_full"] = (adaptation.restored - adaptation.full) * (
        adaptation.removed - adaptation.full) < 0
    adaptation.to_csv(output / "restoration.csv", index=False)
    persistence_comparison(effects, tables["persistence"]).to_csv(
        output / "persistence_vs_current.csv", index=False)
    phase_readouts(source, effects).to_csv(output / "phase_readouts.csv", index=False)
    adaptation_coverage(source, tables["adaptation"]).to_csv(output / "adaptation_coverage.csv", index=False)


def grouped_bars(axis, frame, heads, title):
    colors = dict(supported="#246A9C", unsupported="#C56A29")
    positions = np.arange(len(heads))
    for side, offset in (("supported", -.18), ("unsupported", .18)):
        selected = frame[frame.side.eq(side)].set_index("head_name").loc[heads]
        axis.bar(positions + offset, selected.support, width=.34, color=colors[side], label=side)
    axis.set_xticks(positions, heads)
    axis.axhline(0, color="#555555", lw=.8)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel("Full - cut candidate margin (nats)")
    axis.set_ylim(-2.1, 2.1)
    axis.legend(frameon=False, fontsize=9)


def restoration_panel(axis, frame):
    selected = frame[frame.phase.eq("onset") & frame.early.eq("L17H27_head_total")]
    sites = ["message", "mlp"]
    for side, offset, color in (("supported", -.18, "#246A9C"), ("unsupported", .18, "#C56A29")):
        values = selected[selected.side.eq(side)].set_index("site").loc[sites, "restoration_gain"]
        bars = axis.bar(np.arange(2) + offset, values, .34, label=side, color=color)
        axis.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
    axis.axhline(0, color="#555555", lw=.8)
    axis.set_xticks([0, 1], ["Restore L18H22 message", "Restore layer-18 MLP"])
    axis.set_title("C  Headwear: restore after cutting L17H27", loc="left", fontweight="bold")
    axis.set_ylabel("Restored - removed candidate margin (nats)")
    axis.set_ylim(-.62, .57)
    axis.legend(frameon=False, fontsize=9)


def persistence_panel(axis, comparison):
    selected = comparison[comparison.case_id.eq("14315_headwear_scope")
        & comparison.side.eq("unsupported") & comparison.phase.eq("back_half")
        & comparison.selection.eq("paired_gradient")].sort_values(["layer", "head"])
    names = [f"L{row.layer}H{row.head}" for row in selected.itertuples()]
    positions = np.arange(len(selected))
    axis.bar(positions - .18, selected.current_support, .34, color="#725B96", label="Cut at current receiver")
    axis.bar(positions + .18, selected.carried_support, .34, color="#8A959D", label="Carry onset cut only")
    axis.axhline(0, color="#555555", lw=.8)
    axis.set_xticks(positions, names)
    axis.set_title("D  Headwear unsupported: back half", loc="left", fontweight="bold")
    axis.set_ylabel("Full - cut actual-token log probability (nats)")
    axis.set_ylim(-.18, .45)
    axis.legend(frameon=False, fontsize=9)


def plot_review(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    effects = pd.read_csv(output / "onset_effects.csv")
    selected = effects[effects.source_group.eq("head_total") & effects.dose.eq(1)
        & effects.control.isna() & effects.selection.eq("paired_gradient")].copy()
    selected["head_name"] = [f"L{row.layer}H{row.head}" for row in selected.itertuples()]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        grouped_bars(axes[0, 0], selected[selected.case_id.eq("14315_headwear_scope")],
            ["L17H27", "L18H20", "L18H22", "L31H14"], "A  Headwear: onset head effects")
        grouped_bars(axes[0, 1], selected[selected.case_id.eq("14375_onion_stage")],
            ["L12H1", "L12H2", "L12H12", "L12H31"], "B  Onion: onset head effects")
        restoration_panel(axes[1, 0], pd.read_csv(output / "restoration.csv"))
        persistence_panel(axes[1, 1], pd.read_csv(output / "persistence_vs_current.csv"))
        figure.suptitle("Context-dependent head effects in two paired cases", fontsize=15, y=.98)
        figure.text(.05, .015, "Two source questions; full deletions; locally reviewed clauses. "
            "Onion candidates compare heat with duration. Panel D uses a different readout.", fontsize=9)
        figure.tight_layout(rect=(0, .055, 1, .95), h_pad=3)
        figure.savefig(output / "head_review.png", dpi=180)
        plt.close(figure)


def review(source, output, minimum_effect):
    output.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_csv(source / (name + ".csv"), dtype={"numeric_ok": "boolean"})
              for name in ("effects", "interactions", "adaptation", "persistence")}
    checks = dict(control_rows=control_counts(tables), arithmetic=arithmetic_checks(tables),
        minimum_effect_nats=minimum_effect, purpose="descriptive_review_no_new_model_run")
    (output / "numeric_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False))
    save_tables(source, output, tables, minimum_effect)
    plot_review(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Extracted paired_review directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-effect", type=float, default=.05)
    args = parser.parse_args()
    review(args.input, args.output, args.minimum_effect)
