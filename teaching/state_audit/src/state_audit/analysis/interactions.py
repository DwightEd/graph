"""Pure numerical decompositions; no model, labels, or automatic causal naming."""


def factorial_effects(values):
    """Use explicit retained-state keys 11, 01, 10, 00."""
    return dict(
        **{f"F{key}": float(values[key]) for key in ("11", "01", "10", "00")},
        a_with_b=values["11"] - values["01"],
        a_without_b=values["10"] - values["00"],
        b_with_a=values["11"] - values["10"],
        b_without_a=values["01"] - values["00"],
        interaction=values["11"] - values["10"] - values["01"] + values["00"],
    )


def adaptation_effects(baseline, deleted, restored):
    """Separate the total deletion effect and the change due to downstream adaptation."""
    total = deleted - baseline
    fixed_downstream = restored - baseline
    return dict(
        baseline=baseline,
        upstream_deleted=deleted,
        downstream_restored=restored,
        total_effect=total,
        fixed_downstream_effect=fixed_downstream,
        adaptation=deleted - restored,
        absolute_effect_reduction=abs(fixed_downstream) - abs(total),
    )
