"""Scientific feature registry and pre-registered phase predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str | None
    onset_direction: int = 0
    lockin_direction: int = 0
    control: bool = False


FEATURE_SPECS = (
    FeatureSpec("known_ph_mean_death", "fracture", onset_direction=1),
    FeatureSpec("known_ph_max_death", "fracture", onset_direction=1),
    FeatureSpec("known_ph_persistence_entropy", "fracture", onset_direction=1),
    FeatureSpec("known_ph_largest_gap", "fracture", onset_direction=1),
    FeatureSpec("known_head_route_effective_rank", "fracture", onset_direction=1),
    FeatureSpec("known_head_local_intrinsic_dimension", "fracture", onset_direction=1),
    FeatureSpec("full_ph_mean_death", None, control=True),
    FeatureSpec("full_ph_max_death", None, control=True),
    FeatureSpec("direct_prompt_mean", "integration"),
    FeatureSpec("direct_prompt_std", "integration", onset_direction=1),
    FeatureSpec(
        "grounding_lower_mean", "integration", onset_direction=-1, lockin_direction=-1
    ),
    FeatureSpec("grounding_lower_std", "integration", onset_direction=1),
    FeatureSpec("grounding_upper_mean", "integration"),
    FeatureSpec("grounding_interval_width", "integration", onset_direction=1),
    FeatureSpec(
        "unsupported_rr_lower", "integration", onset_direction=1, lockin_direction=1
    ),
    FeatureSpec("unsupported_rr_upper", "integration"),
    FeatureSpec(
        "response_source_effective_number", "lockin", lockin_direction=-1
    ),
    FeatureSpec("response_source_top1_share", "lockin", lockin_direction=1),
    FeatureSpec("recent_response_mass_fraction", "lockin", lockin_direction=1),
    FeatureSpec("response_mean_lag", "lockin"),
    FeatureSpec("response_anchor_turnover", "lockin", lockin_direction=-1),
    FeatureSpec("source_distribution_velocity", "lockin", lockin_direction=-1),
    FeatureSpec("layer_headset_transition", "fracture", onset_direction=1),
    FeatureSpec(
        "temporal_headset_transition", "lockin", onset_direction=1, lockin_direction=-1
    ),
    FeatureSpec("self_mass_mean", None, control=True),
    FeatureSpec("unresolved_mass_mean", None, control=True),
    FeatureSpec("known_mass_mean", None, control=True),
)

FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FAMILY_NAMES = ("fracture", "integration", "lockin", "all")
FAMILY_FEATURES = {
    family: tuple(spec.name for spec in FEATURE_SPECS if spec.family == family)
    for family in FAMILY_NAMES[:-1]
}
FAMILY_FEATURES["all"] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.control
)
ONSET_DIRECTIONS = {
    spec.name: spec.onset_direction for spec in FEATURE_SPECS if spec.onset_direction
}
LOCKIN_DIRECTIONS = {
    spec.name: spec.lockin_direction for spec in FEATURE_SPECS if spec.lockin_direction
}
