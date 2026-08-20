"""Feature registry for the proposed fracture-to-lock-in mechanism."""

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
    FeatureSpec("prompt_mass_mean", "access", onset_direction=-1),
    FeatureSpec("prompt_effective_sources_mean", "access", onset_direction=-1),
    FeatureSpec("prompt_top1_share_mean", "access", onset_direction=1),
    FeatureSpec("prompt_source_velocity_mean", "access", onset_direction=1),
    FeatureSpec(
        "prompt_response_head_disagreement",
        "fracture",
        onset_direction=1,
        lockin_direction=-1,
    ),
    FeatureSpec("prompt_mass_head_std", "fracture", onset_direction=1),
    FeatureSpec("prompt_provenance_head_std", "fracture", onset_direction=1),
    FeatureSpec("prompt_anchor_head_agreement", "fracture", onset_direction=-1),
    FeatureSpec(
        "prompt_provenance_lower_mean",
        "integration",
        onset_direction=-1,
        lockin_direction=-1,
    ),
    FeatureSpec("prompt_provenance_uncertainty", "integration", onset_direction=1),
    FeatureSpec(
        "unsupported_response_mass_mean",
        "integration",
        onset_direction=1,
        lockin_direction=1,
    ),
    FeatureSpec("response_takeover_mean", "lockin", onset_direction=1),
    FeatureSpec("response_effective_sources_mean", "lockin", lockin_direction=-1),
    FeatureSpec("response_top1_share_mean", "lockin", lockin_direction=1),
    FeatureSpec("recent_response_share_mean", "lockin", lockin_direction=1),
    FeatureSpec("response_mean_lag_mean", "lockin", lockin_direction=-1),
    FeatureSpec(
        "response_source_velocity_mean",
        "lockin",
        onset_direction=1,
        lockin_direction=-1,
    ),
    FeatureSpec("response_anchor_head_agreement", "lockin", lockin_direction=1),
    FeatureSpec("self_mass_mean", None, control=True),
    FeatureSpec("unresolved_mass_mean", None, control=True),
    FeatureSpec("known_mass_mean", None, control=True),
)

FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FAMILY_NAMES = ("access", "fracture", "integration", "lockin", "all")
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
