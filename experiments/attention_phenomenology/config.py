"""Configuration and feature registry for the attention phenomenology audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhenomenologyConfig:
    """Settings shared by representation, reference, and null-model stages."""

    prompt_bins: int = 4
    rr_lag_bins: int = 8
    lid_neighbors: int = 8
    transition_projections: int = 12
    anchor_count: int = 8
    recent_lag_max: int = 4
    causal_position_bins: int = 10
    block_rows: int = 8192
    random_seed: int = 20260819
    epsilon: float = 1e-8

    @property
    def role_count(self) -> int:
        return self.prompt_bins + self.rr_lag_bins + 2

    @property
    def self_role(self) -> int:
        return self.prompt_bins + self.rr_lag_bins

    @property
    def unresolved_role(self) -> int:
        return self.self_role + 1

    @property
    def role_names(self) -> tuple[str, ...]:
        prompt = tuple(f"prompt_bin_{index}" for index in range(self.prompt_bins))
        history = tuple(f"rr_lag_bin_{index}" for index in range(self.rr_lag_bins))
        return prompt + history + ("self", "unresolved")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FEATURE_NAMES = (
    # Head-set geometry: attention analogue of high-dimensionality/fracture.
    "ph_mean_death",
    "ph_max_death",
    "ph_persistence_entropy",
    "ph_largest_gap",
    "head_route_effective_rank",
    "head_local_intrinsic_dimension",
    # Prompt detection and integration across heads/layers.
    "direct_prompt_mean",
    "direct_prompt_std",
    "grounding_lower_mean",
    "grounding_lower_std",
    "grounding_upper_mean",
    "grounding_interval_width",
    "unsupported_feedback_lower",
    "unsupported_feedback_upper",
    # Response-only concentration and temporal lock-in.
    "response_source_effective_number",
    "response_source_top1_share",
    "recent_response_mass_fraction",
    "response_mean_lag",
    "response_anchor_turnover",
    "layer_headset_transition",
    "temporal_headset_transition",
    # Censoring and identity controls. These are not fused into mechanism scores.
    "self_mass_mean",
    "unresolved_mass_mean",
    "known_mass_mean",
)

FAMILY_FEATURES = {
    "fracture": (
        "ph_mean_death",
        "ph_max_death",
        "ph_persistence_entropy",
        "ph_largest_gap",
        "head_route_effective_rank",
        "head_local_intrinsic_dimension",
        "direct_prompt_std",
        "layer_headset_transition",
    ),
    "integration": (
        "direct_prompt_mean",
        "direct_prompt_std",
        "grounding_lower_mean",
        "grounding_lower_std",
        "grounding_upper_mean",
        "grounding_interval_width",
        "unsupported_feedback_lower",
        "unsupported_feedback_upper",
    ),
    "lockin": (
        "response_source_effective_number",
        "response_source_top1_share",
        "recent_response_mass_fraction",
        "response_mean_lag",
        "response_anchor_turnover",
        "temporal_headset_transition",
        "grounding_lower_mean",
        "unsupported_feedback_lower",
    ),
}

FAMILY_NAMES = tuple(FAMILY_FEATURES) + ("all",)


# Pre-registered phase predictions. Zero means that no global direction is claimed;
# the feature is still evaluated as an unsigned anomaly coordinate.
ONSET_DIRECTIONS = {
    "ph_mean_death": 1,
    "ph_max_death": 1,
    "ph_persistence_entropy": 1,
    "ph_largest_gap": 1,
    "head_route_effective_rank": 1,
    "head_local_intrinsic_dimension": 1,
    "direct_prompt_std": 1,
    "grounding_lower_mean": -1,
    "grounding_lower_std": 1,
    "grounding_interval_width": 1,
    "unsupported_feedback_lower": 1,
    "layer_headset_transition": 1,
    "temporal_headset_transition": 1,
}

LOCKIN_DIRECTIONS = {
    "grounding_lower_mean": -1,
    "unsupported_feedback_lower": 1,
    "response_source_effective_number": -1,
    "response_source_top1_share": 1,
    "recent_response_mass_fraction": 1,
    "response_anchor_turnover": -1,
    "temporal_headset_transition": -1,
}
