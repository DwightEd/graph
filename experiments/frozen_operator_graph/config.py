"""Configuration for exact, label-free frozen-operator graph construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraphConstructionConfig:
    """Deterministic construction parameters.

    Retention changes only which individual token-to-token edges are exposed.
    Every unexposed source is still consumed and conserved in an exact
    role-specific quotient edge, so neither attention mass nor vector message
    is discarded.
    """

    route_mass_retention: float = 1.0
    value_energy_retention: float = 1.0
    minimum_role_edges: int = 1
    conservation_atol: float = 5e-3
    conservation_rtol: float = 5e-3
    cache_binding_atol: float = 5e-3
    feature_epsilon: float = 1e-8
    output_dtype: str = "float32"

    def validate(self) -> "GraphConstructionConfig":
        for name, value in (
            ("route_mass_retention", self.route_mass_retention),
            ("value_energy_retention", self.value_energy_retention),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if int(self.minimum_role_edges) < 1:
            raise ValueError("minimum_role_edges must be at least one")
        if not 0.0 < float(self.conservation_atol) < 0.1:
            raise ValueError("conservation_atol must lie in (0, 0.1)")
        if not 0.0 <= float(self.conservation_rtol) < 0.1:
            raise ValueError("conservation_rtol must lie in [0, 0.1)")
        if not 0.0 < float(self.cache_binding_atol) < 0.1:
            raise ValueError("cache_binding_atol must lie in (0, 0.1)")
        if not 0.0 < float(self.feature_epsilon) < 1e-2:
            raise ValueError("feature_epsilon must lie in (0, 1e-2)")
        if self.output_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("output_dtype must be float32, float16, or bfloat16")
        return self

    def as_dict(self) -> dict[str, object]:
        return asdict(self.validate())
