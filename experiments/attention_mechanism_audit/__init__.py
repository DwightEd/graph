"""Frozen-model, answer-level attention hallucination mechanism audit.

The package keeps three axes separate: grounding drift, routing/functional
dispersion, and counterfactual evidence bypass.  It is an audit pipeline, not a
graph encoder and not an unsupervised detector trained on hallucination labels.
"""

from .artifacts import MechanismArtifact, load_artifact, save_artifact

__all__ = ["MechanismArtifact", "load_artifact", "save_artifact"]
