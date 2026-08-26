"""Compatibility entry point for saved GroundedRoute graph bundles.

The active evaluation code lives in ``experiments.grounded_route.evaluation``.
This package only keeps old DBGNN checkpoints runnable after the evaluation
refactor.
"""

from .data import GraphBundle, GraphRecord, load_bundle

__all__ = ["GraphBundle", "GraphRecord", "load_bundle"]
