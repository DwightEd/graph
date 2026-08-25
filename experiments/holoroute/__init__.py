"""HoloRoute: unsupervised learning on causal attention event graphs."""

from .baseline import Flat1024, build_pairs
from .config import HoloRouteConfig
from .detection import ConditionalReference, TokenResiduals, score_graph
from .graph import EventGraph, build_graph
from .learning import self_supervised_loss, train_model
from .model import HoloRoute

__all__ = [
    "ConditionalReference",
    "EventGraph",
    "Flat1024",
    "HoloRoute",
    "HoloRouteConfig",
    "TokenResiduals",
    "build_graph",
    "build_pairs",
    "score_graph",
    "self_supervised_loss",
    "train_model",
]
