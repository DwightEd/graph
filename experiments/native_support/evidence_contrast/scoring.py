"""Frozen log-probability contrasts; no label, rank correction, or fitted fusion."""

import numpy as np

from ..dual_state.scoring import window_mean
from .views import query_positions

BASELINES = ("raw_route", "observable_route", "raw_attention", "entropy")
CONTROLS = ("raw_route", "raw_route_offline_mean", "observable_route", "observable_route_offline_mean")
CANDIDATES = ("source_full", "source_local", "source_pair", "source_pair_span", "surprisal_full")
METHODS = (*BASELINES, "raw_route_offline_mean", "observable_route_offline_mean", *CANDIDATES)
PRIMARY = "source_pair"


def score_contrasts(with_source, without_source, views, baselines, window):
    """Negative source effect is a risk hypothesis, never a truth probability."""
    tokens = np.asarray(views["answer_ids"])
    for observation in (with_source, without_source):
        if not np.array_equal(observation["token_id"], tokens):
            raise ValueError("Four-condition target token identities differ")
        for name in ("full", "local"):
            if observation[name].shape != tokens.shape or not np.isfinite(observation[name]).all():
                raise ValueError(f"Missing/nonfinite {name} log probabilities")
    full_gain = with_source["full"].astype(float) - without_source["full"]
    local_gain = with_source["local"].astype(float) - without_source["local"]
    pair = -(full_gain + local_gain) / 2
    span = np.empty_like(pair)
    for unit in views["units"]:
        selected = slice(unit["start"], unit["stop"])
        span[selected] = pair[selected].mean()
    result = dict(source_full=-full_gain, source_local=-local_gain,
                  source_pair=pair, source_pair_span=span, surprisal_full=-with_source["full"],
                  source_history_interaction=full_gain-local_gain,
                  logp_source_history=with_source["full"], logp_history=without_source["full"],
                  logp_source_local=with_source["local"], logp_local=without_source["local"],
                  **baselines, **query_positions(views))
    for name in ("raw_route", "observable_route"):
        result[f"{name}_offline_mean"] = window_mean(baselines[name], window, offline=True)
    result["risk"] = baselines["raw_route"]
    return result
