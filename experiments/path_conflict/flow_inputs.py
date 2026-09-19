"""Add aggregate functional source groups without changing primitive source roles."""

import numpy as np


def add_flow_groups(probe):
    result = dict(probe)
    groups = dict(probe["groups"])
    groups["condition"] = groups["scope"].copy()
    groups["value"] = groups["supported_value"].copy()
    groups["evidence"] = np.unique(np.concatenate(
        [groups["condition"], groups["value"]]
    ))
    groups["history"] = np.unique(np.concatenate(
        [groups["query_self"], groups["recent_history"], groups["remote_history"]]
    ))
    groups["past_history"] = np.unique(np.concatenate(
        [groups["recent_history"], groups["remote_history"]]
    ))
    groups["wrong_source"] = groups["value_source"].copy()
    groups["all_context"] = np.arange(len(probe["prefix_ids"]))
    result["groups"] = groups
    return result
