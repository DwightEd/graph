"""Observed routing transitions, using the same source sets for current and past rows."""

import numpy as np


def reanchor_rows(
    traces, prompt_length, special_ids, prefix_ids, recent=10, shift=0.10
):
    rows = []
    previous_confirmed = None
    for index, trace in enumerate(traces):
        if index < 3:
            continue
        prior = traces[index - 3 : index]
        query = int(trace["query"])
        if [int(item["query"]) for item in prior] != list(range(query - 3, query)):
            continue
        count = query + 1
        old_end = max(prompt_length, count - recent)
        ordinary = ~np.isin(prefix_ids[:count], special_ids)
        old = ordinary & (np.arange(count) < old_end)
        local = ordinary & ~old
        current_old = trace["attention"][..., old].sum(-1)
        current_local = trace["attention"][..., local].sum(-1)
        past_old = np.mean(
            [p["attention"][..., old[: int(p["query"]) + 1]].sum(-1) for p in prior], 0
        )
        past_local = np.mean(
            [p["attention"][..., local[: int(p["query"]) + 1]].sum(-1) for p in prior],
            0,
        )
        confirmed = (
            (past_local > past_old)
            & (current_old > current_local)
            & (current_old - past_old >= shift)
            & (past_local - current_local >= shift)
        )
        rows.extend(
            event_rows(
                query,
                confirmed,
                previous_confirmed,
                past_old,
                current_old,
                past_local,
                current_local,
            )
        )
        previous_confirmed = confirmed
    return rows


def event_rows(
    query, confirmed, previous, past_old, current_old, past_local, current_local
):
    rows = []
    for layer, head in np.ndindex(confirmed.shape):
        rows.append(
            {
                "query": query,
                "predicted_position": query + 1,
                "layer": layer,
                "head": head,
                "previous_old": float(past_old[layer, head]),
                "current_old": float(current_old[layer, head]),
                "previous_local": float(past_local[layer, head]),
                "current_local": float(current_local[layer, head]),
                "confirmed": bool(confirmed[layer, head]),
                "first_confirmation": bool(
                    confirmed[layer, head]
                    and (previous is None or not previous[layer, head])
                ),
                "prior_state_observed": previous is not None,
            }
        )
    return rows
