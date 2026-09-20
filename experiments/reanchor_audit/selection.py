"""Select using native routes only; attach reviewed roles after selection."""

import numpy as np
import pandas as pd


def ranked_edges(graph, flow):
    source = graph["source"]
    prefix = int(graph["prompt_length"])
    receivers = np.arange(source.shape[2])[None, None, :, None]
    valid = (graph["attention"] > 0) & (source < receivers) & (receivers >= prefix)
    layer, head, receiver, slot = np.where(valid)
    positions = source[layer, head, receiver, slot]
    return pd.DataFrame(dict(layer=layer, head=head, receiver=receiver, source=positions,
        slot=slot, attention_mass=graph["attention"][valid],
        value_energy=graph["value_energy"][layer, head, positions],
        message_norm=graph["attention"][valid] * graph["projected_value_norm"][layer, head, positions],
        lookback_gain=graph["lookback_gain"][layer, head, receiver],
        target_flow=flow["attention_potential_edges"][valid],
        norm_flow=flow["message_norm_potential_edges"][valid],
        uniform_root_flow=flow["attention_uniform_edges"][valid],
        source_region=np.where(positions < prefix, "prompt", "history")))


def select_events(edges, count, target, radius=10):
    # An earlier reanchor must precede the decision query; direct reads are controls.
    earlier = edges[edges.receiver < target]
    positive = earlier[(earlier.lookback_gain > 0) & (earlier.target_flow > 0)]
    ordered = positive.sort_values(["target_flow", "layer", "head", "receiver", "source"],
                                   ascending=[False, True, True, True, True])
    chosen = ordered.drop_duplicates(["layer", "head", "receiver"]).head(count).copy()
    chosen["selection"] = "lookback_target_flow"
    used = set(zip(chosen.layer, chosen["head"], chosen.receiver))
    controls = []
    for row in chosen.itertuples():
        pool = earlier[(earlier.layer == row.layer) & (earlier["head"] == row.head)
            & (earlier.source_region == row.source_region)
            & ((earlier.receiver - row.receiver).abs() <= radius)
            & (np.floor(np.log2(earlier.receiver - earlier.source)) == np.floor(np.log2(row.receiver - row.source)))
            & (earlier.target_flow < row.target_flow)].sort_values("target_flow")
        for control in pool.to_dict("records"):
            key = (control["layer"], control["head"], control["receiver"])
            if key not in used:
                controls.append(dict(control, selection="matched_low_flow", matched_receiver=row.receiver))
                used.add(key)
                break
    no_gain = earlier[(earlier.lookback_gain <= 0) & (earlier.target_flow > 0)].sort_values("target_flow", ascending=False)
    unused = [key not in used for key in zip(no_gain.layer, no_gain["head"], no_gain.receiver)]
    no_gain = no_gain.loc[unused]
    no_gain = no_gain.drop_duplicates(["layer", "head", "receiver"]).head(1).copy()
    no_gain["selection"] = "high_flow_without_positive_lookback"
    direct = edges[edges.receiver == target].sort_values("target_flow", ascending=False).head(1).copy()
    direct["selection"] = "decision_query_control"
    frames = [frame for frame in (chosen, pd.DataFrame(controls), no_gain, direct) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else chosen


def find_relay(edges, event):
    candidates = edges[(edges.layer > event["layer"]) & (edges.source == event["receiver"])
        & (edges.receiver > event["receiver"]) & (edges.target_flow > 0)]
    if candidates.empty:
        return None
    return candidates.sort_values("target_flow", ascending=False).iloc[0].to_dict()


def unit_from_edge(row, name):
    return dict(unit=name, layer=int(row["layer"]), head=int(row["head"]),
                receiver=int(row["receiver"]), sources=[int(row["source"])])


def make_plan(events, edges, roles):
    plan = []
    clean = events.astype(object).where(events.notna(), None)
    for number, row in enumerate(clean.to_dict("records")):
        entry = unit_from_edge(row, f"event_{number}_selected_edge")
        entry["capture_residual"] = True
        relay = find_relay(edges, row)
        units = [dict(entry, source_group="selected_edge")]
        for role, sources in roles.items():
            units.append(dict(entry, unit=f"event_{number}_{role}",
                              sources=list(map(int, sources)), source_group=role))
        joined = sorted(map(int, set(roles["scope"]) | set(roles["supported_value"])))
        units.append(dict(entry, unit=f"event_{number}_scope_and_value", sources=joined,
                          source_group="scope_and_value"))
        plan.append(dict(event=number, selected=row, entry=entry, source_units=units,
            relay=None if relay is None else unit_from_edge(relay, f"event_{number}_relay"),
            relay_status="not_measured_no_retained_cross_position_edge" if relay is None else "planned"))
    return plan


def describe_edges(edges, token_text, roles, prompt_length):
    frame = edges.copy()
    labels = {index: [] for index in range(len(token_text))}
    for role, positions in roles.items():
        for index in positions:
            labels[int(index)].append(role)
    frame["source_text"] = [token_text[index] for index in frame.source]
    frame["receiver_text"] = [token_text[index] for index in frame.receiver]
    frame["source_roles"] = ["|".join(labels[index]) or (
        "other_prompt" if index < prompt_length else "prior_history") for index in frame.source]
    return frame
