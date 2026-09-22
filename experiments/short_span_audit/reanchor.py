"""Join confirmed old reanchor events; an unconfirmed event is never called absent."""

import csv
from collections import defaultdict


def read_events(directory, cohort, threshold):
    records = {str(answer["record"]["id"]): answer["record"] for answer in cohort}
    events = defaultdict(set)
    with (directory / "nodes.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row["id"]
            if identity not in records or float(row["threshold"]) != threshold:
                continue
            record = records[identity]
            target = int(row["target"])
            query = int(row["query"])
            if (row["source_id"] != str(record["source_id"])
                    or row["task"] != record["task"] or row["generator"] != record["generator"]
                    or query != int(record["prompt_length"]) + target - 1):
                raise ValueError(f"Reanchor identity/query mismatch for {identity}")
            events[identity].add(target)
    return events


def reanchor_rows(directory, cohort, threshold=.1, window=8):
    events = read_events(directory, cohort, threshold)
    rows = []
    for answer in cohort:
        record = answer["record"]
        observed = events[str(record["id"])]
        for membership in answer["targets"]:
            if membership["offset"] != 0:
                continue
            target = membership["target"]
            prior = [position for position in observed if max(0, target - window) <= position < target]
            rows.append(dict(id=record["id"], source_id=record["source_id"], task=record["task"],
                             generator=record["generator"], **membership, threshold=threshold,
                             prior_window=window, prior_confirmed_event=bool(prior),
                             decision_confirmed_event=target in observed,
                             nearest_prior_distance=target - max(prior) if prior else None,
                             interpretation="confirmed_local_to_old_switch_or_unconfirmed"))
    return rows
