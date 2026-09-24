"""Read complete v1/v2 carrier caches without opening annotations."""

from contextlib import closing

import numpy as np

from ..choice_cache import CaptureReader
from ..evidence_contrast.aggregation import aggregate_scores
from ..message_carriers.representation import METHODS as UNIT_METHODS, TOKEN_METHODS as UNIT_TOKENS
from ..message_carriers.token_report import method_names

EDGE_FIELDS = ("target", "layer", "head", "receiver", "key", "sham_key",
               "effect_with_source", "effect_without_source")


def read_dataset(path):
    with closing(CaptureReader(path)) as reader:
        settings, protocol = reader.json("settings.json"), reader.json("protocol.json")
        versions = {"message-carriers-v1": "source_selected",
                    "message-carriers-token-v2": "token_source_selected"}
        carrier_score = versions[protocol["version"]]
        records = []
        for index, response in enumerate(settings["responses"]):
            directory = f"responses/{index:04d}"
            views, scores = reader.json(directory + "/views.json"), reader.arrays(directory + "/scores.npz")
            validated = aggregate_scores(scores, views)
            scores.update(validated)
            if not np.isfinite(scores[carrier_score]).all():
                raise ValueError("Nonfinite carrier scores")
            original = response["token_ids"][response["prompt_length"]:]
            if not np.array_equal(original, scores["token_id"]):
                raise ValueError(f"{response['id']}: original token identities differ")
            records.append(dict(response=response, views=views, scores=scores,
                carrier_score=carrier_score, directory=directory))
    return settings, protocol, records


def baseline_methods(protocol):
    if protocol["version"] == "message-carriers-v1":
        return list(UNIT_TOKENS), list(UNIT_METHODS)
    tokens, methods, _ = method_names(protocol)
    return tokens, methods


def edge_block(saved, targets, token_mode):
    """V1 effects are whole-unit query bundles; receiver=-1 records that distinction."""
    identity, sham = saved["edges"], saved["sham_edges"]
    if token_mode:
        effects = saved["keep"][:, None, 0].astype(float) - saved["single"][:, :, 0]
        receivers = identity[:, 2]
        effects = effects[:, :, None]
    else:
        effects = saved["keep"][:, None, :].astype(float) - saved["single"]
        receivers = np.full(len(identity), -1)
    rows = [np.tile(targets, len(identity)), np.repeat(identity[:, 0], len(targets)),
            np.repeat(identity[:, 1], len(targets)), np.repeat(receivers, len(targets)),
            np.repeat(identity[:, -1], len(targets)), np.repeat(sham[:, -1], len(targets)),
            effects[0].ravel(), effects[1].ravel()]
    return dict(zip(EDGE_FIELDS, rows))


def read_edges(reader, record, protocol):
    directory, count = record["directory"], len(record["scores"]["target"])
    token_mode = protocol["version"] == "message-carriers-token-v2"
    if token_mode:
        blocks = [edge_block(reader.arrays(f"{directory}/token_{target:06d}.npz"),
                             np.asarray([target]), True) for target in range(count)]
    else:
        blocks = [edge_block(reader.arrays(f"{directory}/unit_{unit['start']:06d}/measurements.npz"),
                             np.arange(unit["start"], unit["stop"]), False) for unit in record["views"]["units"]]
    edges = {name: np.concatenate([block[name] for block in blocks]) for name in EDGE_FIELDS}
    for name in EDGE_FIELDS[:6]:
        edges[name] = edges[name].astype(np.int64)
    if not all(np.isfinite(value).all() for value in edges.values()):
        raise ValueError("Nonfinite carrier edge measurements")
    if not ((edges["key"] >= 0) & (edges["key"] < edges["target"]) &
            (edges["sham_key"] >= 0) & (edges["sham_key"] < edges["target"])).all():
        raise ValueError("Carrier graph contains non-history endpoints")
    return edges
