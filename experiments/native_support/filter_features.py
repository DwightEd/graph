"""Extract compact head profiles once; no repeated full-array or model work."""

import numpy as np
from state_audit.storage import read_arrays, write_arrays
from tqdm import tqdm

from .filtering import head_profile
from .pipeline import verify_position
from .routes import ROUTE_SCORES, route_scores

READ_FIELDS = (
    "attention", "edge_value_energy", "group_ids", "query", "target", "observed_id",
    "entropy", "ledger_error",
)


def extract_features(response, directory, evidence):
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    collected = []
    for target in tqdm(range(count), desc=response["id"], leave=False):
        with np.load(directory / f"token_{target:06d}.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in READ_FIELDS}
        verify_position(arrays, response, target)
        magnitude = np.sqrt(np.maximum(arrays["edge_value_energy"], 0))
        measured = route_scores(arrays["attention"], magnitude, arrays["group_ids"], prompt, evidence)
        profile = head_profile(magnitude, arrays["group_ids"], prompt, evidence)
        collected.append({**measured, "head_profile": profile, "entropy": arrays["entropy"],
                          "ledger_error": arrays["ledger_error"]})
    result = {name: np.stack([row[name] for row in collected]) for name in collected[0]}
    result.update(all_token_ids=np.asarray(response["token_ids"]),
                  source_mask=np.asarray(evidence) if evidence is not None else np.empty(0, bool),
                  token_id=np.asarray(response["token_ids"][prompt:]),
                  target=np.arange(count), query=np.arange(prompt - 1, prompt + count - 1))
    return result


def cached_features(response, raw_directory, cache_path, evidence, fields=None):
    if not cache_path.is_file():
        features = extract_features(response, raw_directory, evidence)
        write_arrays(cache_path, **features)
        return features
    if fields is None:
        features = read_arrays(cache_path)
    else:
        with np.load(cache_path, allow_pickle=False) as saved:
            names = tuple(dict.fromkeys((*fields, "all_token_ids", "source_mask")))
            features = {name: saved[name] for name in names}
    mask = np.asarray(evidence) if evidence is not None else np.empty(0, bool)
    if not np.array_equal(features["all_token_ids"], response["token_ids"]):
        raise ValueError(f"{response['id']}: routing feature token IDs differ from saved input")
    if not np.array_equal(features["source_mask"], mask):
        raise ValueError(f"{response['id']}: routing feature source mask differs")
    return features


def score_arrays(features, filtered):
    fields = (*ROUTE_SCORES, "entropy", "target", "query", "token_id", "ledger_error")
    return {**{name: features[name] for name in fields}, **filtered}


def token_rows(response, scores, methods):
    rows = []
    diagnostics = ("filter_current_weight", "filter_effective_tokens", "filter_bandwidth", "ledger_error")
    for target in range(len(scores["token_id"])):
        values = {name: float(scores[name][target]) for name in (*methods, *diagnostics)}
        rows.append({"response_id": response["id"], "source_id": response["source_id"],
                     "target": target, "query": int(scores["query"][target]),
                     "token_id": int(scores["token_id"][target]),
                     "token": response["token_text"][response["prompt_length"] + target], **values})
    return rows
