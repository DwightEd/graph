"""Read baseline token features from existing caches, without head filtering."""

import numpy as np
from state_audit.storage import read_arrays, write_arrays
from tqdm import tqdm

from .pipeline import verify_position
from .routes import route_scores

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
        measured.update(entropy=arrays["entropy"], ledger_error=arrays["ledger_error"])
        collected.append(measured)
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
