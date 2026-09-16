"""Join the preceding learned_audit's saved interventions without rerunning them."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def attach_learned_controls(samples, directory):
    """Only identical original answers/scores may share a token-level audit table."""
    directory = Path(directory)
    if not (directory / "samples").is_dir():
        return samples, dict(status="unavailable", path=str(directory), loaded_answers=0)
    result, loaded, controls = [], [], set()
    for sample in samples:
        path = directory / "samples" / (sample["id"] + ".npz")
        values = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                identity = json.loads(str(saved["identity"]))
                for key in ("id", "source_id"):
                    if str(identity[key]) != sample[key]:
                        raise ValueError("learned_audit identity differs: " + key)
                for key in ("score", "gold", "onset", "offsets"):
                    if not np.array_equal(saved[key], sample[key]):
                        raise ValueError("learned_audit original token data differs: " + key)
                if str(saved["response"]) != sample["response"]:
                    raise ValueError("learned_audit response differs")
                for key in saved.files:
                    if key.startswith("control_"):
                        logits = saved[key]
                        if logits.shape != sample["score"].shape:
                            raise ValueError("learned_audit control is not token aligned: " + key)
                        values["score_learned_" + key[8:]] = expit(logits)
            loaded.append(sample["id"])
            controls.update(values)
        result.append(dict(sample, **values))
    return result, dict(status="completed" if len(loaded) == len(samples) else "partial",
                        path=str(directory), loaded_answers=len(loaded), total_answers=len(samples),
                        loaded_ids=loaded, controls=sorted(controls),
                        note="Existing logits converted to probabilities; no intervention rerun; each control uses common coordinates.")
