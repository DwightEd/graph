"""Read selected arrays from a complete value-path directory or compact ZIP."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from state_audit.storage import read_json

FIELDS = (
    "root_positive", "root_negative", "root_token_choice", "group_ids",
    "candidate_ids", "candidate_logits", "candidate_tail_mass", "margin",
    "ledger_error", "target", "query", "token_id", "entropy", "surprisal",
)


class CaptureReader:
    """A ZIP is read in place; no extraction or dense attention reconstruction."""

    def __init__(self, path):
        self.path = Path(path)
        self.archive = None if self.path.is_dir() else ZipFile(self.path)

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def exists(self, name):
        if self.archive is not None:
            return name in self.archive.namelist()
        return (self.path / name).is_file()

    def json(self, name):
        if self.archive is not None:
            return json.loads(self.archive.read(name))
        return read_json(self.path / name)

    def bytes(self, name):
        if self.archive is not None:
            return self.archive.read(name)
        return (self.path / name).read_bytes()

    def arrays(self, name, fields=None):
        source = self.path / name
        if self.archive is not None:
            source = io.BytesIO(self.archive.read(name))
        with np.load(source, allow_pickle=False) as saved:
            return {key: saved[key] for key in (saved.files if fields is None else fields)}

    def capture(self, name):
        source = self.path / name
        if self.archive is not None:
            source = io.BytesIO(self.archive.read(name))
        with np.load(source, allow_pickle=False) as saved:
            row = {key: saved[key] for key in FIELDS}
            if "head_group_attention" in saved.files:
                row["group_attention"] = saved["head_group_attention"]
            else:
                # Complete captures use key addresses; compact captures already grouped them.
                attention = saved["attention"]
                groups = row["group_ids"].copy()
                groups[-1] = len(row["root_positive"])
                row["group_attention"] = np.stack([
                    attention[..., groups == group].sum(-1)
                    for group in range(len(row["root_positive"]) + 1)
                ], axis=-1)
            return row


def validate_row(row, response, target, groups, source_count):
    """Validate scientific alignment once at the disk boundary."""
    prompt = response["prompt_length"]
    query = prompt + target - 1
    token = response["token_ids"][query + 1]
    if (int(row["query"]), int(row["target"]), int(row["token_id"])) != (query, target, token):
        raise ValueError(f"{response['id']}/{target}: prediction alignment differs")
    if not np.array_equal(row["group_ids"], groups[:query + 1]):
        raise ValueError(f"{response['id']}/{target}: root group alignment differs")
    candidates = row["candidate_ids"]
    if int(candidates[0]) != token or len(np.unique(candidates)) != len(candidates):
        raise ValueError(f"{response['id']}/{target}: invalid candidate identities")
    if row["root_token_choice"].shape != (query + 1, len(candidates) - 1):
        raise ValueError(f"{response['id']}/{target}: root/candidate axes differ")
    if row["root_positive"].shape != (source_count + 3, len(candidates) - 1):
        raise ValueError(f"{response['id']}/{target}: unexpected source groups")
    if not all(np.isfinite(value).all() for value in row.values()):
        raise ValueError(f"{response['id']}/{target}: nonfinite capture")
    positive = np.zeros_like(row["root_positive"], dtype=float)
    negative = np.zeros_like(positive)
    np.add.at(positive, row["group_ids"], np.maximum(row["root_token_choice"], 0))
    np.add.at(negative, row["group_ids"], np.maximum(-row["root_token_choice"], 0))
    if not (np.allclose(positive, row["root_positive"], atol=1e-5, rtol=1e-5)
            and np.allclose(negative, row["root_negative"], atol=1e-5, rtol=1e-5)):
        raise ValueError(f"{response['id']}/{target}: token/group provenance differs")
    margin = row["candidate_logits"][0] - row["candidate_logits"][1:]
    if not np.allclose(margin, row["margin"], atol=1e-5, rtol=1e-5):
        raise ValueError(f"{response['id']}/{target}: native conditional odds differ")
    residual = (positive - negative).sum(0) - row["margin"]
    if not np.allclose(residual, row["ledger_error"], atol=1e-5, rtol=1e-5):
        raise ValueError(f"{response['id']}/{target}: attribution ledger differs")
