"""Frozen, token-aligned minimal pairs for counterfactual route studies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

PAIR_SCHEMA = 1
CONSTRAINT_KINDS = frozenset({"entity", "negation", "time", "scope"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _required_text(record, name):
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _check_single_edit(record):
    plus = _required_text(record, "prompt_plus")
    minus = _required_text(record, "prompt_minus")
    constraint_plus = _required_text(record, "constraint_plus")
    constraint_minus = _required_text(record, "constraint_minus")
    if constraint_plus == constraint_minus:
        raise ValueError("the declared constraint texts must differ")
    if plus.count(constraint_plus) != 1 or minus.count(constraint_minus) != 1:
        raise ValueError("each declared constraint must occur exactly once in its world")
    plus_before, plus_after = plus.split(constraint_plus)
    minus_before, minus_after = minus.split(constraint_minus)
    if plus_before != minus_before or plus_after != minus_after:
        raise ValueError("the two prompts differ outside the declared constraint")
    return plus, minus, constraint_plus, constraint_minus


@dataclass(frozen=True)
class CounterfactualPair:
    """One reviewed semantic edit with frozen tokens and candidate orientation."""

    pair_id: str
    constraint_kind: str
    token_ids: np.ndarray
    response_start: int
    changed_positions: tuple[int, ...]
    candidate_plus_ids: tuple[int, ...]
    candidate_minus_ids: tuple[int, ...]
    constraint_plus: str
    constraint_minus: str
    response_prefix: str
    source_id: str
    dataset: str
    reviewed: bool = True
    labels_used: bool = False

    @classmethod
    def compile(cls, record, tokenizer):
        """Compile text once; later execution consumes only the frozen manifest."""

        if record.get("reviewed") is not True:
            raise ValueError("counterfactual pairs must be externally reviewed")
        pair_id = _required_text(record, "pair_id")
        kind = _required_text(record, "constraint_kind")
        plus, minus, constraint_plus, constraint_minus = _check_single_edit(record)
        response_prefix = _required_text(record, "response_prefix")
        prompt_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for prompt in (plus, minus)
        ]
        if len(prompt_ids[0]) != len(prompt_ids[1]):
            raise ValueError("counterfactual prompts must remain exactly token-aligned")
        response_ids = tokenizer.encode(response_prefix, add_special_tokens=False)
        if len(response_ids) < 2:
            raise ValueError("response_prefix needs at least two tokens for event propagation")
        token_ids = np.asarray(
            [list(ids) + list(response_ids) for ids in prompt_ids], dtype=np.int64
        )
        response_start = len(prompt_ids[0])
        changed = tuple(np.flatnonzero(token_ids[0] != token_ids[1]).tolist())
        candidate_plus = tuple(
            tokenizer.encode(" " + _required_text(record, "candidate_plus"), add_special_tokens=False)
        )
        candidate_minus = tuple(
            tokenizer.encode(" " + _required_text(record, "candidate_minus"), add_special_tokens=False)
        )
        return cls(
            pair_id=pair_id,
            constraint_kind=kind,
            token_ids=token_ids,
            response_start=response_start,
            changed_positions=changed,
            candidate_plus_ids=candidate_plus,
            candidate_minus_ids=candidate_minus,
            constraint_plus=constraint_plus,
            constraint_minus=constraint_minus,
            response_prefix=response_prefix,
            source_id=str(record.get("source_id", pair_id)),
            dataset=str(record.get("dataset", "externally_reviewed")),
        ).check()

    @classmethod
    def from_manifest(cls, manifest):
        if manifest.get("pair_schema") != PAIR_SCHEMA:
            raise ValueError("unsupported counterfactual pair schema")
        return cls(
            pair_id=manifest["pair_id"],
            constraint_kind=manifest["constraint_kind"],
            token_ids=np.asarray(manifest["token_ids"], dtype=np.int64),
            response_start=int(manifest["response_start"]),
            changed_positions=tuple(map(int, manifest["changed_positions"])),
            candidate_plus_ids=tuple(map(int, manifest["candidate_plus_ids"])),
            candidate_minus_ids=tuple(map(int, manifest["candidate_minus_ids"])),
            constraint_plus=manifest["constraint_plus"],
            constraint_minus=manifest["constraint_minus"],
            response_prefix=manifest["response_prefix"],
            source_id=manifest["source_id"],
            dataset=manifest["dataset"],
            reviewed=manifest["reviewed"],
            labels_used=manifest["labels_used"],
        ).check()

    def check(self):
        if not isinstance(self.pair_id, str) or not _SAFE_ID.fullmatch(self.pair_id):
            raise ValueError("pair_id must use only letters, digits, dot, dash or underscore")
        if self.constraint_kind not in CONSTRAINT_KINDS:
            raise ValueError(f"unsupported constraint_kind: {self.constraint_kind!r}")
        if self.reviewed is not True or self.labels_used is not False:
            raise ValueError("pairs must be reviewed and constructed without H/N labels")
        ids = np.asarray(self.token_ids)
        if ids.ndim != 2 or ids.shape[0] != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token_ids must contain two aligned integer worlds")
        if not 0 < self.response_start <= ids.shape[1] - 2:
            raise ValueError("response_start must leave a prompt and two response-prefix tokens")
        if not np.array_equal(ids[0, self.response_start:], ids[1, self.response_start:]):
            raise ValueError("the frozen worlds do not share an identical response prefix")
        changed = tuple(np.flatnonzero(ids[0] != ids[1]).tolist())
        if not changed or changed != tuple(self.changed_positions):
            raise ValueError("changed_positions must name every and only changed prompt token")
        if max(changed) >= self.response_start:
            raise ValueError("constraint edits cannot enter the shared response prefix")
        if not self.candidate_plus_ids or not self.candidate_minus_ids:
            raise ValueError("both candidate sequences must be nonempty")
        if self.candidate_plus_ids == self.candidate_minus_ids:
            raise ValueError("the evidence-oriented candidate sequences must differ")
        return self

    def to_manifest(self):
        self.check()
        return {
            "pair_schema": PAIR_SCHEMA,
            "pair_id": self.pair_id,
            "constraint_kind": self.constraint_kind,
            "token_ids": np.asarray(self.token_ids).tolist(),
            "response_start": self.response_start,
            "changed_positions": list(self.changed_positions),
            "candidate_plus_ids": list(self.candidate_plus_ids),
            "candidate_minus_ids": list(self.candidate_minus_ids),
            "constraint_plus": self.constraint_plus,
            "constraint_minus": self.constraint_minus,
            "response_prefix": self.response_prefix,
            "source_id": self.source_id,
            "dataset": self.dataset,
            "reviewed": self.reviewed,
            "labels_used": self.labels_used,
        }

    @property
    def study_identity(self):
        encoded = json.dumps(
            self.to_manifest(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()
