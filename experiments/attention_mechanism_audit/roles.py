"""Label-free prompt-role reconstruction for the RAGTruth attention cache.

The cache was produced with the historical system+user chat rendering in
``tokenize_ragtruth_sample``.  This module reconstructs that exact prefix from
``source_info.jsonl``, verifies it token-for-token against the cache, and
partitions every prompt token into evidence, question, constraint, or other
prompt material.  No response file or hallucination label is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np


ROLE_NAMES = ("evidence", "question", "constraint", "other_prompt")
EVIDENCE, QUESTION, CONSTRAINT, OTHER_PROMPT = range(len(ROLE_NAMES))
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}
ROLE_ARTIFACT_SCHEMA = "ragtruth-prompt-role-index-v1"
HISTORICAL_SYSTEM_PROMPT = "You are a helpful assistant."


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_record_sha256(value) -> str:
    """Public identity digest for one label-free ``source_info`` row."""

    return _canonical_sha256(value)


def prompt_token_sha256(token_ids) -> str:
    """Return a dtype-independent SHA256 for an ordered token-ID prefix."""

    tokens = np.asarray(token_ids)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("prompt token IDs must be a one-dimensional integer array")
    payload = json.dumps(
        [int(value) for value in tokens], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sample_role_permutation_seed(
    global_seed: int,
    sample_id: str,
    token_ids,
) -> int:
    """Derive an order-independent seed for one sample's role null.

    The seed depends on the complete sample identifier and ordered token
    sequence, rather than Python's process-randomized ``hash`` or the order in
    which samples happen to be traversed.
    """

    if isinstance(global_seed, (bool, np.bool_)) or not isinstance(
        global_seed, (int, np.integer)
    ):
        raise ValueError("global_seed must be an integer")
    sample_id = str(sample_id)
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    identity = {
        "global_seed": int(global_seed),
        "sample_id": sample_id,
        "token_ids_sha256": prompt_token_sha256(token_ids),
    }
    digest = bytes.fromhex(_canonical_sha256(identity))
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def position_stratified_role_permutation(
    role_ids,
    bin_width: int,
    seed: int,
) -> np.ndarray:
    """Permute prompt roles within distance-to-response position bins.

    Prompt position ``response_idx - 1`` has distance zero.  Positions are
    stratified by ``distance // bin_width`` before roles are permuted, so the
    null preserves every role count both globally and within every position
    bin.  A fresh array is always returned and ``role_ids`` is never modified.
    """

    roles = np.asarray(role_ids)
    if roles.ndim != 1 or roles.dtype.kind not in "iu":
        raise ValueError("role_ids must be a one-dimensional integer array")
    if roles.size == 0:
        raise ValueError("role_ids cannot be empty")
    if bool(((roles < 0) | (roles >= len(ROLE_NAMES))).any()):
        raise ValueError("role_ids contain an unknown role")
    if isinstance(bin_width, (bool, np.bool_)) or not isinstance(
        bin_width, (int, np.integer)
    ):
        raise ValueError("bin_width must be an integer")
    bin_width = int(bin_width)
    if bin_width < 2:
        raise ValueError("bin_width must be at least 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")

    permuted = roles.copy()
    distance = roles.size - 1 - np.arange(roles.size, dtype=np.int64)
    position_bin = distance // bin_width
    random = np.random.default_rng(int(seed))
    for current_bin in np.unique(position_bin):
        rows = np.flatnonzero(position_bin == current_bin)
        permuted[rows] = random.permutation(roles[rows])
    return permuted


def _role_runs(role_ids: np.ndarray) -> tuple[tuple[str, int, int], ...]:
    if role_ids.size == 0:
        return ()
    starts = np.flatnonzero(
        np.concatenate(([True], role_ids[1:] != role_ids[:-1]))
    )
    stops = np.concatenate((starts[1:], [role_ids.size]))
    return tuple(
        (ROLE_NAMES[int(role_ids[start])], int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    )


@dataclass(frozen=True)
class PromptRoleMap:
    """An exhaustive, disjoint role assignment for cached prompt positions."""

    source_id: str
    task_type: str
    response_idx: int
    role_ids: np.ndarray
    prompt_token_sha256: str
    source_info_sha256: str
    prompt_token_ids: np.ndarray | None = None

    @property
    def prompt_length(self) -> int:
        return self.response_idx

    @property
    def role_spans(self) -> tuple[tuple[str, int, int], ...]:
        return _role_runs(np.asarray(self.role_ids))

    def role_mask(self, role: str | int) -> np.ndarray:
        role_id = ROLE_TO_ID[role] if isinstance(role, str) else int(role)
        if not 0 <= role_id < len(ROLE_NAMES):
            raise ValueError(f"unknown prompt role: {role}")
        return np.asarray(self.role_ids) == role_id

    def validate(self, cached_token_ids=None) -> "PromptRoleMap":
        roles = np.asarray(self.role_ids)
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if self.response_idx < 1 or roles.shape != (self.response_idx,):
            raise ValueError("role_ids must cover exactly [0, response_idx)")
        if roles.dtype.kind not in "iu" or bool(
            ((roles < 0) | (roles >= len(ROLE_NAMES))).any()
        ):
            raise ValueError("role_ids contain an unknown role")
        hexadecimal = set("0123456789abcdef")
        if (
            len(self.prompt_token_sha256) != 64
            or len(self.source_info_sha256) != 64
            or not set(self.prompt_token_sha256).issubset(hexadecimal)
            or not set(self.source_info_sha256).issubset(hexadecimal)
        ):
            raise ValueError("role-map hashes must be SHA256 hex digests")
        if self.prompt_token_ids is not None:
            prompt = np.asarray(self.prompt_token_ids)
            if prompt.shape != (self.response_idx,) or prompt.dtype.kind not in "iu":
                raise ValueError("prompt_token_ids must match response_idx")
            if prompt_token_sha256(prompt) != self.prompt_token_sha256:
                raise ValueError("prompt token IDs do not match prompt_token_sha256")
        if cached_token_ids is not None:
            validate_cached_prompt(
                self.prompt_token_ids,
                cached_token_ids,
                self.response_idx,
                expected_sha256=self.prompt_token_sha256,
            )
        return self

    def to_json(self) -> dict:
        self.validate()
        if self.prompt_token_ids is None:
            raise ValueError("prompt_token_ids are required in a replayable role artifact")
        return {
            "schema": ROLE_ARTIFACT_SCHEMA,
            "source_id": self.source_id,
            "task_type": self.task_type,
            "response_idx": self.response_idx,
            "prompt_token_sha256": self.prompt_token_sha256,
            "prompt_token_ids": [int(value) for value in self.prompt_token_ids],
            "source_info_sha256": self.source_info_sha256,
            "role_spans": [
                {"role": role, "start": start, "end": end}
                for role, start, end in self.role_spans
            ],
        }

    @classmethod
    def from_json(cls, row: Mapping) -> "PromptRoleMap":
        if row.get("schema") != ROLE_ARTIFACT_SCHEMA:
            raise ValueError("unknown prompt-role artifact schema")
        response_idx = int(row["response_idx"])
        roles = expand_role_spans(row["role_spans"], response_idx)
        prompt_token_ids = np.asarray(row["prompt_token_ids"])
        if prompt_token_ids.ndim != 1 or prompt_token_ids.dtype.kind not in "iu":
            raise ValueError("prompt_token_ids must be an integer vector")
        return cls(
            source_id=str(row["source_id"]),
            task_type=str(row["task_type"]),
            response_idx=response_idx,
            role_ids=roles,
            prompt_token_sha256=str(row["prompt_token_sha256"]),
            source_info_sha256=str(row["source_info_sha256"]),
            prompt_token_ids=prompt_token_ids.astype(np.int64, copy=False),
        ).validate()


def expand_role_spans(spans: Iterable[Mapping], prompt_length: int) -> np.ndarray:
    """Expand sorted JSON spans and reject overlaps, gaps, or unknown roles."""

    prompt_length = int(prompt_length)
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    roles = np.full(prompt_length, -1, dtype=np.int8)
    cursor = 0
    for span in spans:
        role = str(span["role"])
        if role not in ROLE_TO_ID:
            raise ValueError(f"unknown prompt role: {role}")
        start, end = int(span["start"]), int(span["end"])
        if start != cursor or not start < end <= prompt_length:
            raise ValueError("role spans must be ordered and exactly partition the prompt")
        roles[start:end] = ROLE_TO_ID[role]
        cursor = end
    if cursor != prompt_length or bool((roles < 0).any()):
        raise ValueError("role spans must cover exactly [0, prompt_length)")
    return roles


def render_historical_prompt(tokenizer, source_prompt: str) -> str:
    """Render the system+user prefix used by the original cache extractor."""

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": HISTORICAL_SYSTEM_PROMPT},
            {"role": "user", "content": str(source_prompt)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer did not render a chat prompt")
    return rendered


def _flatten_encoding(encoding) -> tuple[np.ndarray, list[tuple[int, int]]]:
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if len(input_ids) and isinstance(input_ids[0], (list, tuple, np.ndarray)):
        input_ids = input_ids[0]
    if (
        len(offsets)
        and isinstance(offsets[0], (list, tuple, np.ndarray))
        and len(offsets[0])
        and isinstance(offsets[0][0], (list, tuple, np.ndarray))
    ):
        offsets = offsets[0]
    tokens = np.asarray(input_ids)
    normalized_offsets = [(int(start), int(end)) for start, end in offsets]
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("tokenizer input_ids must be a one-dimensional integer array")
    if len(normalized_offsets) != tokens.size:
        raise ValueError("tokenizer input IDs and offsets do not align")
    return tokens.astype(np.int64, copy=False), normalized_offsets


def tokenize_historical_prompt(tokenizer, source_prompt: str):
    """Return rendered text, exact token IDs, and character offsets."""

    rendered = render_historical_prompt(tokenizer, source_prompt)
    encoding = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    tokens, offsets = _flatten_encoding(encoding)
    return rendered, tokens, offsets


def validate_cached_prompt(
    expected_prompt_ids,
    cached_token_ids,
    prompt_length: int,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Require an exact rebuilt-prefix match and return its stable SHA256."""

    cached = np.asarray(cached_token_ids)
    prompt_length = int(prompt_length)
    if cached.ndim != 1 or cached.dtype.kind not in "iu":
        raise ValueError("cached_token_ids must be a one-dimensional integer array")
    if not 0 < prompt_length <= cached.size:
        raise ValueError("prompt_length is outside the cached sequence")
    prefix = cached[:prompt_length].astype(np.int64, copy=False)
    if expected_prompt_ids is not None:
        expected = np.asarray(expected_prompt_ids)
        if expected.shape != (prompt_length,) or expected.dtype.kind not in "iu":
            raise ValueError("rebuilt prompt does not have the cached prompt length")
        expected = expected.astype(np.int64, copy=False)
        if not np.array_equal(expected, prefix):
            mismatch = np.flatnonzero(expected != prefix)
            where = int(mismatch[0]) if mismatch.size else -1
            raise ValueError(f"rebuilt prompt token prefix differs at position {where}")
    digest = prompt_token_sha256(prefix)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("cached prompt prefix does not match prompt_token_sha256")
    return digest


def _between(prompt: str, start_marker: str, end_markers: tuple[str, ...]):
    marker = prompt.find(start_marker)
    if marker < 0:
        raise ValueError(f"prompt is missing task anchor {start_marker!r}")
    start = marker + len(start_marker)
    candidates = [prompt.find(end, start) for end in end_markers]
    candidates = [position for position in candidates if position >= 0]
    if not candidates:
        raise ValueError("prompt is missing the closing task anchor")
    end = min(candidates)
    if end <= start:
        raise ValueError("task evidence anchor is empty")
    return start, end


def task_character_roles(source_record: Mapping) -> np.ndarray:
    """Partition the raw RAGTruth user prompt with task-specific anchors.

    Non-evidence/question characters in the user instruction are constraints.
    Chat template and system text are assigned later as ``other_prompt``.
    """

    prompt = str(source_record["prompt"])
    task = str(source_record["task_type"]).casefold()
    source_info = source_record["source_info"]
    roles = np.full(len(prompt), CONSTRAINT, dtype=np.int8)

    if task == "qa":
        if not isinstance(source_info, Mapping):
            raise ValueError("QA source_info must be an object")
        question = str(source_info.get("question", ""))
        passages = str(source_info.get("passages", ""))
        if not question or not passages:
            raise ValueError("QA source_info must contain question and passages")
        question_start = prompt.find(question)
        if question_start < 0:
            raise ValueError("QA question is not an exact prompt substring")
        question_end = question_start + len(question)
        evidence_start, evidence_end = _between(
            prompt,
            "passages:\n",
            ("\nIn case the passages", "\noutput:"),
        )
        if evidence_start < question_end:
            raise ValueError("QA evidence precedes the question unexpectedly")
        roles[question_start:question_end] = QUESTION
        roles[evidence_start:evidence_end] = EVIDENCE
    elif task == "summary":
        if not isinstance(source_info, str) or not source_info:
            raise ValueError("Summary source_info must be non-empty text")
        evidence_start = prompt.find(source_info)
        if evidence_start < 0:
            raise ValueError("Summary source text is not an exact prompt substring")
        evidence_end = evidence_start + len(source_info)
        roles[evidence_start:evidence_end] = EVIDENCE
    elif task == "data2txt":
        if not isinstance(source_info, Mapping):
            raise ValueError("Data2txt source_info must be an object")
        evidence_start, evidence_end = _between(
            prompt,
            "Structured data:\n",
            ("\nOverview:",),
        )
        roles[evidence_start:evidence_end] = EVIDENCE
    else:
        raise ValueError(f"unsupported RAGTruth task type: {source_record['task_type']}")
    if not bool((roles == EVIDENCE).any()):
        raise ValueError("task anchors did not identify evidence")
    return roles


def _unique_substring(text: str, substring: str) -> tuple[int, int]:
    start = text.find(substring)
    if start < 0:
        raise ValueError("rendered chat does not contain the exact source prompt")
    if text.find(substring, start + 1) >= 0:
        raise ValueError("source prompt is ambiguous inside rendered chat")
    return start, start + len(substring)


def build_prompt_role_map(
    source_record: Mapping,
    tokenizer,
    cached_token_ids,
    response_idx: int,
) -> PromptRoleMap:
    """Rebuild, validate, and role-partition one cached RAGTruth prompt."""

    required = {"source_id", "task_type", "source_info", "prompt"}
    if required.difference(source_record):
        raise ValueError("source_info row is missing required fields")
    source_prompt = str(source_record["prompt"])
    rendered, rebuilt_ids, offsets = tokenize_historical_prompt(tokenizer, source_prompt)
    response_idx = int(response_idx)
    digest = validate_cached_prompt(
        rebuilt_ids,
        cached_token_ids,
        response_idx,
    )
    user_start, user_end = _unique_substring(rendered, source_prompt)
    user_roles = task_character_roles(source_record)
    token_roles = np.full(response_idx, OTHER_PROMPT, dtype=np.int8)
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        overlap_start = max(start, user_start)
        overlap_end = min(end, user_end)
        if overlap_end <= overlap_start:
            continue
        relative_start = overlap_start - user_start
        relative_end = overlap_end - user_start
        candidates = user_roles[relative_start:relative_end]
        # Match the corpus label alignment rule: any overlap assigns a token
        # to the anchored span.  Evidence wins at a boundary so an intervention
        # cannot leave a mixed evidence token available as a key.
        token_roles[index] = next(
            role
            for role in (EVIDENCE, QUESTION, CONSTRAINT)
            if bool((candidates == role).any())
        )

    role_map = PromptRoleMap(
        source_id=str(source_record["source_id"]),
        task_type=str(source_record["task_type"]),
        response_idx=response_idx,
        role_ids=token_roles,
        prompt_token_sha256=digest,
        source_info_sha256=_canonical_sha256(source_record),
        prompt_token_ids=rebuilt_ids.copy(),
    ).validate(cached_token_ids)
    if not bool(role_map.role_mask("evidence").any()):
        raise ValueError("tokenization removed every evidence token")
    return role_map


def read_source_info(source_info_path) -> dict[str, dict]:
    """Read only RAGTruth ``source_info.jsonl`` and reject duplicate IDs."""

    path = Path(source_info_path)
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row["source_id"])
            if source_id in rows:
                raise ValueError(f"duplicate source_info row: {source_id}")
            rows[source_id] = row
    if not rows:
        raise ValueError("source_info.jsonl is empty")
    return rows


load_source_info = read_source_info


@dataclass(frozen=True)
class CachedPrompt:
    """Minimal label-free cache view needed to construct a role artifact."""

    source_id: str
    token_ids: np.ndarray
    response_idx: int


def build_role_index(
    source_info_path,
    tokenizer,
    cached_prompts: Iterable[CachedPrompt],
) -> dict[str, PromptRoleMap]:
    """Build one validated role map per source from label-free inputs only."""

    sources = read_source_info(source_info_path)
    result: dict[str, PromptRoleMap] = {}
    for cached in cached_prompts:
        source_id = str(cached.source_id)
        if source_id not in sources:
            raise ValueError(f"cached source is absent from source_info.jsonl: {source_id}")
        current = build_prompt_role_map(
            sources[source_id],
            tokenizer,
            cached.token_ids,
            cached.response_idx,
        )
        previous = result.get(source_id)
        if previous is not None:
            if (
                previous.response_idx != current.response_idx
                or previous.prompt_token_sha256 != current.prompt_token_sha256
                or not np.array_equal(previous.role_ids, current.role_ids)
            ):
                raise ValueError("samples sharing a source have inconsistent prompts")
        else:
            result[source_id] = current
    return result


def write_role_jsonl(role_maps: Mapping[str, PromptRoleMap], output_path) -> Path:
    """Write deterministic source-level prompt roles without token labels."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for source_id in sorted(role_maps):
        role_map = role_maps[source_id].validate()
        if source_id != role_map.source_id:
            raise ValueError("role index key and source_id differ")
        lines.append(
            json.dumps(
                role_map.to_json(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write("".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_role_jsonl(path) -> dict[str, PromptRoleMap]:
    """Load and structurally validate a deterministic prompt-role artifact."""

    result: dict[str, PromptRoleMap] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            role_map = PromptRoleMap.from_json(json.loads(line))
            if role_map.source_id in result:
                raise ValueError(f"duplicate prompt-role row: {role_map.source_id}")
            result[role_map.source_id] = role_map
    if not result:
        raise ValueError("prompt-role artifact is empty")
    return result
