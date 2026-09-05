"""Matched evidence worlds and fixed target contrasts for ETCC audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .artifacts import save_result

PAIR_SCHEMA = 1


@dataclass(frozen=True)
class SourceUnits:
    """Semantic source units attached to every causal source-token position."""

    token_unit_id: Tensor
    name: tuple[str, ...]
    kind: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.name)

    def positions(self, unit_ids: tuple[int, ...]) -> Tensor:
        selected = torch.zeros_like(self.token_unit_id, dtype=torch.bool)
        for unit_id in unit_ids:
            selected |= self.token_unit_id == unit_id
        return torch.nonzero(selected, as_tuple=False).flatten()

    def check(self, source_count: int) -> "SourceUnits":
        ids = self.token_unit_id
        if ids.shape != (source_count,) or ids.dtype != torch.long:
            raise ValueError(
                "token_unit_id must contain one int64 unit per source token"
            )
        if ids.device.type != "cpu":
            raise ValueError("source-unit coordinates must be stored on CPU")
        if len(self.name) != len(self.kind) or not self.name:
            raise ValueError(
                "source-unit names and kinds must form one non-empty table"
            )
        if int(ids.min()) < 0 or int(ids.max()) >= self.count:
            raise ValueError("token_unit_id points outside the source-unit table")
        used = torch.unique(ids)
        if not torch.equal(used, torch.arange(int(used.max()) + 1)):
            raise ValueError("represented source-unit IDs must be contiguous")
        return self


@dataclass(frozen=True)
class TargetContrast:
    """A fixed positive-versus-negative vocabulary decision at predictor ``q``."""

    query_position: int
    positive_token_id: int
    negative_token_id: int
    origin: str


@dataclass(frozen=True)
class PairedWorld:
    """Clean and candidate-corrupted prompts with one shared forced response."""

    sample_id: str
    tokenizer_id: str
    corruption: str
    clean_token_ids: Tensor
    corrupt_token_ids: Tensor
    response_start: int
    units: SourceUnits
    candidate_unit_id: tuple[int, ...]
    targets: tuple[TargetContrast, ...]

    def check(self) -> "PairedWorld":
        clean = self.clean_token_ids
        corrupt = self.corrupt_token_ids
        if not self.sample_id or not self.tokenizer_id or not self.corruption:
            raise ValueError(
                "pair identity, tokenizer, and corruption must be recorded"
            )
        if (
            Path(self.sample_id).name != self.sample_id
            or "\\" in self.sample_id
            or self.sample_id in {".", ".."}
        ):
            raise ValueError("sample_id must be a single safe filename component")
        if (
            clean.ndim != 1
            or clean.dtype != torch.long
            or corrupt.dtype != torch.long
            or corrupt.shape != clean.shape
        ):
            raise ValueError(
                "clean and corrupt token IDs must be aligned int64 vectors"
            )
        if clean.device.type != "cpu" or corrupt.device.type != "cpu":
            raise ValueError("paired-world token IDs must be stored on CPU")
        if not 0 < self.response_start < len(clean):
            raise ValueError("response_start does not define a non-empty response")
        if not torch.equal(
            clean[self.response_start :], corrupt[self.response_start :]
        ):
            raise ValueError(
                "paired worlds must share the exact teacher-forced response"
            )
        self.units.check(len(clean) - 1)
        if not self.candidate_unit_id:
            raise ValueError("a paired world must name corrupted candidate units")
        if len(set(self.candidate_unit_id)) != len(self.candidate_unit_id):
            raise ValueError("candidate_unit_id must not contain duplicates")
        if (
            min(self.candidate_unit_id) < 0
            or max(self.candidate_unit_id) >= self.units.count
        ):
            raise ValueError("candidate_unit_id points outside the source-unit table")
        changed = clean[:-1] != corrupt[:-1]
        allowed = torch.zeros_like(changed)
        for unit_id in self.candidate_unit_id:
            allowed |= self.units.token_unit_id == unit_id
        if bool((changed & ~allowed).any()):
            raise ValueError(
                "clean/corrupt differences extend outside candidate source units"
            )
        if not bool(changed.any()):
            raise ValueError("clean and corrupt worlds are identical")
        for unit_id in self.candidate_unit_id:
            positions = self.units.token_unit_id == unit_id
            if not bool((changed & positions).any()):
                raise ValueError("every candidate unit must change in the paired world")
        for target in self.targets:
            if not self.response_start - 1 <= target.query_position < len(clean) - 1:
                raise ValueError("target query is outside the response prediction rows")
            if target.positive_token_id == target.negative_token_id:
                raise ValueError("target candidates must differ")
            if not target.origin:
                raise ValueError("every target contrast must record its origin")
        if not self.targets:
            raise ValueError("a paired world must contain a target contrast")
        return self

    def prefix(self, target: TargetContrast) -> "PairedWorld":
        """Return the causal prefix ending immediately after the target token."""

        stop = target.query_position + 2
        return PairedWorld(
            sample_id=self.sample_id,
            tokenizer_id=self.tokenizer_id,
            corruption=self.corruption,
            clean_token_ids=self.clean_token_ids[:stop],
            corrupt_token_ids=self.corrupt_token_ids[:stop],
            response_start=self.response_start,
            units=SourceUnits(
                self.units.token_unit_id[: stop - 1],
                self.units.name,
                self.units.kind,
            ),
            candidate_unit_id=self.candidate_unit_id,
            targets=(target,),
        ).check()

    def isolate(self, unit_id: int) -> "PairedWorld":
        """Keep only one candidate corruption for root-conditioned capture."""

        if unit_id not in self.candidate_unit_id:
            raise ValueError("isolated unit is not a declared root candidate")
        corrupt = self.clean_token_ids.clone()
        positions = self.units.token_unit_id == unit_id
        corrupt_source = corrupt[:-1]
        corrupt_source[positions] = self.corrupt_token_ids[:-1][positions]
        return PairedWorld(
            sample_id=self.sample_id,
            tokenizer_id=self.tokenizer_id,
            corruption=f"{self.corruption}; isolated unit {unit_id}",
            clean_token_ids=self.clean_token_ids,
            corrupt_token_ids=corrupt,
            response_start=self.response_start,
            units=self.units,
            candidate_unit_id=(unit_id,),
            targets=self.targets,
        ).check()


def save_world(path: str | Path, world: PairedWorld) -> None:
    """Persist the small, model-independent input contract for one audit."""

    world.check()
    save_result(
        path,
        {
            "pair_schema": PAIR_SCHEMA,
            "sample_id": world.sample_id,
            "tokenizer_id": world.tokenizer_id,
            "corruption": world.corruption,
            "clean_token_ids": world.clean_token_ids,
            "corrupt_token_ids": world.corrupt_token_ids,
            "response_start": world.response_start,
            "token_unit_id": world.units.token_unit_id,
            "unit_name": np.asarray(world.units.name),
            "unit_kind": np.asarray(world.units.kind),
            "candidate_unit_id": np.asarray(
                world.candidate_unit_id, dtype=np.int32
            ),
            "query_position": np.asarray(
                [target.query_position for target in world.targets], dtype=np.int32
            ),
            "positive_token_id": np.asarray(
                [target.positive_token_id for target in world.targets], dtype=np.int32
            ),
            "negative_token_id": np.asarray(
                [target.negative_token_id for target in world.targets], dtype=np.int32
            ),
            "contrast_origin": np.asarray(
                [target.origin for target in world.targets]
            ),
        },
    )


def load_world(path: str | Path) -> PairedWorld:
    """Load one controlled pair without opening any hallucination label."""

    with np.load(Path(path), allow_pickle=False) as stored:
        if int(stored["pair_schema"]) != PAIR_SCHEMA:
            raise ValueError("unsupported paired-world schema")
        query = stored["query_position"].astype(np.int64)
        positive = stored["positive_token_id"].astype(np.int64)
        negative = stored["negative_token_id"].astype(np.int64)
        origin = stored["contrast_origin"].astype(str)
        targets = tuple(
            TargetContrast(int(q), int(a), int(b), str(source))
            for q, a, b, source in zip(
                query, positive, negative, origin, strict=True
            )
        )
        world = PairedWorld(
            sample_id=str(stored["sample_id"].item()),
            tokenizer_id=str(stored["tokenizer_id"].item()),
            corruption=str(stored["corruption"].item()),
            clean_token_ids=torch.from_numpy(
                stored["clean_token_ids"].astype(np.int64)
            ),
            corrupt_token_ids=torch.from_numpy(
                stored["corrupt_token_ids"].astype(np.int64)
            ),
            response_start=int(stored["response_start"]),
            units=SourceUnits(
                torch.from_numpy(stored["token_unit_id"].astype(np.int64)),
                tuple(stored["unit_name"].astype(str).tolist()),
                tuple(stored["unit_kind"].astype(str).tolist()),
            ),
            candidate_unit_id=tuple(
                stored["candidate_unit_id"].astype(np.int64).tolist()
            ),
            targets=targets,
        )
    return world.check()
