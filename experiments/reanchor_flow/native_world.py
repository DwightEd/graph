"""Native teacher-forced world and persistent Value-source cut state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from experiments.common.llama_message_intervention import (
    ForwardCache,
    MessageGate,
    forward_layers,
    gate_to,
)

from .artifacts import save_result
from .worlds import SourceUnits, TargetContrast

NATIVE_WORLD_SCHEMA = 1


@dataclass(frozen=True)
class NativeWorld:
    """One native teacher-forced sample with label-free target contrasts."""

    sample_id: str
    tokenizer_id: str
    token_ids: Tensor
    response_start: int
    units: SourceUnits
    evidence_unit_id: tuple[int, ...]
    targets: tuple[TargetContrast, ...]

    def check(self) -> "NativeWorld":
        ids = self.token_ids
        if (
            not self.sample_id
            or Path(self.sample_id).name != self.sample_id
            or "\\" in self.sample_id
            or self.sample_id in {".", ".."}
        ):
            raise ValueError("sample_id must be one safe filename component")
        if not self.tokenizer_id:
            raise ValueError("tokenizer identity is required")
        if ids.ndim != 1 or ids.dtype != torch.long or ids.device.type != "cpu":
            raise ValueError("native token IDs must be one CPU int64 vector")
        if not 0 < self.response_start < len(ids):
            raise ValueError("response_start does not define a non-empty response")
        self.units.check(len(ids) - 1)
        if not self.evidence_unit_id:
            raise ValueError("native world has no represented evidence units")
        if len(set(self.evidence_unit_id)) != len(self.evidence_unit_id):
            raise ValueError("evidence unit IDs must not contain duplicates")
        for unit_id in self.evidence_unit_id:
            if not 0 <= unit_id < self.units.count:
                raise ValueError("evidence unit ID is outside the unit table")
            if self.units.kind[unit_id] in {"other_prompt", "response"}:
                raise ValueError("evidence unit table includes a non-evidence unit")
            if not bool((self.units.token_unit_id == unit_id).any()):
                raise ValueError("evidence unit has no represented source token")
        if not self.targets:
            raise ValueError("native world has no target contrast")
        target_keys = {
            (
                target.query_position,
                target.positive_token_id,
                target.negative_token_id,
            )
            for target in self.targets
        }
        if len(target_keys) != len(self.targets):
            raise ValueError("native target contrasts must not contain duplicates")
        for target in self.targets:
            if not self.response_start - 1 <= target.query_position < len(ids) - 1:
                raise ValueError("target query is outside the response predictor rows")
            if int(ids[target.query_position + 1]) != target.positive_token_id:
                raise ValueError("positive candidate is not the observed target token")
            if target.positive_token_id == target.negative_token_id:
                raise ValueError("target candidates must differ")
            if not target.origin:
                raise ValueError("target contrast origin is required")
        return self

    def prefix(self, target: TargetContrast) -> "NativeWorld":
        if target not in self.targets:
            raise ValueError("native audit target was not frozen in this world")
        stop = target.query_position + 2
        return NativeWorld(
            self.sample_id,
            self.tokenizer_id,
            self.token_ids[:stop],
            self.response_start,
            SourceUnits(
                self.units.token_unit_id[: stop - 1],
                self.units.name,
                self.units.kind,
            ),
            self.evidence_unit_id,
            (target,),
        ).check()


def save_native_world(path: str | Path, world: NativeWorld) -> None:
    world.check()
    save_result(
        path,
        {
            "native_world_schema": NATIVE_WORLD_SCHEMA,
            "sample_id": world.sample_id,
            "tokenizer_id": world.tokenizer_id,
            "token_ids": world.token_ids,
            "response_start": world.response_start,
            "token_unit_id": world.units.token_unit_id,
            "unit_name": np.asarray(world.units.name),
            "unit_kind": np.asarray(world.units.kind),
            "evidence_unit_id": np.asarray(world.evidence_unit_id, dtype=np.int32),
            "query_position": np.asarray(
                [target.query_position for target in world.targets],
                dtype=np.int32,
            ),
            "positive_token_id": np.asarray(
                [target.positive_token_id for target in world.targets],
                dtype=np.int32,
            ),
            "negative_token_id": np.asarray(
                [target.negative_token_id for target in world.targets],
                dtype=np.int32,
            ),
            "contrast_origin": np.asarray([target.origin for target in world.targets]),
        },
    )


def load_native_world(path: str | Path) -> NativeWorld:
    with np.load(Path(path), allow_pickle=False) as stored:
        if int(stored["native_world_schema"]) != NATIVE_WORLD_SCHEMA:
            raise ValueError("unsupported native-world schema")
        targets = tuple(
            TargetContrast(int(query), int(positive), int(negative), str(origin))
            for query, positive, negative, origin in zip(
                stored["query_position"],
                stored["positive_token_id"],
                stored["negative_token_id"],
                stored["contrast_origin"].astype(str),
                strict=True,
            )
        )
        world = NativeWorld(
            str(stored["sample_id"].item()),
            str(stored["tokenizer_id"].item()),
            torch.from_numpy(stored["token_ids"].astype(np.int64)),
            int(stored["response_start"]),
            SourceUnits(
                torch.from_numpy(stored["token_unit_id"].astype(np.int64)),
                tuple(stored["unit_name"].astype(str).tolist()),
                tuple(stored["unit_kind"].astype(str).tolist()),
            ),
            tuple(stored["evidence_unit_id"].astype(np.int64).tolist()),
            targets,
        )
    return world.check()


def source_gate(
    world: NativeWorld,
    unit_ids: tuple[int, ...],
) -> MessageGate:
    """Block Value messages emitted by named unit positions in every layer."""

    source = torch.zeros(len(world.token_ids) - 1, dtype=torch.bool)
    for unit_id in unit_ids:
        source |= world.units.token_unit_id == unit_id
    return MessageGate(split_layer=0, source_mask=source)


def gated_forward_cache(
    model,
    baseline: ForwardCache,
    gate: MessageGate,
) -> ForwardCache:
    """Capture every computation stage under a gate with frozen readout."""

    device = model.get_input_embeddings().weight.device
    layer_input: dict[int, Tensor] = {}
    attention_write: dict[int, Tensor] = {}
    mlp_write: dict[int, Tensor] = {}
    layers = set(range(baseline.layer_count))
    with torch.inference_mode():
        final = forward_layers(
            model,
            baseline.layer_input[0].to(device)[None],
            0,
            gate=gate_to(gate, device),
            save_inputs=layer_input,
            save_layers=layers,
            save_attention=attention_write,
            save_mlp=mlp_write,
            attention_query_chunk=baseline.attention_query_chunk,
        )[0]
        response = final.index_select(0, baseline.query.to(device))
        fixed_margin = torch.einsum(
            "td,td->t",
            response.float(),
            baseline.readout_direction.to(device),
        )
        fixed_margin += baseline.readout_bias.to(device)
    empty = torch.full_like(baseline.baseline_target_logprob, float("nan"))
    return ForwardCache(
        layer_input,
        final.detach().cpu(),
        baseline.layer_count,
        baseline.query,
        baseline.target,
        baseline.runner,
        baseline.readout_direction,
        baseline.readout_bias,
        fixed_margin.cpu(),
        empty,
        empty.clone(),
        baseline.attention_query_chunk,
        attention_write,
        mlp_write,
    )
