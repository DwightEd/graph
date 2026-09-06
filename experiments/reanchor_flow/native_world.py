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

NATIVE_WORLD_SCHEMA = 2


@dataclass(frozen=True)
class TargetReanchorSelection:
    """Structured provenance for one re-anchor-selected target row."""

    query_position: int
    policy: str
    has_event: bool
    fallback: bool
    center_position: int
    window_offset: int
    layer: int
    head: int
    source_kind: str
    source_position: int
    source_unit_id: int
    score: float
    support: int
    previous_anchor_fraction: float
    anchor_fraction: float
    relative_anchor_rise: float
    relative_local_fall: float
    scan_signal: str = "exact_full_row_message_transport"

    @property
    def is_center(self) -> bool:
        return self.has_event and self.window_offset == 0

    def check(self, target: TargetContrast) -> TargetReanchorSelection:
        if self.query_position != target.query_position:
            raise ValueError("target selection metadata names another query row")
        if self.policy not in {"reanchor", "reanchor-window"}:
            raise ValueError("target selection metadata has an invalid policy")
        if self.scan_signal != "exact_full_row_message_transport":
            raise ValueError("target selection metadata has an invalid scan signal")
        if self.has_event:
            if self.fallback:
                raise ValueError("a re-anchor event cannot also be a fallback")
            if (
                self.center_position < 0
                or self.query_position - self.center_position != self.window_offset
                or self.window_offset not in {-1, 0, 1}
                or (self.policy == "reanchor" and self.window_offset != 0)
                or self.layer < 0
                or self.head < 0
                or self.source_kind
                not in {"prompt_evidence", "other_prompt", "remote_response"}
                or self.source_position < 0
                or self.source_unit_id < 0
                or not np.isfinite(self.score)
                or self.score <= 0
                or self.support < 1
            ):
                raise ValueError("re-anchor event metadata is invalid")
        elif (
            self.center_position != -1
            or self.window_offset != 0
            or self.layer != -1
            or self.head != -1
            or self.source_kind != "none"
            or self.source_position != -1
            or self.source_unit_id != -1
            or self.score != 0
            or self.support != 0
        ):
            raise ValueError("non-event target metadata must use sentinel values")
        fractions = (
            self.previous_anchor_fraction,
            self.anchor_fraction,
            self.relative_anchor_rise,
            self.relative_local_fall,
        )
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("target selection fractions must lie in [0,1]")
        return self


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
    target_selection: tuple[TargetReanchorSelection, ...] = ()

    def check(self) -> NativeWorld:
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
        if self.target_selection:
            if len(self.target_selection) != len(self.targets):
                raise ValueError("target selection metadata must align with targets")
            for target, selection in zip(
                self.targets, self.target_selection, strict=True
            ):
                selection.check(target)
                if selection.has_event:
                    # A ``-1`` window target ends with the event-center token;
                    # its metadata legitimately refers to that final token even
                    # though the center is not a predictor row in this prefix.
                    if not self.response_start <= selection.center_position < len(ids):
                        raise ValueError(
                            "re-anchor center is outside response predictors"
                        )
                    if not 0 <= selection.source_position <= selection.center_position:
                        raise ValueError("re-anchor source is outside its causal row")
                    if selection.source_unit_id >= self.units.count:
                        raise ValueError(
                            "re-anchor source unit is outside the unit table"
                        )
        return self

    def prefix(self, target: TargetContrast) -> NativeWorld:
        if target not in self.targets:
            raise ValueError("native audit target was not frozen in this world")
        stop = target.query_position + 2
        target_index = self.targets.index(target)
        selection = (
            () if not self.target_selection else (self.target_selection[target_index],)
        )
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
            selection,
        ).check()


def save_native_world(path: str | Path, world: NativeWorld) -> None:
    world.check()
    selection = world.target_selection
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
            "target_reanchor_query_position": np.asarray(
                [item.query_position for item in selection], dtype=np.int32
            ),
            "target_reanchor_policy": np.asarray(
                [item.policy for item in selection], dtype=np.str_
            ),
            "target_reanchor_has_event": np.asarray(
                [item.has_event for item in selection], dtype=np.bool_
            ),
            "target_reanchor_fallback": np.asarray(
                [item.fallback for item in selection], dtype=np.bool_
            ),
            "target_reanchor_center_position": np.asarray(
                [item.center_position for item in selection], dtype=np.int32
            ),
            "target_reanchor_window_offset": np.asarray(
                [item.window_offset for item in selection], dtype=np.int8
            ),
            "target_reanchor_layer": np.asarray(
                [item.layer for item in selection], dtype=np.int16
            ),
            "target_reanchor_head": np.asarray(
                [item.head for item in selection], dtype=np.int16
            ),
            "target_reanchor_source_kind": np.asarray(
                [item.source_kind for item in selection], dtype=np.str_
            ),
            "target_reanchor_source_position": np.asarray(
                [item.source_position for item in selection], dtype=np.int32
            ),
            "target_reanchor_source_unit_id": np.asarray(
                [item.source_unit_id for item in selection], dtype=np.int32
            ),
            "target_reanchor_score": np.asarray(
                [item.score for item in selection], dtype=np.float64
            ),
            "target_reanchor_support": np.asarray(
                [item.support for item in selection], dtype=np.int32
            ),
            "target_reanchor_previous_anchor_fraction": np.asarray(
                [item.previous_anchor_fraction for item in selection],
                dtype=np.float64,
            ),
            "target_reanchor_anchor_fraction": np.asarray(
                [item.anchor_fraction for item in selection], dtype=np.float64
            ),
            "target_reanchor_relative_anchor_rise": np.asarray(
                [item.relative_anchor_rise for item in selection], dtype=np.float64
            ),
            "target_reanchor_relative_local_fall": np.asarray(
                [item.relative_local_fall for item in selection], dtype=np.float64
            ),
            "target_reanchor_scan_signal": np.asarray(
                [item.scan_signal for item in selection], dtype=np.str_
            ),
        },
    )


def load_native_world(path: str | Path) -> NativeWorld:
    with np.load(Path(path), allow_pickle=False) as stored:
        schema = int(stored["native_world_schema"])
        if schema not in {1, NATIVE_WORLD_SCHEMA}:
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
        selection = ()
        if schema >= 2:
            selection = tuple(
                TargetReanchorSelection(
                    int(query),
                    str(policy),
                    bool(has_event),
                    bool(fallback),
                    int(center),
                    int(offset),
                    int(layer),
                    int(head),
                    str(source_kind),
                    int(source_position),
                    int(source_unit),
                    float(score),
                    int(support),
                    float(previous_fraction),
                    float(anchor_fraction),
                    float(anchor_rise),
                    float(local_fall),
                    str(scan_signal),
                )
                for (
                    query,
                    policy,
                    has_event,
                    fallback,
                    center,
                    offset,
                    layer,
                    head,
                    source_kind,
                    source_position,
                    source_unit,
                    score,
                    support,
                    previous_fraction,
                    anchor_fraction,
                    anchor_rise,
                    local_fall,
                    scan_signal,
                ) in zip(
                    stored["target_reanchor_query_position"],
                    stored["target_reanchor_policy"].astype(str),
                    stored["target_reanchor_has_event"],
                    stored["target_reanchor_fallback"],
                    stored["target_reanchor_center_position"],
                    stored["target_reanchor_window_offset"],
                    stored["target_reanchor_layer"],
                    stored["target_reanchor_head"],
                    stored["target_reanchor_source_kind"].astype(str),
                    stored["target_reanchor_source_position"],
                    stored["target_reanchor_source_unit_id"],
                    stored["target_reanchor_score"],
                    stored["target_reanchor_support"],
                    stored["target_reanchor_previous_anchor_fraction"],
                    stored["target_reanchor_anchor_fraction"],
                    stored["target_reanchor_relative_anchor_rise"],
                    stored["target_reanchor_relative_local_fall"],
                    stored["target_reanchor_scan_signal"].astype(str),
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
            selection,
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
