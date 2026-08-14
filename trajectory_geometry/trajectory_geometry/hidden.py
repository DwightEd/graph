"""Validated, label-free hidden-state sidecars and attention pairing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import SparseAttentionSample, _load_payload, _numpy


HIDDEN_KEYS = ("hidden_states", "all_hidden_states", "hidden_state", "states")


def _sample_id(path: Path) -> str:
    name = path.stem
    prefixes = ("hidden_states_", "hidden_state_", "hidden_", "states_", "attention_")
    suffixes = ("_hidden_states", "_hidden_state", "_hidden", "_states")
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        raise ValueError(f"cannot derive sample id from {path.name}")
    return name


def _hidden_array(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        layers = []
        for layer in value:
            array = _numpy(layer)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 2:
                raise ValueError("each hidden-state layer must have shape [tokens, dim]")
            layers.append(array)
        if not layers:
            raise ValueError("hidden-state sequence is empty")
        return np.stack(layers)
    array = _numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("hidden states must have shape [states,tokens,dim]")
    return array


@dataclass(frozen=True)
class HiddenStateSample:
    path: Path
    sample_id: str
    states: np.ndarray
    token_ids: np.ndarray | None

    @property
    def state_count(self) -> int:
        return int(self.states.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.states.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.states.shape[2])

    def align(self, attention: SparseAttentionSample) -> tuple[np.ndarray, int]:
        """Return `[state, token, dim]` and the aligned attention-layer offset."""
        if self.sample_id != attention.sample_id:
            raise ValueError(
                f"hidden/attention sample ids differ: {self.sample_id} != {attention.sample_id}"
            )
        states = self.states
        if states.shape[1] != attention.token_count and states.shape[0] == attention.token_count:
            states = states.transpose(1, 0, 2)
        if states.shape[1] != attention.token_count:
            raise ValueError(
                "hidden states must contain the complete prompt+response token sequence; "
                f"got {states.shape[1]} states for {attention.token_count} attention tokens"
            )
        if states.shape[0] == attention.layers + 1:
            attention_layer_offset = 0
        elif states.shape[0] == attention.layers:
            # A cache with only transformer block outputs has no embedding input
            # state.  Transition i -> i+1 is therefore explained by attention i+1.
            attention_layer_offset = 1
        else:
            raise ValueError(
                "hidden sidecar must contain all consecutive layer states: expected "
                f"{attention.layers + 1} (embedding + blocks) or {attention.layers} "
                f"(block outputs), got {states.shape[0]}"
            )
        if self.token_ids is not None:
            token_ids = np.asarray(self.token_ids).reshape(-1)
            if token_ids.shape != attention.token_ids.shape or not np.array_equal(
                token_ids.astype(np.int64, copy=False), attention.token_ids
            ):
                raise ValueError("hidden and attention token_ids are not aligned")
        if not np.issubdtype(states.dtype, np.floating):
            raise ValueError("hidden states must be floating point")
        if not np.all(np.isfinite(states)):
            raise ValueError("hidden states contain non-finite values")
        return states.astype(np.float32, copy=False), attention_layer_offset


def load_hidden_sample(path: str | Path) -> HiddenStateSample:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix == ".npy":
        payload: Any = np.load(resolved, mmap_mode="r", allow_pickle=False)
    else:
        payload = _load_payload(resolved)
    token_ids = None
    sample_id = _sample_id(resolved)
    value = payload
    if isinstance(payload, dict):
        key = next((name for name in HIDDEN_KEYS if name in payload), None)
        if key is None:
            raise ValueError(
                f"hidden sidecar {resolved.name} has none of {list(HIDDEN_KEYS)}"
            )
        value = payload[key]
        if "token_ids" in payload:
            token_ids = _numpy(payload["token_ids"]).astype(np.int64, copy=False)
        for key_name in ("response_id", "sample_id"):
            if key_name in payload:
                raw = payload[key_name]
                if hasattr(raw, "item"):
                    raw = raw.item()
                sample_id = str(raw)
                break
    return HiddenStateSample(
        path=resolved,
        sample_id=sample_id,
        states=_hidden_array(value),
        token_ids=token_ids,
    )


def discover_hidden_files(root: str | Path, split: str | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if str(root).startswith("/path/to/"):
        raise FileNotFoundError(f"{root} is a documentation placeholder")
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"hidden-state root does not exist: {root}")
    selected = root / split if split and (root / split).is_dir() else root
    files = []
    for suffix in ("*.pt", "*.npz", "*.npy"):
        files.extend(selected.rglob(suffix))
    files = sorted(path for path in files if "manifest" not in path.name)
    if not files:
        raise FileNotFoundError(f"no .pt, .npz, or .npy hidden sidecars under {selected}")
    return files


@dataclass(frozen=True)
class AttentionHiddenPair:
    sample_id: str
    attention_path: Path
    hidden_path: Path


def pair_attention_hidden(
    attention_files: list[Path], hidden_files: list[Path]
) -> list[AttentionHiddenPair]:
    hidden_by_id: dict[str, Path] = {}
    for path in hidden_files:
        identifier = _sample_id(path)
        if identifier in hidden_by_id:
            raise ValueError(f"duplicate hidden sidecars for sample {identifier}")
        hidden_by_id[identifier] = path
    pairs = []
    missing = []
    for attention_path in attention_files:
        identifier = _sample_id(attention_path)
        hidden_path = hidden_by_id.get(identifier)
        if hidden_path is None:
            missing.append(identifier)
        else:
            pairs.append(AttentionHiddenPair(identifier, attention_path, hidden_path))
    if missing:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(
            f"missing hidden sidecars for {len(missing)} attention samples: {preview}"
        )
    if not pairs:
        raise ValueError("attention and hidden inventories have no matching sample ids")
    return pairs
