"""Exact frozen replay for the SELECT--RELAY--OVERRIDE audit.

The audit compares candidate A and candidate B at their first point of
divergence.  For a fixed teacher-forced prefix, the log-probability margin is
exactly the raw logit difference ``z(B) - z(A)``; no vocabulary-wide softmax,
generation, training, or backward pass is needed.  The audit layer determines
which candidate is the model prior from the question-only margin.

Two interventions are implemented and nothing else:

* isolate selected prompt sources by removing them as keys for every later
  query, or remove earlier response keys only from the predictor row; and
* replace the strictly earlier response-prefix K/V projections in a
  counter-evidence replay by those captured at the same positions in the
  prior-context replay.

The captured keys are pre-RoPE ``k_proj`` outputs.  Prior and counter branches
have identical length and token positions, so applying the unchanged rotary
embedding after replacement yields the exact prior-branch keys at those
positions.  Values are the exact ``v_proj`` outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def predictor_allowed_attention(
    sequence_length: int,
    predictor_index: int,
    blocked_positions: Sequence[int] = (),
) -> np.ndarray:
    """Return causal support with keys removed only from one predictor row."""

    sequence_length = int(sequence_length)
    predictor_index = int(predictor_index)
    if sequence_length < 1 or not 0 <= predictor_index < sequence_length:
        raise ValueError("predictor_index must lie inside the input sequence")
    blocked = np.asarray(tuple(blocked_positions), dtype=np.int64)
    if blocked.size and (
        int(blocked.min()) < 0 or int(blocked.max()) >= predictor_index
    ):
        raise ValueError("blocked keys must strictly precede the predictor")

    allowed = np.tri(sequence_length, sequence_length, dtype=np.bool_)
    if blocked.size:
        allowed[predictor_index, np.unique(blocked)] = False
    return allowed


def prompt_source_allowed_attention(
    sequence_length: int,
    predictor_index: int,
    source_positions: Sequence[int],
) -> np.ndarray:
    """Block prompt sources as keys for every strictly later query.

    This removes both the direct source-to-predictor route and indirect routes
    in which a later prompt/history token first absorbs the source message.
    Each source's diagonal remains available.
    """

    sequence_length = int(sequence_length)
    predictor_index = int(predictor_index)
    if sequence_length < 1 or not 0 <= predictor_index < sequence_length:
        raise ValueError("predictor_index must lie inside the input sequence")
    sources = np.asarray(tuple(source_positions), dtype=np.int64)
    if sources.size == 0:
        raise ValueError("source_positions must be non-empty")
    if int(sources.min()) < 0 or int(sources.max()) >= predictor_index:
        raise ValueError("prompt sources must strictly precede the predictor")

    allowed = np.tri(sequence_length, sequence_length, dtype=np.bool_)
    for source in np.unique(sources):
        allowed[int(source) + 1 :, int(source)] = False
    return allowed


def history_source_positions(
    history_start: int,
    history_stop: int,
    predictor_index: int,
) -> np.ndarray:
    """Return prior response keys, excluding the predictor's diagonal key."""

    history_start = int(history_start)
    history_stop = int(history_stop)
    predictor_index = int(predictor_index)
    if not 0 <= history_start <= history_stop:
        raise ValueError("history span must be a valid half-open interval")
    stop = min(history_stop, predictor_index)
    return np.arange(history_start, max(history_start, stop), dtype=np.int64)


@dataclass(frozen=True)
class HistoryKV:
    """Per-layer response-prefix K/V projections from the prior branch."""

    checkpoint: str
    sequence_length: int
    history_start: int
    history_stop: int
    keys: tuple[Any, ...]
    values: tuple[Any, ...]


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("frozen replay requires PyTorch") from error
    return torch


class FrozenMarginReplay:
    """Frozen Llama replay that returns one candidate-B-vs-A token margin."""

    def __init__(self, model: Any, *, checkpoint: str = "<in-memory>") -> None:
        self.torch = _require_torch()
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.checkpoint = str(checkpoint)
        if int(getattr(model.config, "pretraining_tp", 1)) != 1:
            raise ValueError(
                "K/V projection hooks require model.config.pretraining_tp == 1"
            )
        backbone = getattr(model, "model", None)
        if backbone is None or not hasattr(backbone, "layers"):
            raise TypeError("model must expose LlamaForCausalLM.model.layers")
        self.backbone = backbone
        self.layers = tuple(backbone.layers)
        if not all(
            hasattr(layer.self_attn, "k_proj")
            and hasattr(layer.self_attn, "v_proj")
            for layer in self.layers
        ):
            raise TypeError("every attention layer must expose k_proj and v_proj")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: Any = "auto",
    ) -> "FrozenMarginReplay":
        """Load the local Llama checkpoint with exact eager attention."""

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("frozen replay requires Hugging Face transformers") from error

        path = Path(checkpoint).expanduser().resolve()
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        model.to(device)
        return cls(model, checkpoint=str(path))

    def _input_ids(self, input_ids: Sequence[int], predictor_index: int):
        ids = self.torch.as_tensor(
            input_ids,
            dtype=self.torch.long,
            device=self.model.get_input_embeddings().weight.device,
        )
        if ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        if int(predictor_index) != int(ids.numel()) - 1:
            raise ValueError("the teacher-forced branch must end at its predictor")
        return ids

    def _attention_mask(self, ids: Any, allowed_attention: np.ndarray | None):
        torch = self.torch
        if allowed_attention is None:
            return torch.ones((1, ids.numel()), dtype=torch.long, device=ids.device)
        allowed_tensor = torch.as_tensor(
            allowed_attention, dtype=torch.bool, device=ids.device
        )
        dtype = self.model.get_input_embeddings().weight.dtype
        additive = torch.zeros(allowed_tensor.shape, dtype=dtype, device=ids.device)
        additive.masked_fill_(~allowed_tensor, torch.finfo(dtype).min)
        return additive[None, None, :, :]

    def _predictor_hidden(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        allowed_attention: np.ndarray | None = None,
    ):
        ids = self._input_ids(input_ids, predictor_index)
        mask = self._attention_mask(ids, allowed_attention)
        output = self.backbone(
            input_ids=ids[None, :],
            attention_mask=mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return output.last_hidden_state[0, int(predictor_index)]

    def _margin(
        self,
        hidden: Any,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
    ) -> float:
        candidate_b = int(candidate_b_token_id)
        candidate_a = int(candidate_a_token_id)
        if candidate_b == candidate_a:
            raise ValueError("candidate tokens must differ at the audited position")
        weight = self.model.lm_head.weight
        vocabulary = int(weight.shape[0])
        if not 0 <= candidate_b < vocabulary or not 0 <= candidate_a < vocabulary:
            raise ValueError("candidate token id lies outside the vocabulary")
        # Use the checkpoint's actual lm_head dtype and implementation.  Only
        # the two resulting logits are converted to float32 for subtraction.
        logits = self.model.lm_head(hidden[None, None, :])[0, 0]
        candidate_ids = self.torch.tensor(
            [candidate_b, candidate_a], dtype=self.torch.long, device=logits.device
        )
        selected = logits.index_select(0, candidate_ids).float()
        return float((selected[0] - selected[1]).cpu().item())

    def score_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
    ) -> float:
        """Score ``z(B)-z(A)`` for one fixed teacher-forced prefix."""

        with self.torch.inference_mode():
            hidden = self._predictor_hidden(input_ids, predictor_index)
            return self._margin(hidden, candidate_b_token_id, candidate_a_token_id)

    def score_without_prompt_sources_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        source_positions: Sequence[int],
    ) -> float:
        """Score after removing selected prompt keys from all later queries."""

        allowed = prompt_source_allowed_attention(
            len(input_ids), predictor_index, source_positions
        )
        with self.torch.inference_mode():
            hidden = self._predictor_hidden(input_ids, predictor_index, allowed)
            return self._margin(hidden, candidate_b_token_id, candidate_a_token_id)

    def score_without_history_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_start: int,
        history_stop: int,
    ) -> float:
        """Score after the predictor stops reading strictly earlier history."""

        history = history_source_positions(
            history_start, history_stop, predictor_index
        )
        if history.size == 0:
            raise ValueError("the audited predictor has no earlier response history")
        allowed = predictor_allowed_attention(
            len(input_ids), predictor_index, history.tolist()
        )
        with self.torch.inference_mode():
            hidden = self._predictor_hidden(input_ids, predictor_index, allowed)
            return self._margin(hidden, candidate_b_token_id, candidate_a_token_id)

    def capture_history_kv(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_start: int,
        history_stop: int,
    ) -> tuple[HistoryKV, float]:
        """Capture prior history K/V and its margin in one frozen replay."""

        ids = self._input_ids(input_ids, predictor_index)
        start = int(history_start)
        declared_stop = int(history_stop)
        if not 0 <= start < declared_stop <= int(ids.numel()):
            raise ValueError("history span must be non-empty and inside input_ids")
        stop = min(declared_stop, int(predictor_index))
        if start >= stop:
            raise ValueError("the audited predictor has no earlier response history")

        keys: list[Any | None] = [None] * len(self.layers)
        values: list[Any | None] = [None] * len(self.layers)
        handles = []

        def capture(destination: list[Any | None], layer_index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                destination[layer_index] = output[0, start:stop].detach().cpu().clone()

            return hook

        for index, layer in enumerate(self.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture(keys, index)))
            handles.append(
                layer.self_attn.v_proj.register_forward_hook(capture(values, index))
            )
        try:
            with self.torch.inference_mode():
                hidden = self._predictor_hidden(input_ids, predictor_index)
                raw_margin = self._margin(
                    hidden, candidate_b_token_id, candidate_a_token_id
                )
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in keys + values):
            raise RuntimeError("a Llama attention projection hook did not run")
        return (
            HistoryKV(
                checkpoint=self.checkpoint,
                sequence_length=int(ids.numel()),
                history_start=start,
                history_stop=stop,
                keys=tuple(value for value in keys if value is not None),
                values=tuple(value for value in values if value is not None),
            ),
            raw_margin,
        )

    @contextmanager
    def _patched_history(self, history_kv: HistoryKV):
        handles = []
        start, stop = history_kv.history_start, history_kv.history_stop

        def replace(replacement: Any):
            def hook(_module: Any, _arguments: Any, output: Any):
                patched = output.clone()
                patched[:, start:stop] = replacement.to(
                    device=output.device, dtype=output.dtype
                )[None, :]
                return patched

            return hook

        for layer, key, value in zip(
            self.layers, history_kv.keys, history_kv.values, strict=True
        ):
            handles.append(layer.self_attn.k_proj.register_forward_hook(replace(key)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(replace(value)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def score_hybrid_history_margin(
        self,
        counter_input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_kv: HistoryKV,
    ) -> float:
        """Score the counter branch after only its history K/V are replaced."""

        ids = self._input_ids(counter_input_ids, predictor_index)
        if history_kv.checkpoint != self.checkpoint:
            raise ValueError("history K/V came from a different checkpoint")
        if history_kv.sequence_length != int(ids.numel()):
            raise ValueError("prior and counter branches must have identical length")
        if len(history_kv.keys) != len(self.layers) or len(history_kv.values) != len(
            self.layers
        ):
            raise ValueError("history K/V must contain every model layer")

        with self.torch.inference_mode(), self._patched_history(history_kv):
            hidden = self._predictor_hidden(counter_input_ids, predictor_index)
            return self._margin(hidden, candidate_b_token_id, candidate_a_token_id)
