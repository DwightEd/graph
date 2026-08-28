"""Faithful, frozen causal replay for answer-level attention audits.

The replay path deliberately does not generate text.  It teacher-forces the
cached token sequence and always scores the factual next token at predictor
position ``prompt_length - 1 + t``.  Counterfactuals modify either the causal
attention support or the evidence tokens; they never branch on ``argmax``.

The module imports ``transformers`` only in :meth:`FrozenCausalReplay.from_pretrained`.
This keeps pure mask/alignment helpers usable in lightweight test and analysis
environments while giving a direct, actionable error when model replay is
requested without the optional runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment import predecessor_alignment
from .cache_binding import validate_replay_attention
from .counterfactuals import COUNTERFACTUAL_NAMES, SWAPPED_EVIDENCE_NAMES


VARIANT_NAMES = COUNTERFACTUAL_NAMES

TOKEN_DIAGONAL_PROBE_SCHEME = "rademacher_token_jacobian_diagonal_v1"
DEFAULT_GRADIENT_PROBES = 8
DEFAULT_ATTRIBUTION_SEED = 20260828
MAX_GRADIENT_PROBE_ELEMENTS = 536_870_912  # 2 GiB in float32


def predictor_indices(
    prompt_length: int,
    response_length: int,
    *,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Return causal positions whose logits predict the response tokens.

    Response token ``y_t`` is stored at sequence position ``P + t`` and is
    predicted by the logits at ``P - 1 + t``.  The final response token is an
    input target, never a predictor for itself.
    """

    prompt_length = int(prompt_length)
    response_length = int(response_length)
    if prompt_length < 1:
        raise ValueError("prompt_length must be at least one")
    if response_length < 1:
        raise ValueError("response_length must be at least one")
    required = prompt_length + response_length
    if sequence_length is not None and int(sequence_length) != required:
        raise ValueError(
            "sequence_length must equal prompt_length + response_length"
        )
    return np.arange(
        prompt_length - 1,
        prompt_length - 1 + response_length,
        dtype=np.int64,
    )


def teacher_forced_alignment(
    token_ids: Sequence[int] | np.ndarray,
    prompt_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictor positions and the cached factual response targets."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    prompt_length = int(prompt_length)
    if not 0 < prompt_length < tokens.size:
        raise ValueError("prompt_length must split a non-empty prompt and response")
    alignment = predecessor_alignment(tokens, prompt_length)
    return (
        alignment.predictor_position.copy(),
        alignment.target_token_id.copy(),
    )


def causal_allowed_attention(sequence_length: int) -> np.ndarray:
    """Return a lower-triangular boolean causal support matrix."""

    sequence_length = int(sequence_length)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    return np.tri(sequence_length, sequence_length, dtype=bool)


def allowed_attention_sha256(allowed_attention: np.ndarray) -> str:
    """Bind a functional capture to one exact boolean attention support."""

    allowed = np.asarray(allowed_attention, dtype=np.bool_)
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("allowed attention must be a square matrix")
    digest = hashlib.sha256()
    digest.update(np.asarray(allowed.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(allowed).tobytes())
    return digest.hexdigest()


def build_variant_allowed_attention(
    sequence_length: int,
    prompt_length: int,
    evidence_positions: Sequence[int] | np.ndarray,
    variant: str,
) -> np.ndarray:
    """Build the attention support for one same-length replay control.

    Only strict source-to-later-query edges are intervened on.  In particular,
    the diagonal remains available, so every row retains at least one valid key.
    ``swapped_evidence`` changes token identities and therefore uses full causal
    support.
    """

    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown replay variant: {variant}")
    sequence_length = int(sequence_length)
    prompt_length = int(prompt_length)
    if not 0 < prompt_length < sequence_length:
        raise ValueError("prompt_length must split a non-empty prompt and response")
    evidence = np.asarray(evidence_positions, dtype=np.int64)
    if evidence.ndim != 1:
        raise ValueError("evidence_positions must be one-dimensional")
    if evidence.size and (
        int(evidence.min()) < 0 or int(evidence.max()) >= prompt_length
    ):
        raise ValueError("evidence positions must lie inside the prompt")
    evidence = np.unique(evidence)

    allowed = causal_allowed_attention(sequence_length)
    remove_evidence = variant in {"no_evidence", "no_evidence_no_history"}
    remove_history = variant in {"no_history", "no_evidence_no_history"}
    if remove_evidence and evidence.size:
        for query in range(sequence_length):
            sources = evidence[evidence < query]
            allowed[query, sources] = False
    if remove_history:
        for query in range(prompt_length + 1, sequence_length):
            allowed[query, prompt_length:query] = False

    if not allowed.any(axis=1).all():
        raise ValueError("a replay intervention fully masked an attention row")
    return allowed


def replace_evidence_tokens(
    token_ids: Sequence[int] | np.ndarray,
    evidence_positions: Sequence[int] | np.ndarray,
    replacement_token_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Replace a fixed evidence span without changing causal alignment."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    positions = np.asarray(evidence_positions, dtype=np.int64)
    replacements = np.asarray(replacement_token_ids, dtype=np.int64)
    if tokens.ndim != 1 or positions.ndim != 1 or replacements.ndim != 1:
        raise ValueError("token and evidence arrays must be one-dimensional")
    if positions.size != replacements.size:
        raise ValueError(
            "replacement evidence must have exactly one token per evidence position"
        )
    if positions.size and (
        int(positions.min()) < 0 or int(positions.max()) >= tokens.size
    ):
        raise ValueError("evidence position is outside token_ids")
    output = tokens.copy()
    output[positions] = replacements
    return output


def _numpy_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def score_dense_logits(
    logits: np.ndarray,
    target_ids: Sequence[int] | np.ndarray,
    *,
    reference_logits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference scorer used by tests and small diagnostic arrays.

    Returns per-token chosen log-probability, chosen-vs-best-other margin, and
    Jensen-Shannon divergence from ``reference_logits``.  The chosen class is
    always ``target_ids`` even when another vocabulary item is the argmax.
    Production model replay uses the equivalent chunked Torch implementation.
    """

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("logits must be [token, vocab] and targets must be [token]")
    rows, vocabulary = logits.shape
    if vocabulary < 2:
        raise ValueError("at least two vocabulary entries are required")
    if targets.size and (int(targets.min()) < 0 or int(targets.max()) >= vocabulary):
        raise ValueError("target id lies outside the vocabulary")

    chosen = logits[np.arange(rows), targets]
    log_partition = _numpy_logsumexp(logits, axis=-1)
    without_chosen = logits.copy()
    without_chosen[np.arange(rows), targets] = -np.inf
    margin = chosen - np.max(without_chosen, axis=-1)
    chosen_logprob = chosen - log_partition

    if reference_logits is None:
        jsd = np.zeros(rows, dtype=np.float64)
    else:
        reference = np.asarray(reference_logits, dtype=np.float64)
        if reference.shape != logits.shape:
            raise ValueError("reference_logits must have the same shape as logits")
        log_p = logits - log_partition[:, None]
        reference_log_partition = _numpy_logsumexp(reference, axis=-1)
        log_q = reference - reference_log_partition[:, None]
        log_m = np.logaddexp(log_p, log_q) - math.log(2.0)
        p = np.exp(log_p)
        q = np.exp(log_q)
        jsd = 0.5 * (
            (p * (log_p - log_m)).sum(axis=-1)
            + (q * (log_q - log_m)).sum(axis=-1)
        )
        jsd = np.maximum(jsd, 0.0)
    return (
        chosen_logprob.astype(np.float32),
        margin.astype(np.float32),
        jsd.astype(np.float32),
    )


def q_to_kv_mapping(query_heads: int, kv_heads: int) -> np.ndarray:
    """Return the grouped-query attention map from query to KV head."""

    query_heads, kv_heads = int(query_heads), int(kv_heads)
    if query_heads < 1 or kv_heads < 1 or query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    return np.arange(query_heads, dtype=np.int64) // (query_heads // kv_heads)


def rademacher_token_probes(
    probe_count: int,
    response_length: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic iid token-space Rademacher probes.

    A local NumPy generator is used so audit reproducibility does not depend on
    the global Torch/NumPy random state or the selected accelerator.
    """

    probe_count = int(probe_count)
    response_length = int(response_length)
    seed = int(seed)
    if probe_count < 1:
        raise ValueError("gradient probe count must be at least one")
    if response_length < 1:
        raise ValueError("response length must be at least one")
    if seed < 0:
        raise ValueError("attribution seed must be non-negative")
    bits = np.random.default_rng(seed).integers(
        0,
        2,
        size=(probe_count, response_length),
        dtype=np.int8,
    )
    return (bits.astype(np.float32) * 2.0) - 1.0


def apply_token_probe_to_vjp(vjp: Any, probe: Any):
    """Multiply VJP predictor row ``s`` by the same probe sign ``z_s``.

    If ``vjp[s] = sum_t z_t d y_t / d c_s``, averaging this result over
    Rademacher probes estimates the token-diagonal Jacobian block
    ``d y_s / d c_s``.  Signed probes must be averaged before downstream code
    forms absolute functional energy.
    """

    torch = _require_torch()
    if not torch.is_tensor(vjp):
        vjp = torch.as_tensor(vjp)
    signs = torch.as_tensor(probe, dtype=vjp.dtype, device=vjp.device)
    if signs.ndim != 1 or signs.shape[0] != vjp.shape[0]:
        raise ValueError("probe must have one sign per VJP token row")
    return vjp * signs.reshape(signs.shape[0], *((1,) * (vjp.ndim - 1)))


@dataclass(frozen=True)
class VariantScores:
    """Teacher-forced response scores for one counterfactual replay."""

    name: str
    token_ids: np.ndarray | None
    prompt_length: int
    predictor_indices: np.ndarray
    target_ids: np.ndarray
    chosen_logprob: np.ndarray
    chosen_vs_best_other_margin: np.ndarray
    vocab_jsd_from_full: np.ndarray
    available: bool = True
    unavailable_reason: str | None = None

    def summary(self) -> dict[str, float]:
        return {
            "chosen_logprob_mean": float(np.mean(self.chosen_logprob)),
            "chosen_vs_best_other_margin_mean": float(
                np.mean(self.chosen_vs_best_other_margin)
            ),
            "vocab_jsd_from_full_mean": float(np.mean(self.vocab_jsd_from_full)),
        }


@dataclass(frozen=True)
class ReplayResult:
    """Aligned scores for all registered mechanism counterfactuals."""

    checkpoint: str
    variants: Mapping[str, VariantScores]

    def validate(self) -> "ReplayResult":
        if tuple(self.variants) != VARIANT_NAMES:
            raise ValueError("replay variants are missing or out of canonical order")
        full = self.variants["full"]
        if not full.available:
            raise ValueError("the full replay variant must be available")
        targets = full.target_ids
        for name, score in self.variants.items():
            if not np.array_equal(score.target_ids, targets):
                raise ValueError(f"{name} does not teacher-force the same response")
            expected_shape = targets.shape
            arrays = (
                score.predictor_indices,
                score.chosen_logprob,
                score.chosen_vs_best_other_margin,
                score.vocab_jsd_from_full,
            )
            if any(np.asarray(value).shape != expected_shape for value in arrays):
                raise ValueError(f"{name} score arrays are not response aligned")
            metrics = np.concatenate(
                (
                    np.asarray(score.chosen_logprob, dtype=np.float64),
                    np.asarray(
                        score.chosen_vs_best_other_margin, dtype=np.float64
                    ),
                    np.asarray(score.vocab_jsd_from_full, dtype=np.float64),
                )
            )
            if score.available:
                if score.token_ids is None:
                    raise ValueError(f"{name} is available but has no replay tokens")
                if score.unavailable_reason is not None:
                    raise ValueError(
                        f"{name} is available but states an unavailable reason"
                    )
                if not np.isfinite(metrics).all():
                    raise ValueError(f"{name} available metrics must be finite")
            else:
                if score.token_ids is not None:
                    raise ValueError(
                        f"{name} is unavailable but exposes fabricated replay tokens"
                    )
                if not score.unavailable_reason:
                    raise ValueError(f"{name} must state why replay is unavailable")
                if not np.isnan(metrics).all():
                    raise ValueError(
                        f"{name} unavailable metrics must be explicit NaN values"
                    )
                if not np.array_equal(
                    score.predictor_indices, full.predictor_indices
                ):
                    raise ValueError(
                        f"{name} unavailable predictors must preserve factual alignment"
                    )
        return self


@dataclass(frozen=True)
class FunctionalCapture:
    """Baseline value-path capture with token-diagonal Jacobian probes.

    Shapes are ``value_states=[layer, sequence, kv_head, head_dim]`` and
    ``o_proj_input_gradients=[layer, response_token, query_head, head_dim]``.
    The latter is the signed mean of
    ``o_proj_input_gradient_probes=[probe, layer, response_token, query_head,
    head_dim]``.  Each probe estimates the token-diagonal chosen-logprob
    Jacobian; it is not the gradient of the diagnostic answer-mean objective.
    """

    checkpoint: str
    token_ids: Any
    predictor_indices: Any
    target_ids: Any
    chosen_logprob: Any
    predictor_hidden: Any
    allowed_attention_sha256: str
    attention_cache_binding: Mapping[str, object] | None
    objective: float
    value_states: Any
    o_proj_input_gradient_probes: Any
    o_proj_input_gradients: Any
    gradient_probe_count: int
    gradient_probe_seed: int
    gradient_probe_scheme: str
    q_to_kv: Any
    head_count: int
    kv_head_count: int
    head_dim: int

    def payload(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "token_ids": self.token_ids,
            "predictor_indices": self.predictor_indices,
            "target_ids": self.target_ids,
            "chosen_logprob": self.chosen_logprob,
            "predictor_hidden": self.predictor_hidden,
            "allowed_attention_sha256": self.allowed_attention_sha256,
            "attention_cache_binding": self.attention_cache_binding,
            "objective": self.objective,
            "value_states": self.value_states,
            "o_proj_input_gradient_probes": self.o_proj_input_gradient_probes,
            "o_proj_input_gradients": self.o_proj_input_gradients,
            "gradient_probe_count": self.gradient_probe_count,
            "gradient_probe_seed": self.gradient_probe_seed,
            "gradient_probe_scheme": self.gradient_probe_scheme,
            "q_to_kv": self.q_to_kv,
            "head_count": self.head_count,
            "kv_head_count": self.kv_head_count,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True)
class _ChunkScores:
    chosen_logprob: Any
    margin: Any
    log_partition: Any


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "Frozen causal replay requires PyTorch; install the project runtime "
            "dependencies before loading a checkpoint"
        ) from error
    return torch


def _as_1d_long(values: Any, *, device: Any = None):
    torch = _require_torch()
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    if tensor.ndim != 1:
        raise ValueError("token ids must be one-dimensional")
    return tensor


def _target_logit(hidden: Any, lm_head: Any, targets: Any):
    torch = _require_torch()
    weight = lm_head.weight
    target_weight = weight.index_select(0, targets.to(weight.device)).float()
    hidden_float = hidden.to(weight.device).float()
    result = (hidden_float * target_weight).sum(dim=-1)
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        result = result + bias.index_select(0, targets.to(bias.device)).float()
    return result


def _chunk_logits(hidden: Any, lm_head: Any, start: int, stop: int):
    torch = _require_torch()
    import torch.nn.functional as functional

    weight = lm_head.weight[start:stop]
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        bias = bias[start:stop]
    return functional.linear(hidden.to(weight.device), weight, bias).float()


def _chunk_scores(
    hidden: Any,
    targets: Any,
    lm_head: Any,
    *,
    vocab_chunk_size: int,
) -> _ChunkScores:
    torch = _require_torch()
    if hidden.ndim != 2 or targets.shape != (hidden.shape[0],):
        raise ValueError("hidden must be [token, hidden] and targets must be [token]")
    vocabulary = int(lm_head.weight.shape[0])
    if vocabulary < 2 or vocab_chunk_size < 1:
        raise ValueError("vocabulary and vocab_chunk_size must be positive")
    targets = targets.to(lm_head.weight.device)
    if bool(((targets < 0) | (targets >= vocabulary)).any()):
        raise ValueError("target id lies outside the language-model vocabulary")

    chosen = _target_logit(hidden, lm_head, targets)
    log_partition = torch.full_like(chosen, -torch.inf, dtype=torch.float32)
    best_other = torch.full_like(chosen, -torch.inf, dtype=torch.float32)
    rows = torch.arange(hidden.shape[0], device=lm_head.weight.device)
    for start in range(0, vocabulary, int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), vocabulary)
        logits = _chunk_logits(hidden, lm_head, start, stop)
        log_partition = torch.logaddexp(
            log_partition,
            torch.logsumexp(logits, dim=-1),
        )
        local = targets - start
        inside = (local >= 0) & (local < stop - start)
        if bool(inside.any()):
            logits = logits.clone()
            logits[rows[inside], local[inside]] = -torch.inf
        best_other = torch.maximum(best_other, logits.max(dim=-1).values)
    return _ChunkScores(
        chosen_logprob=chosen - log_partition,
        margin=chosen - best_other,
        log_partition=log_partition,
    )


def _chunked_jsd(
    hidden: Any,
    scores: _ChunkScores,
    reference_hidden: Any,
    reference_scores: _ChunkScores,
    lm_head: Any,
    *,
    vocab_chunk_size: int,
):
    torch = _require_torch()
    if hidden.shape != reference_hidden.shape:
        raise ValueError("variant and full hidden states must have matching shapes")
    vocabulary = int(lm_head.weight.shape[0])
    jsd = torch.zeros(
        hidden.shape[0],
        dtype=torch.float32,
        device=lm_head.weight.device,
    )
    for start in range(0, vocabulary, int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), vocabulary)
        log_p = (
            _chunk_logits(hidden, lm_head, start, stop)
            - scores.log_partition[:, None]
        )
        log_q = (
            _chunk_logits(reference_hidden, lm_head, start, stop)
            - reference_scores.log_partition[:, None]
        )
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
        jsd = jsd + 0.5 * (
            (log_p.exp() * (log_p - log_m)).sum(dim=-1)
            + (log_q.exp() * (log_q - log_m)).sum(dim=-1)
        )
    return jsd.clamp_min(0.0)


def _differentiable_chosen_logprob(
    hidden: Any,
    targets: Any,
    lm_head: Any,
    *,
    vocab_chunk_size: int,
):
    """Chunked log-probabilities that preserve the hidden-state graph."""

    return _chunk_scores(
        hidden,
        targets,
        lm_head,
        vocab_chunk_size=vocab_chunk_size,
    ).chosen_logprob


class FrozenCausalReplay:
    """Frozen, teacher-forced replay for Llama-like Hugging Face causal LMs."""

    def __init__(self, model: Any, *, checkpoint: str = "<in-memory>") -> None:
        torch = _require_torch()
        self.model = model
        self.checkpoint = str(checkpoint)
        self.model.eval()
        self.model.requires_grad_(False)
        config = self.model.config
        self.head_count = int(getattr(config, "num_attention_heads"))
        self.kv_head_count = int(
            getattr(config, "num_key_value_heads", self.head_count)
        )
        hidden_size = int(getattr(config, "hidden_size"))
        self.head_dim = int(getattr(config, "head_dim", hidden_size // self.head_count))
        if self.head_count * self.head_dim != hidden_size:
            raise ValueError("model hidden size is incompatible with its query heads")
        q_to_kv_mapping(self.head_count, self.kv_head_count)
        if not hasattr(self.model, "lm_head"):
            raise TypeError("causal model must expose lm_head")
        layers = self._layers()
        for index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "v_proj") or not hasattr(
                attention, "o_proj"
            ):
                raise TypeError(
                    f"layer {index} is not Llama-like: v_proj/o_proj are required"
                )
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("failed to freeze causal language model parameters")
        self._torch = torch

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        torch_dtype: Any = None,
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> "FrozenCausalReplay":
        """Load exactly one checkpoint with eager custom-mask attention."""

        _require_torch()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Frozen causal replay requires Hugging Face transformers; "
                "install transformers in the experiment environment"
            ) from error

        identifier = str(checkpoint)
        kwargs: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
            "attn_implementation": "eager",
        }
        if revision is not None:
            kwargs["revision"] = revision
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(identifier, **kwargs)
        if device is not None:
            model.to(device)
        checkpoint_path = Path(identifier).expanduser()
        if checkpoint_path.exists():
            checkpoint_identity = str(checkpoint_path.resolve())
        else:
            checkpoint_identity = (
                identifier if revision is None else f"{identifier}@{revision}"
            )
        return cls(model, checkpoint=checkpoint_identity)

    def _backbone(self):
        backbone = getattr(self.model, "model", None)
        if backbone is None or not hasattr(backbone, "layers"):
            raise TypeError("causal model must expose a Llama-like .model.layers backbone")
        return backbone

    def _layers(self):
        return list(self._backbone().layers)

    def _embedding_device(self):
        return self.model.get_input_embeddings().weight.device

    def _additive_mask(self, allowed_attention: np.ndarray, *, dtype: Any, device: Any):
        torch = self._torch
        allowed = torch.as_tensor(allowed_attention, dtype=torch.bool, device=device)
        if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
            raise ValueError("allowed attention must be a square [query, key] matrix")
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("allowed attention contains a fully masked query")
        additive = torch.zeros(allowed.shape, dtype=dtype, device=device)
        additive.masked_fill_(~allowed, torch.finfo(dtype).min)
        return additive[None, None, :, :]

    def _forward_hidden(self, token_ids: Any, allowed_attention: np.ndarray):
        torch = self._torch
        token_ids = _as_1d_long(token_ids, device=self._embedding_device())
        embeddings = self.model.get_input_embeddings()(token_ids[None, :])
        position_ids = torch.arange(
            token_ids.numel(), dtype=torch.long, device=embeddings.device
        )[None, :]
        attention_mask = self._additive_mask(
            allowed_attention,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        output = self._backbone()(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        return hidden[0]

    @staticmethod
    def _variant_field(variant: Any, name: str, default: Any = None) -> Any:
        if isinstance(variant, Mapping):
            return variant.get(name, default)
        return getattr(variant, name, default)

    def _default_variants(
        self,
        token_ids: np.ndarray,
        prompt_length: int,
        evidence_positions: np.ndarray,
        replacement_evidence_token_ids: Any,
    ) -> dict[str, dict[str, Any]]:
        if replacement_evidence_token_ids is None:
            raise ValueError(
                "replacement_evidence_token_ids is required for swapped_evidence"
            )
        swapped = replace_evidence_tokens(
            token_ids,
            evidence_positions,
            replacement_evidence_token_ids,
        )
        result = {
            name: {
                "token_ids": token_ids,
                "prompt_length": prompt_length,
                "allowed_attention": build_variant_allowed_attention(
                    token_ids.size,
                    prompt_length,
                    evidence_positions,
                    name,
                ),
            }
            for name in VARIANT_NAMES[:4]
        }
        result[SWAPPED_EVIDENCE_NAMES[0]] = {
            "token_ids": swapped,
            "prompt_length": prompt_length,
            "allowed_attention": build_variant_allowed_attention(
                token_ids.size,
                prompt_length,
                evidence_positions,
                SWAPPED_EVIDENCE_NAMES[0],
            ),
        }
        for name in SWAPPED_EVIDENCE_NAMES[1:]:
            result[name] = {
                "available": False,
                "unavailable_reason": (
                    "the compatibility replay API supplied only one evidence donor"
                ),
            }
        return result

    def _validated_capture_hidden(
        self,
        capture: FunctionalCapture,
        *,
        token_ids: np.ndarray,
        predictor_indices: np.ndarray,
        target_ids: np.ndarray,
        allowed_attention: np.ndarray,
    ):
        """Return a detached full-reference hidden state after exact binding checks."""

        torch = self._torch

        def numpy_integer(value: Any, name: str) -> np.ndarray:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError(f"baseline capture {name} must be an integer vector")
            return array.astype(np.int64, copy=False)

        if str(capture.checkpoint) != self.checkpoint:
            raise ValueError("baseline capture checkpoint does not match replay model")
        if not np.array_equal(
            numpy_integer(capture.token_ids, "token_ids"), token_ids
        ):
            raise ValueError("baseline capture token ids do not match replay tokens")
        if not np.array_equal(
            numpy_integer(capture.predictor_indices, "predictor_indices"),
            predictor_indices,
        ):
            raise ValueError("baseline capture predictors are misaligned")
        if not np.array_equal(
            numpy_integer(capture.target_ids, "target_ids"), target_ids
        ):
            raise ValueError("baseline capture factual targets are misaligned")
        expected_mask_sha = allowed_attention_sha256(allowed_attention)
        if str(capture.allowed_attention_sha256) != expected_mask_sha:
            raise ValueError("baseline capture attention support does not match full replay")

        hidden = capture.predictor_hidden
        if not torch.is_tensor(hidden):
            hidden = torch.as_tensor(hidden)
        if hidden.ndim != 2 or hidden.shape != (
            target_ids.size,
            self.model.lm_head.weight.shape[1],
        ):
            raise ValueError("baseline capture predictor hidden state has the wrong shape")
        if bool(hidden.requires_grad):
            raise ValueError("baseline capture predictor hidden state must be detached")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("baseline capture predictor hidden state is non-finite")
        return hidden.detach()

    def replay(
        self,
        token_ids: Sequence[int] | np.ndarray,
        prompt_length: int,
        evidence_positions: Sequence[int] | np.ndarray,
        *,
        replacement_evidence_token_ids: Sequence[int] | np.ndarray | None = None,
        variants: Mapping[str, Any] | None = None,
        baseline_capture: FunctionalCapture | None = None,
        vocab_chunk_size: int = 4096,
    ) -> ReplayResult:
        """Replay the seven registered counterfactuals against fixed targets.

        ``variants`` may contain dataclasses supplied by the role/counterfactual
        module.  Each entry must expose ``token_ids`` and ``allowed_attention``;
        masks are consumed verbatim and are never reconstructed from labels.
        """

        torch = self._torch
        baseline_tokens = np.asarray(token_ids, dtype=np.int64)
        if baseline_tokens.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        evidence = np.asarray(evidence_positions, dtype=np.int64)
        baseline_predictors, baseline_targets = teacher_forced_alignment(
            baseline_tokens,
            prompt_length,
        )
        if variants is None:
            variants = self._default_variants(
                baseline_tokens,
                int(prompt_length),
                evidence,
                replacement_evidence_token_ids,
            )
        if set(variants) != set(VARIANT_NAMES):
            raise ValueError("variants must contain exactly the seven registered names")

        hidden_by_name: dict[str, Any] = {}
        alignment_by_name: dict[
            str, tuple[np.ndarray | None, np.ndarray, np.ndarray, int]
        ] = {}
        unavailable_reason: dict[str, str] = {}
        with torch.inference_mode():
            for name in VARIANT_NAMES:
                variant = variants[name]
                if not bool(self._variant_field(variant, "available", True)):
                    if name not in SWAPPED_EVIDENCE_NAMES:
                        raise ValueError(
                            f"required replay variant {name} is unavailable"
                        )
                    reason = self._variant_field(
                        variant,
                        "unavailable_reason",
                        "no runnable counterfactual was provided",
                    )
                    if not reason:
                        reason = "no runnable counterfactual was provided"
                    unavailable_reason[name] = str(reason)
                    alignment_by_name[name] = (
                        None,
                        baseline_predictors.copy(),
                        baseline_targets.copy(),
                        int(prompt_length),
                    )
                    continue
                variant_tokens = np.asarray(
                    self._variant_field(variant, "token_ids"), dtype=np.int64
                )
                variant_prompt = int(
                    self._variant_field(variant, "prompt_length", prompt_length)
                )
                allowed = self._variant_field(variant, "allowed_attention")
                if allowed is None:
                    allowed = self._variant_field(variant, "attention_mask")
                allowed = np.asarray(allowed, dtype=bool)
                if allowed.shape != (variant_tokens.size, variant_tokens.size):
                    raise ValueError(f"{name} attention support has the wrong shape")
                predictors, targets = teacher_forced_alignment(
                    variant_tokens,
                    variant_prompt,
                )
                if not np.array_equal(targets, baseline_targets):
                    raise ValueError(
                        f"{name} changes factual response targets; causal replay must "
                        "teacher-force the cached answer"
                    )
                if name == "full" and baseline_capture is not None:
                    if not np.array_equal(variant_tokens, baseline_tokens):
                        raise ValueError(
                            "full variant tokens do not match the captured baseline"
                        )
                    hidden_by_name[name] = self._validated_capture_hidden(
                        baseline_capture,
                        token_ids=baseline_tokens,
                        predictor_indices=predictors,
                        target_ids=targets,
                        allowed_attention=allowed,
                    )
                else:
                    hidden = self._forward_hidden(variant_tokens, allowed)
                    predictor_tensor = _as_1d_long(predictors, device=hidden.device)
                    hidden_by_name[name] = hidden.index_select(
                        0, predictor_tensor
                    ).detach()
                alignment_by_name[name] = (
                    variant_tokens,
                    predictors,
                    targets,
                    variant_prompt,
                )

            target_tensor = _as_1d_long(
                baseline_targets,
                device=self.model.lm_head.weight.device,
            )
            reference_hidden = hidden_by_name["full"].to(
                self.model.lm_head.weight.device
            )
            reference_scores = _chunk_scores(
                reference_hidden,
                target_tensor,
                self.model.lm_head,
                vocab_chunk_size=vocab_chunk_size,
            )
            if baseline_capture is not None:
                captured_logprob = baseline_capture.chosen_logprob
                if torch.is_tensor(captured_logprob):
                    captured_logprob = captured_logprob.detach().float().cpu().numpy()
                captured_logprob = np.asarray(captured_logprob, dtype=np.float32)
                rescored_logprob = (
                    reference_scores.chosen_logprob.detach().float().cpu().numpy()
                )
                if captured_logprob.shape != baseline_targets.shape or not np.allclose(
                    captured_logprob,
                    rescored_logprob,
                    atol=2e-4,
                    rtol=2e-4,
                ):
                    raise ValueError(
                        "baseline capture chosen log-probabilities do not bind its "
                        "predictor hidden state"
                    )
            output: dict[str, VariantScores] = {}
            for name in VARIANT_NAMES:
                variant_tokens, predictors, targets, variant_prompt = alignment_by_name[
                    name
                ]
                if name in unavailable_reason:
                    missing = np.full(targets.shape, np.nan, dtype=np.float32)
                    output[name] = VariantScores(
                        name=name,
                        token_ids=None,
                        prompt_length=variant_prompt,
                        predictor_indices=predictors.copy(),
                        target_ids=targets.copy(),
                        chosen_logprob=missing.copy(),
                        chosen_vs_best_other_margin=missing.copy(),
                        vocab_jsd_from_full=missing.copy(),
                        available=False,
                        unavailable_reason=unavailable_reason[name],
                    )
                    continue
                hidden = hidden_by_name[name].to(self.model.lm_head.weight.device)
                scores = (
                    reference_scores
                    if name == "full"
                    else _chunk_scores(
                        hidden,
                        target_tensor,
                        self.model.lm_head,
                        vocab_chunk_size=vocab_chunk_size,
                    )
                )
                jsd = (
                    torch.zeros_like(scores.chosen_logprob)
                    if name == "full"
                    else _chunked_jsd(
                        hidden,
                        scores,
                        reference_hidden,
                        reference_scores,
                        self.model.lm_head,
                        vocab_chunk_size=vocab_chunk_size,
                    )
                )
                assert variant_tokens is not None
                output[name] = VariantScores(
                    name=name,
                    token_ids=variant_tokens.copy(),
                    prompt_length=variant_prompt,
                    predictor_indices=predictors.copy(),
                    target_ids=targets.copy(),
                    chosen_logprob=scores.chosen_logprob.float().cpu().numpy(),
                    chosen_vs_best_other_margin=scores.margin.float().cpu().numpy(),
                    vocab_jsd_from_full=jsd.float().cpu().numpy(),
                    available=True,
                    unavailable_reason=None,
                )
        return ReplayResult(checkpoint=self.checkpoint, variants=output).validate()

    def capture_baseline(
        self,
        token_ids: Sequence[int] | np.ndarray,
        prompt_length: int,
        *,
        allowed_attention: np.ndarray | None = None,
        vocab_chunk_size: int = 4096,
        gradient_probes: int = DEFAULT_GRADIENT_PROBES,
        attribution_seed: int = DEFAULT_ATTRIBUTION_SEED,
        expected_graph: Any | None = None,
    ) -> FunctionalCapture:
        """Capture values and estimate each token's same-row chosen-logprob gradient.

        Let ``y_t`` be the factual chosen-token log-probability and ``c_s`` the
        head context entering ``o_proj`` at predictor row ``s``.  For each iid
        Rademacher vector ``z`` this method backpropagates ``sum_t z_t y_t`` and
        records ``z_s d(sum_t z_t y_t)/d c_s``.  Averaging probes estimates the
        Jacobian diagonal ``d y_s/d c_s`` while cancelling later-token effects
        on earlier predictor rows.  The stored ``objective`` is only the mean
        log-probability diagnostic; no gradient is taken from that mean.
        """

        torch = self._torch
        tokens_np = np.asarray(token_ids, dtype=np.int64)
        predictors_np, targets_np = teacher_forced_alignment(tokens_np, prompt_length)
        if allowed_attention is None:
            allowed_attention = causal_allowed_attention(tokens_np.size)
        allowed_attention = np.asarray(allowed_attention, dtype=bool)
        if allowed_attention.shape != (tokens_np.size, tokens_np.size):
            raise ValueError("allowed_attention has the wrong shape")
        probe_values = rademacher_token_probes(
            gradient_probes,
            targets_np.size,
            attribution_seed,
        )

        layers = self._layers()
        probe_elements = (
            int(gradient_probes)
            * len(layers)
            * int(targets_np.size)
            * self.head_count
            * self.head_dim
        )
        if probe_elements > MAX_GRADIENT_PROBE_ELEMENTS:
            gibibytes = probe_elements * 4 / (1024**3)
            raise ValueError(
                "token-diagonal gradient probes would require "
                f"{gibibytes:.2f} GiB of CPU float32 storage; reduce "
                "--gradient-probes or split unusually long responses"
            )
        value_outputs: list[Any | None] = [None] * len(layers)
        o_proj_inputs: list[Any | None] = [None] * len(layers)
        gradient_outputs: list[list[Any | None]] = [
            [None] * len(layers) for _ in range(int(gradient_probes))
        ]
        module_handles = []
        tensor_handles = []
        active_probe: list[int | None] = [None]

        def capture_value(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                if not torch.is_tensor(output):
                    raise RuntimeError("v_proj hook expected a tensor output")
                value_outputs[index] = output.detach().cpu()

            return hook

        def capture_o_input(index: int):
            def hook(_module: Any, arguments: Any) -> None:
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise RuntimeError("o_proj hook expected a tensor input")
                tensor = arguments[0]
                o_proj_inputs[index] = tensor

                def capture_gradient(gradient: Any):
                    probe_index = active_probe[0]
                    if probe_index is None:
                        raise RuntimeError(
                            "o_proj gradient arrived outside an attribution probe"
                        )
                    if gradient.shape != (
                        1,
                        tokens_np.size,
                        self.head_count * self.head_dim,
                    ):
                        raise RuntimeError(
                            f"unexpected o_proj gradient shape at layer {index}"
                        )
                    predictor = torch.as_tensor(
                        predictors_np,
                        dtype=torch.long,
                        device=gradient.device,
                    )
                    rows = gradient[0].index_select(0, predictor).reshape(
                        targets_np.size,
                        self.head_count,
                        self.head_dim,
                    )
                    signed = apply_token_probe_to_vjp(
                        rows,
                        probe_values[probe_index],
                    )
                    gradient_outputs[probe_index][index] = (
                        signed.detach().float().cpu()
                    )
                    return gradient

                tensor_handles.append(tensor.register_hook(capture_gradient))

            return hook

        for index, layer in enumerate(layers):
            module_handles.append(
                layer.self_attn.v_proj.register_forward_hook(capture_value(index))
            )
            module_handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(capture_o_input(index))
            )

        self.model.zero_grad(set_to_none=True)
        try:
            token_tensor = _as_1d_long(tokens_np, device=self._embedding_device())
            embeddings = (
                self.model.get_input_embeddings()(token_tensor[None, :])
                .detach()
                .requires_grad_(True)
            )
            position_ids = torch.arange(
                token_tensor.numel(), dtype=torch.long, device=embeddings.device
            )[None, :]
            attention_mask = self._additive_mask(
                allowed_attention,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            output = self._backbone()(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=expected_graph is not None,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = output[0]
            attention_cache_binding = None
            if expected_graph is not None:
                replay_attentions = getattr(output, "attentions", None)
                if replay_attentions is None:
                    raise RuntimeError(
                        "the eager replay did not return attention weights needed "
                        "to bind the checkpoint to the frozen cache"
                    )
                attention_cache_binding = validate_replay_attention(
                    expected_graph,
                    replay_attentions,
                ).as_dict()
            predictor_tensor = _as_1d_long(predictors_np, device=hidden.device)
            target_tensor = _as_1d_long(
                targets_np,
                device=self.model.lm_head.weight.device,
            )
            predictor_hidden = hidden[0].index_select(0, predictor_tensor).to(
                self.model.lm_head.weight.device
            )
            chosen_logprob = _differentiable_chosen_logprob(
                predictor_hidden,
                target_tensor,
                self.model.lm_head,
                vocab_chunk_size=vocab_chunk_size,
            )
            objective = chosen_logprob.mean()
            detached_predictor_hidden = predictor_hidden.detach()
            for probe_index in range(int(gradient_probes)):
                active_probe[0] = probe_index
                embeddings.grad = None
                probe = torch.as_tensor(
                    probe_values[probe_index],
                    device=chosen_logprob.device,
                    dtype=chosen_logprob.dtype,
                )
                probe_objective = (chosen_logprob * probe).sum()
                probe_objective.backward(
                    retain_graph=probe_index + 1 < int(gradient_probes)
                )
                if any(value is None for value in gradient_outputs[probe_index]):
                    raise RuntimeError(
                        f"attribution probe {probe_index} missed an o_proj gradient"
                    )
            active_probe[0] = None
            embeddings.grad = None

            if any(value is None for value in value_outputs) or any(
                value is None for value in o_proj_inputs
            ):
                raise RuntimeError(
                    "projection hooks did not fire; the checkpoint does not use the "
                    "expected Llama-like v_proj/o_proj path"
                )
            values = []
            for index, (value, o_input) in enumerate(
                zip(value_outputs, o_proj_inputs, strict=True)
            ):
                assert value is not None and o_input is not None
                if value.shape != (
                    1,
                    tokens_np.size,
                    self.kv_head_count * self.head_dim,
                ):
                    raise RuntimeError(f"unexpected v_proj shape at layer {index}")
                if o_input.shape != (
                    1,
                    tokens_np.size,
                    self.head_count * self.head_dim,
                ):
                    raise RuntimeError(f"unexpected o_proj input shape at layer {index}")
                values.append(
                    value[0].reshape(
                        tokens_np.size,
                        self.kv_head_count,
                        self.head_dim,
                    )
                )
            value_states = torch.stack(values)
            probe_layers = [
                torch.stack([value for value in row if value is not None])
                for row in gradient_outputs
            ]
            o_gradient_probes = torch.stack(probe_layers)
            expected_probe_shape = (
                int(gradient_probes),
                len(layers),
                targets_np.size,
                self.head_count,
                self.head_dim,
            )
            if o_gradient_probes.shape != expected_probe_shape:
                raise RuntimeError("token-diagonal gradient probes have the wrong shape")
            # Average signed diagonal estimates first.  Downstream functional
            # flow may then take absolute energy without introducing probe bias.
            o_gradients = o_gradient_probes.mean(dim=0)
        finally:
            active_probe[0] = None
            for handle in tensor_handles:
                handle.remove()
            for handle in module_handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)

        return FunctionalCapture(
            checkpoint=self.checkpoint,
            token_ids=torch.as_tensor(tokens_np, dtype=torch.long),
            predictor_indices=torch.as_tensor(predictors_np, dtype=torch.long),
            target_ids=torch.as_tensor(targets_np, dtype=torch.long),
            chosen_logprob=chosen_logprob.detach().float().cpu(),
            predictor_hidden=detached_predictor_hidden,
            allowed_attention_sha256=allowed_attention_sha256(
                allowed_attention
            ),
            attention_cache_binding=attention_cache_binding,
            objective=float(objective.detach().float().cpu().item()),
            value_states=value_states,
            o_proj_input_gradient_probes=o_gradient_probes,
            o_proj_input_gradients=o_gradients,
            gradient_probe_count=int(gradient_probes),
            gradient_probe_seed=int(attribution_seed),
            gradient_probe_scheme=TOKEN_DIAGONAL_PROBE_SCHEME,
            q_to_kv=torch.as_tensor(
                q_to_kv_mapping(self.head_count, self.kv_head_count),
                dtype=torch.long,
            ),
            head_count=self.head_count,
            kv_head_count=self.kv_head_count,
            head_dim=self.head_dim,
        )
