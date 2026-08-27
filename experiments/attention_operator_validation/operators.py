"""Extract reusable head-operator geometry from a frozen causal language model.

For one Transformer layer and query head ``h`` the linear value-path operator is

    B_h = W^O_h W^V_{kv(h)}.

The full ``B_h`` matrices are never required to compare attention codes.  Their
Frobenius Gram matrix

    G[h, g] = <B_h, B_g>_F

is sufficient because ``||W(z)-W(z')||_F^2 = (z-z')^T G (z-z')``.  This module
extracts and stores ``G`` and a square-root factor once, so later attention-cache
analyses do not need to reload the language model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Iterable

import torch


OPERATOR_SCHEMA = "attention-hypernetwork-operator-geometry"
OPERATOR_VERSION = 1


@dataclass(frozen=True)
class OperatorGeometry:
    """Reusable functional geometry for cross-head attention codes."""

    model_path: str
    architecture: str
    layer_count: int
    head_count: int
    kv_head_count: int
    head_dim: int
    hidden_size: int
    gram: torch.Tensor
    normalized_gram: torch.Tensor
    factor: torch.Tensor
    normalized_factor: torch.Tensor
    head_norm: torch.Tensor
    q_to_kv: torch.Tensor

    def validate(self) -> "OperatorGeometry":
        layers, heads = self.layer_count, self.head_count
        square = (layers, heads, heads)
        if self.gram.shape != square or self.normalized_gram.shape != square:
            raise ValueError("operator Gram tensors must be [layer, head, head]")
        if self.factor.shape != square or self.normalized_factor.shape != square:
            raise ValueError("operator factors must be [layer, head, head]")
        if self.head_norm.shape != (layers, heads):
            raise ValueError("head norms must be [layer, head]")
        if self.q_to_kv.shape != (heads,):
            raise ValueError("query-to-KV mapping must be [head]")
        if not torch.isfinite(self.gram).all() or not torch.isfinite(
            self.normalized_gram
        ).all():
            raise ValueError("operator geometry contains non-finite values")
        if not torch.allclose(
            self.gram,
            self.gram.transpose(-1, -2),
            atol=2e-4,
            rtol=2e-4,
        ):
            raise ValueError("operator Gram matrix must be symmetric")
        if bool((self.head_norm < 0).any()):
            raise ValueError("operator head norms must be non-negative")
        return self

    def factor_for(self, mode: str, *, seed: int = 0) -> torch.Tensor:
        """Return a code embedding factor for one registered control.

        For row-vector code ``z``, ``z @ factor`` is the operator embedding.
        ``operator_permuted`` intentionally mismatches head indices and operator
        bases while preserving every Gram eigenvalue.
        """

        if mode == "identity":
            eye = torch.eye(self.head_count, dtype=self.gram.dtype)
            return eye.unsqueeze(0).expand(self.layer_count, -1, -1).clone()
        if mode == "operator_raw":
            return self.factor
        if mode == "operator_normalized":
            return self.normalized_factor
        if mode == "operator_permuted":
            blocks = []
            for layer in range(self.layer_count):
                generator = torch.Generator().manual_seed(int(seed) + layer * 104729)
                permutation = torch.randperm(self.head_count, generator=generator)
                blocks.append(self.normalized_factor[layer][permutation])
            return torch.stack(blocks)
        raise ValueError(
            "factor mode must be identity, operator_raw, "
            "operator_normalized, or operator_permuted"
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": OPERATOR_SCHEMA,
            "version": OPERATOR_VERSION,
            "model_path": self.model_path,
            "architecture": self.architecture,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "kv_head_count": self.kv_head_count,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "gram": self.gram.detach().cpu().float(),
            "normalized_gram": self.normalized_gram.detach().cpu().float(),
            "factor": self.factor.detach().cpu().float(),
            "normalized_factor": self.normalized_factor.detach().cpu().float(),
            "head_norm": self.head_norm.detach().cpu().float(),
            "q_to_kv": self.q_to_kv.detach().cpu().long(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "OperatorGeometry":
        if (
            payload.get("schema") != OPERATOR_SCHEMA
            or int(payload.get("version", -1)) != OPERATOR_VERSION
        ):
            raise ValueError("unsupported operator-geometry artifact")
        return cls(
            model_path=str(payload["model_path"]),
            architecture=str(payload["architecture"]),
            layer_count=int(payload["layer_count"]),
            head_count=int(payload["head_count"]),
            kv_head_count=int(payload["kv_head_count"]),
            head_dim=int(payload["head_dim"]),
            hidden_size=int(payload["hidden_size"]),
            gram=torch.as_tensor(payload["gram"]).float(),
            normalized_gram=torch.as_tensor(payload["normalized_gram"]).float(),
            factor=torch.as_tensor(payload["factor"]).float(),
            normalized_factor=torch.as_tensor(payload["normalized_factor"]).float(),
            head_norm=torch.as_tensor(payload["head_norm"]).float(),
            q_to_kv=torch.as_tensor(payload["q_to_kv"]).long(),
        ).validate()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_operator_geometry(path, geometry: OperatorGeometry) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as file:
        temporary = Path(file.name)
    torch.save(geometry.validate().payload(), temporary)
    temporary.replace(path)


def load_operator_geometry(path) -> OperatorGeometry:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("operator geometry must contain a dictionary payload")
    return OperatorGeometry.from_payload(payload)


def load_factorized_basis(path) -> dict[str, object]:
    """Load one optional per-layer ``W_O``/``W_V`` factor artifact."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "attention-hypernetwork-factorized-basis"
        or int(payload.get("version", -1)) != 1
    ):
        raise ValueError("unsupported factorized operator-basis artifact")
    output = torch.as_tensor(payload["output_factor"])
    value = torch.as_tensor(payload["value_factor"])
    q_to_kv = torch.as_tensor(payload["q_to_kv"]).long()
    if output.ndim != 3 or value.ndim != 3 or q_to_kv.shape != (output.shape[0],):
        raise ValueError("factorized operator basis has invalid dimensions")
    return {**payload, "output_factor": output, "value_factor": value, "q_to_kv": q_to_kv}


def apply_factorized_operator(
    code: torch.Tensor,
    source_state: torch.Tensor,
    basis: dict[str, object],
) -> torch.Tensor:
    """Apply ``sum_h z_h W^O_h W^V_{kv(h)} x`` from a cached factor basis.

    ``code`` is ``[..., H]`` and ``source_state`` is ``[..., D]`` with matching
    leading dimensions.  The function computes the linear value-path message;
    callers remain responsible for supplying the model's actual pre-attention
    normalized hidden state.  ``o_proj`` bias is deliberately not added because
    it belongs once to the complete attention output, not once per source pair.
    """

    output = torch.as_tensor(basis["output_factor"], device=code.device, dtype=code.dtype)
    value_unique = torch.as_tensor(
        basis["value_factor"], device=code.device, dtype=code.dtype
    )
    q_to_kv = torch.as_tensor(basis["q_to_kv"], device=code.device).long()
    if code.shape[-1] != output.shape[0] or source_state.shape[-1] != value_unique.shape[-1]:
        raise ValueError("code/source state does not match the cached operator basis")
    if code.shape[:-1] != source_state.shape[:-1]:
        raise ValueError("code and source state must have matching leading dimensions")
    value = value_unique[q_to_kv]
    projected = torch.einsum("hdi,...i->...hd", value, source_state)
    value_bias = basis.get("value_bias")
    if value_bias is not None:
        bias = torch.as_tensor(value_bias, device=code.device, dtype=code.dtype)
        bias = bias.reshape(value_unique.shape[0], value_unique.shape[1])[q_to_kv]
        projected = projected + bias
    return torch.einsum("hod,...hd->...o", output, projected * code[..., :, None])


def _gram_factor(gram: torch.Tensor) -> torch.Tensor:
    """Return ``F`` such that ``G ~= F F^T`` for a symmetric PSD matrix."""

    gram = (gram + gram.transpose(-1, -2)) * 0.5
    eigenvalue, eigenvector = torch.linalg.eigh(gram.double())
    tolerance = max(float(eigenvalue.max().item()), 1.0) * 1e-10
    eigenvalue = eigenvalue.clamp_min(0.0)
    eigenvalue = torch.where(
        eigenvalue >= tolerance,
        eigenvalue,
        torch.zeros_like(eigenvalue),
    )
    return (eigenvector * eigenvalue.sqrt().unsqueeze(0)).float()


def operator_gram_from_factors(
    output_factors: torch.Tensor,
    value_factors: torch.Tensor,
    *,
    block_heads: int = 4,
) -> torch.Tensor:
    """Compute exact Frobenius Gram without materializing ``B_h``.

    Args:
        output_factors: ``[H, D_out, d]`` matrices ``W^O_h``.
        value_factors: ``[H, d, D_in]`` matrices ``W^V_{kv(h)}`` after GQA
            expansion to query-head indexing.
    """

    if output_factors.ndim != 3 or value_factors.ndim != 3:
        raise ValueError("operator factors must be three-dimensional")
    heads, output_size, head_dim = output_factors.shape
    if value_factors.shape[0] != heads or value_factors.shape[1] != head_dim:
        raise ValueError("output/value head factors have incompatible dimensions")
    if block_heads < 1:
        raise ValueError("block_heads must be positive")

    result = torch.empty(
        (heads, heads),
        device=output_factors.device,
        dtype=torch.float32,
    )
    output_all = output_factors
    value_all = value_factors
    for start in range(0, heads, block_heads):
        stop = min(start + block_heads, heads)
        output_cross = torch.einsum(
            "boa,goc->bgac",
            output_all[start:stop],
            output_all,
        )
        value_cross = torch.einsum(
            "bai,gci->bgac",
            value_all[start:stop],
            value_all,
        )
        result[start:stop] = (output_cross * value_cross).sum(dim=(-1, -2)).float()
        del output_cross, value_cross
    return ((result + result.T) * 0.5).cpu()


def geometry_from_factors(
    output_factors_by_layer: Iterable[torch.Tensor],
    value_factors_by_layer: Iterable[torch.Tensor],
    *,
    model_path: str = "synthetic",
    architecture: str = "synthetic",
    kv_head_count: int | None = None,
    q_to_kv: torch.Tensor | None = None,
    block_heads: int = 4,
) -> OperatorGeometry:
    """Build a reusable geometry from already-split factors."""

    output_layers = list(output_factors_by_layer)
    value_layers = list(value_factors_by_layer)
    if not output_layers or len(output_layers) != len(value_layers):
        raise ValueError("output and value factors must contain the same non-zero layers")
    heads, hidden_size, head_dim = output_layers[0].shape
    grams = []
    for output, value in zip(output_layers, value_layers, strict=True):
        grams.append(
            operator_gram_from_factors(
                output,
                value,
                block_heads=block_heads,
            )
        )
    gram = torch.stack(grams).float()
    head_norm = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    denominator = head_norm[:, :, None] * head_norm[:, None, :]
    normalized = torch.where(
        denominator > 1e-12,
        gram / denominator.clamp_min(1e-12),
        torch.zeros_like(gram),
    )
    normalized = (normalized + normalized.transpose(-1, -2)) * 0.5
    factor = torch.stack([_gram_factor(layer) for layer in gram])
    normalized_factor = torch.stack([_gram_factor(layer) for layer in normalized])
    if q_to_kv is None:
        q_to_kv = torch.arange(heads)
    if kv_head_count is None:
        kv_head_count = int(torch.unique(q_to_kv).numel())
    return OperatorGeometry(
        model_path=model_path,
        architecture=architecture,
        layer_count=len(output_layers),
        head_count=heads,
        kv_head_count=kv_head_count,
        head_dim=head_dim,
        hidden_size=hidden_size,
        gram=gram,
        normalized_gram=normalized,
        factor=factor,
        normalized_factor=normalized_factor,
        head_norm=head_norm,
        q_to_kv=q_to_kv.cpu().long(),
    ).validate()


def _decoder_layers(model) -> list[torch.nn.Module]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return list(candidate)
    raise ValueError("cannot locate decoder layers on this causal language model")


def _self_attention(layer) -> torch.nn.Module:
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError("cannot locate self-attention module in decoder layer")


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    return mapping[name]


def extract_operator_geometry(
    model_path,
    *,
    device: str = "cpu",
    load_dtype: str = "bfloat16",
    compute_dtype: str = "float32",
    block_heads: int = 4,
    trust_remote_code: bool = False,
    basis_dir=None,
) -> OperatorGeometry:
    """Load a frozen model once and extract reusable per-layer operator geometry.

    ``basis_dir`` is optional.  When supplied, each layer additionally stores the
    factorized ``W^O_h`` and unique ``W^V_kv`` tensors.  These factors are enough
    to compute actual messages after supplying hidden states, and are much
    smaller than explicitly saving every ``D x D`` matrix ``B_h``.
    """

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "operator extraction requires transformers in the research environment"
        ) from error

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(load_dtype),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    config = model.config
    layers = _decoder_layers(model)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // heads))
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    q_to_kv = torch.div(
        torch.arange(heads),
        heads // kv_heads,
        rounding_mode="floor",
    )
    compute = _dtype(compute_dtype)
    if basis_dir is not None:
        basis_dir = Path(basis_dir)
        basis_dir.mkdir(parents=True, exist_ok=True)

    grams = []
    for layer_index, layer in enumerate(layers):
        attention = _self_attention(layer)
        if not hasattr(attention, "o_proj") or not hasattr(attention, "v_proj"):
            raise ValueError("attention module must expose o_proj and v_proj")
        output_weight = attention.o_proj.weight.detach()
        value_weight = attention.v_proj.weight.detach()
        expected_output = heads * head_dim
        expected_value = kv_heads * head_dim
        if output_weight.shape[1] != expected_output:
            raise ValueError("o_proj input dimension does not match attention heads")
        if value_weight.shape[0] != expected_value:
            raise ValueError("v_proj output dimension does not match KV heads")

        output = (
            output_weight.to(device=device, dtype=compute)
            .reshape(hidden_size, heads, head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )
        value_unique = value_weight.to(device=device, dtype=compute).reshape(
            kv_heads,
            head_dim,
            hidden_size,
        )
        value = value_unique[q_to_kv.to(device)].contiguous()
        gram = operator_gram_from_factors(
            output,
            value,
            block_heads=block_heads,
        )
        grams.append(gram)

        if basis_dir is not None:
            payload = {
                "schema": "attention-hypernetwork-factorized-basis",
                "version": 1,
                "layer": layer_index,
                "model_path": str(model_path),
                "output_factor": output.detach().cpu().to(torch.float16),
                "value_factor": value_unique.detach().cpu().to(torch.float16),
                "head_count": heads,
                "kv_head_count": kv_heads,
                "head_dim": head_dim,
                "hidden_size": hidden_size,
                "q_to_kv": q_to_kv,
                "value_bias": (
                    attention.v_proj.bias.detach().cpu().to(torch.float16)
                    if attention.v_proj.bias is not None
                    else None
                ),
                "output_bias": (
                    attention.o_proj.bias.detach().cpu().to(torch.float16)
                    if attention.o_proj.bias is not None
                    else None
                ),
            }
            torch.save(payload, basis_dir / f"layer_{layer_index:03d}.pt")
        del output, value, value_unique, output_weight, value_weight
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    gram = torch.stack(grams).float()
    head_norm = gram.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    denominator = head_norm[:, :, None] * head_norm[:, None, :]
    normalized = torch.where(
        denominator > 1e-12,
        gram / denominator.clamp_min(1e-12),
        torch.zeros_like(gram),
    )
    normalized = (normalized + normalized.transpose(-1, -2)) * 0.5
    factor = torch.stack([_gram_factor(layer) for layer in gram])
    normalized_factor = torch.stack([_gram_factor(layer) for layer in normalized])
    architecture = (
        config.architectures[0]
        if getattr(config, "architectures", None)
        else model.__class__.__name__
    )
    geometry = OperatorGeometry(
        model_path=str(model_path),
        architecture=str(architecture),
        layer_count=len(layers),
        head_count=heads,
        kv_head_count=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        gram=gram,
        normalized_gram=normalized,
        factor=factor,
        normalized_factor=normalized_factor,
        head_norm=head_norm,
        q_to_kv=q_to_kv,
    ).validate()
    del model
    return geometry
