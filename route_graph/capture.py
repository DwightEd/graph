"""Frozen local Llama replay: real causal edges to fixed route features."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from route_graph.data import read_responses, write_jsonl
from route_graph.operator import PathEncoder


@dataclass(frozen=True)
class CaptureConfig:
    input_path: Path
    output_dir: Path
    model_path: Path
    end_layers: tuple[int, ...] = ()  # Empty selects the final layer only.
    depths: tuple[int, ...] = (1, 2)
    candidates: int = 2
    device: str = "cpu"
    dtype: str = "float32"
    max_tokens: int = 2048
    max_attention_mb: int = 1024


class FrozenGraphCapture:
    """Teacher-forced causal replay; features for y_t use its predictor only.

    The checkpoint is a probe, not necessarily the original response generator.
    The raw RAGTruth prompt is tokenized with BOS; response tokens are appended
    separately. No chat-template or original-generator equivalence is claimed.
    """

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config

    def run(self) -> dict:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = self.config
        records = read_responses(config.input_path)
        encoder = PathEncoder(config.depths)
        if (
            config.candidates < 2
            or config.max_tokens < 1
            or config.max_attention_mb < 1
        ):
            raise ValueError(
                "candidates >= 2 and positive context/memory limits required"
            )
        if config.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if config.output_dir.exists():
            raise FileExistsError(config.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, local_files_only=True
        )
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required for character offsets")
        model = (
            AutoModelForCausalLM.from_pretrained(
                config.model_path,
                local_files_only=True,
                torch_dtype=getattr(torch, config.dtype),
                attn_implementation="eager",
            )
            .to(config.device)
            .eval()
            .requires_grad_(False)
        )
        if model.config.model_type != "llama":
            raise ValueError(
                "native candidate projection currently supports Llama only"
            )
        layers = config.end_layers or (model.config.num_hidden_layers - 1,)
        if (
            min(layers) < max(config.depths) - 1
            or max(layers) >= model.config.num_hidden_layers
            or tuple(sorted(set(layers))) != layers
        ):
            raise ValueError("end_layers must be ordered and have sufficient ancestry")
        if config.candidates > model.config.vocab_size:
            raise ValueError("candidates exceed vocabulary size")
        first_layer, last_layer = min(layers) - max(config.depths) + 1, max(layers) + 1
        config.output_dir.mkdir(parents=True)
        count, names = 0, None
        feature_path = config.output_dir / "features.jsonl"

        def features():
            nonlocal count, names
            with (
                torch.inference_mode(),
                tqdm(records, desc="capture responses", unit="response") as samples,
            ):
                for record in samples:
                    samples.set_postfix(sample=record["id"], phase="tokenize")
                    prompt = tokenizer(
                        record["prompt"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    response = tokenizer(
                        record["response"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    bos = (
                        []
                        if tokenizer.bos_token_id is None
                        else [tokenizer.bos_token_id]
                    )
                    prompt_ids = bos + prompt["input_ids"]
                    response_ids = response["input_ids"]
                    if not prompt_ids or not response_ids:
                        raise ValueError(
                            f"empty tokenized prompt/response: {record['id']}"
                        )
                    ids = prompt_ids + response_ids[:-1]
                    length = len(ids)
                    if length > min(
                        config.max_tokens, model.config.max_position_embeddings
                    ):
                        raise ValueError(
                            f"context limit exceeded by response {record['id']}: {length}"
                        )
                    # Conservative float32 attention storage estimate, excluding weights,
                    # logits, activations and temporary workspace. This is not an OOM guarantee.
                    attention_mb = (
                        model.config.num_hidden_layers
                        * model.config.num_attention_heads
                        * length**2
                        * 4
                        / 2**20
                    )
                    if attention_mb > config.max_attention_mb:
                        raise ValueError(
                            f"attention storage estimate {attention_mb:.1f} MiB exceeds limit"
                        )
                    roles = [3] * len(bos)
                    source_units = [-1] * len(bos)
                    start, end = record["source_span"]
                    # Blank paragraphs define structural source units,
                    # using only the original source text (no answer annotations).
                    boundaries = [start] + [
                        start + match.end()
                        for match in re.finditer(
                            r"\n\s*\n", record["prompt"][start:end]
                        )
                    ]
                    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
                    for token_id, (left, right) in zip(
                        prompt["input_ids"], prompt["offset_mapping"], strict=True
                    ):
                        roles.append(
                            3
                            if token_id in special
                            else int(not (left < end and right > start))
                        )
                        source_units.append(
                            max(
                                0,
                                int(np.searchsorted(boundaries, left, side="right"))
                                - 1,
                            )
                            if roles[-1] == 0
                            else -1
                        )
                    roles.extend(
                        3 if token in special else 2 for token in response_ids[:-1]
                    )
                    source_units.extend([-1] * (len(response_ids) - 1))
                    samples.set_postfix(
                        sample=record["id"], phase="forward", tokens=len(ids)
                    )
                    outputs = model(
                        torch.tensor([ids], device=config.device),
                        use_cache=False,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    attention = np.stack(
                        [
                            a[0].float().cpu().numpy()
                            for a in outputs.attentions[first_layer:last_layer]
                        ]
                    )
                    attention /= attention.sum(-1, keepdims=True)
                    states = torch.stack(
                        [
                            model.model.norm(h[0]).float()
                            for h in outputs.hidden_states[first_layer:last_layer]
                        ]
                    )
                    all_logits = outputs.logits[0]
                    del outputs
                    states = torch.nn.functional.normalize(states, dim=-1)
                    output_weights = model.get_output_embeddings().weight
                    for token_index, (left, right) in tqdm(
                        enumerate(response["offset_mapping"]),
                        total=len(response_ids),
                        desc="route tokens",
                        unit="token",
                        leave=False,
                    ):
                        query = len(prompt_ids) + token_index - 1
                        logits = all_logits[query].float()
                        top = torch.topk(logits, config.candidates)
                        directions = torch.nn.functional.normalize(
                            output_weights[top.indices].float(), dim=-1
                        )
                        contrasts = directions[0:1] - directions[1:]
                        signals = (states[:, : query + 1] @ contrasts.T).cpu().numpy()
                        encoded = encoder.encode(
                            attention[:, :, : query + 1, : query + 1],
                            signals,
                            np.asarray(roles[: query + 1]),
                            query=query,
                            end_layers=tuple(layer - first_layer for layer in layers),
                            source_units=np.asarray(source_units[: query + 1]),
                            layer_offset=first_layer,
                        )
                        names = encoded.pop("names")
                        probabilities = logits.softmax(-1)
                        entropy = -(probabilities * logits.log_softmax(-1)).sum().item()
                        count += 1
                        yield {
                            "schema": "route-graph/features@1",
                            "response_id": record["id"],
                            "source_id": record["source_id"],
                            "split": record["split"],
                            "task": record["task"],
                            "generator": record["generator"],
                            "response_sha256": record["response_sha256"],
                            "token_index": token_index,
                            "token_count": len(response_ids),
                            "char_span": [left, right],
                            "token_id": response_ids[token_index],
                            "prompt_tokens": len(prompt_ids),
                            "predictor_index": query,
                            "candidate_ids": top.indices.tolist(),
                            "entropy": entropy,
                            "negative_margin": -(top.values[0] - top.values[1]).item(),
                            **encoded,
                        }
                    del all_logits, attention, states

        write_jsonl(feature_path, features())
        # Written last: a partial extraction is never accepted by the detector.
        manifest = {
            "schema": "route-graph/capture@1",
            "tokens": count,
            "responses": len(records),
            "labels_used": False,
            "model_type": model.config.model_type,
            "probe_model": str(config.model_path.resolve()),
            "end_layers": list(layers),
            "feature_names": names,
            "config": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(config).items()
            },
            "prompt_protocol": "raw-bos-segmented-response@1",
            "null_model": "role_source-unit_log-lag_self@1",
            "attention_numeric_normalization": "float32 row renormalization after capture",
            "features_sha256": file_digest(feature_path),
            "checkpoint_sha256": {
                path.name: file_digest(path)
                for path in tqdm(
                    [
                        path
                        for path in sorted(config.model_path.iterdir())
                        if path.suffix in {".json", ".safetensors", ".bin", ".model"}
                    ],
                    desc="record checkpoint",
                    unit="file",
                )
            },
        }
        (config.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "tokens": count,
            "responses": len(records),
            "output": str(config.output_dir),
        }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
