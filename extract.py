"""Extract compact, label-free attention artifacts from a causal language model."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import AttentionSample, save_attention_sample
from ragtruth import load_ragtruth_samples, tokenize_ragtruth_sample


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class ExtractionConfig:
    model_path: str
    dataset_path: str | Path
    output_dir: str | Path
    split: str
    generator_model: str = "llama-2-7b-chat"
    task_type: str = "all"
    floor: float = 0.01
    dtype: str = "float16"
    device: str = "cuda"
    limit: int | None = None


class AttentionCollector:
    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        num_tokens: int,
        response_idx: int,
        floor: float,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.response_idx = response_idx
        self.floor = floor
        self.dtype = dtype
        self.diagonals: list[torch.Tensor | None] = [None] * num_layers
        self.row_counts: list[torch.Tensor] = []
        self.column_indices: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def consume(self, layer_index: int, attention: torch.Tensor) -> None:
        if attention.ndim != 3 or tuple(attention.shape[1:]) != (self.num_tokens, self.num_tokens):
            raise ValueError("layer attention must have shape [heads, tokens, tokens]")
        if attention.shape[0] != self.num_heads:
            raise ValueError("layer attention head count changed")
        self.diagonals[layer_index] = attention.diagonal(dim1=-2, dim2=-1).to(
            dtype=self.dtype,
            device="cpu",
        )
        response_attention = attention[:, self.response_idx :, :]
        source_indices = torch.arange(self.num_tokens, device=attention.device)
        target_indices = torch.arange(self.response_idx, self.num_tokens, device=attention.device)
        mask = (response_attention.to(torch.float32) > self.floor) & (
            source_indices[None, None, :] < target_indices[None, :, None]
        )
        flattened_mask = mask.reshape(-1, self.num_tokens)
        coordinates = torch.nonzero(flattened_mask, as_tuple=False)
        self.row_counts.append(flattened_mask.sum(dim=1, dtype=torch.int64).cpu())
        self.column_indices.append(coordinates[:, 1].to(dtype=torch.int32, device="cpu"))
        self.values.append(response_attention[mask].to(dtype=self.dtype, device="cpu"))

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(value is None for value in self.diagonals):
            raise ValueError("not all layer attentions were collected")
        row_counts = torch.cat(self.row_counts)
        return (
            torch.stack([value for value in self.diagonals if value is not None]),
            torch.cat([torch.zeros(1, dtype=torch.int64), row_counts.cumsum(dim=0)]),
            torch.cat(self.column_indices)
            if self.column_indices
            else torch.empty(0, dtype=torch.int32),
            torch.cat(self.values) if self.values else torch.empty(0, dtype=self.dtype),
        )


class AttentionExtractor:
    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def run(self) -> None:
        if self.config.dtype not in DTYPES:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if not math.isfinite(self.config.floor) or not 0 < self.config.floor <= 1:
            raise ValueError("floor must be finite and in (0, 1]")
        if self.config.limit is not None and (
            type(self.config.limit) is not int or self.config.limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        dtype = DTYPES[self.config.dtype]
        samples = load_ragtruth_samples(
            self.config.dataset_path,
            split=self.config.split,
            generator_model=self.config.generator_model,
            task_type=self.config.task_type,
        )
        if self.config.limit is not None:
            samples = samples[:self.config.limit]
        if not samples:
            raise ValueError("no matching samples found")
        output_dir = Path(self.config.output_dir)
        if output_dir.exists():
            if not output_dir.is_dir() or any(output_dir.iterdir()):
                raise FileExistsError("output directory must be empty")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map={"": self.config.device},
        ).eval()
        output_dir.mkdir(parents=True, exist_ok=True)
        index_rows: list[dict[str, str]] = []
        layer_count = len(model.model.layers)
        head_count: int | None = None
        for sample in tqdm(samples, desc=f"RAGTruth {self.config.split}"):
            token_ids, response_idx = tokenize_ragtruth_sample(
                tokenizer,
                prompt=sample.prompt,
                response=sample.response,
            )
            context_limit = model.config.max_position_embeddings
            if context_limit is not None and token_ids.numel() > int(context_limit):
                raise ValueError("full context length exceeds model context limit")
            collector: AttentionCollector | None = None

            def collect_attention(
                _: torch.nn.Module,
                __: tuple[Any, ...],
                output: tuple[Any, ...],
                layer_index: int,
            ) -> tuple[Any, ...]:
                nonlocal collector, head_count
                attention = output[1]
                if attention.ndim != 4 or attention.shape[0] != 1:
                    raise ValueError(
                        "Llama decoder layer must return "
                        "[batch, heads, tokens, tokens] attention"
                    )
                layer_attention = attention[0]
                if collector is None:
                    head_count = int(layer_attention.shape[0])
                    collector = AttentionCollector(
                        num_layers=layer_count,
                        num_heads=head_count,
                        num_tokens=token_ids.numel(),
                        response_idx=response_idx,
                        floor=self.config.floor,
                        dtype=dtype,
                    )
                collector.consume(layer_index, layer_attention)
                compact = list(output)
                compact[1] = None
                return tuple(compact)

            hooks = [
                layer.self_attn.register_forward_hook(
                    lambda module, args, output, index=index: collect_attention(
                        module,
                        args,
                        output,
                        index,
                    )
                )
                for index, layer in enumerate(model.model.layers)
            ]
            try:
                input_ids = token_ids.unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    model.model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        return_dict=True,
                        use_cache=False,
                        output_attentions=True,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
            if collector is None:
                raise ValueError("model produced no layer attentions")
            diagonal, row_ptr, columns, values = collector.finalize()
            artifact = AttentionSample(
                sample.response_id,
                sample.source_id,
                response_idx,
                token_ids,
                diagonal,
                row_ptr,
                columns,
                values,
                self.config.floor,
            )
            filename = f"{sample.response_id}.pt"
            save_attention_sample(artifact, output_dir / filename)
            index_rows.append(
                {
                    "sample_id": sample.response_id,
                    "source_id": sample.source_id,
                    "path": filename,
                }
            )
        manifest = {
            "schema": "attention-response-csr-v1",
            "observer_model": self.config.model_path,
            "tokenizer": getattr(tokenizer, "name_or_path", self.config.model_path),
            "num_layers": layer_count,
            "num_heads": head_count,
            "dtype": self.config.dtype,
            "floor": self.config.floor,
            "channel_order": "layer,head",
            "split": self.config.split,
            "generator_model": self.config.generator_model,
            "task_type": self.config.task_type,
            "input_policy": "full_context_no_truncation",
            "count": len(index_rows),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
            for row in index_rows:
                handle.write(json.dumps(row) + "\n")
