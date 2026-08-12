"""Extract the canonical attention representation used by the graph method."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import AttentionSample, index_row, save_attention_sample, sha256, write_split_index
from ragtruth import load_ragtruth_samples, tokenize_ragtruth_sample


@dataclass(frozen=True)
class ExtractionConfig:
    model_path: str
    dataset_path: str | Path
    output_dir: str | Path
    split: str
    generator_model: str = "llama-2-7b-chat"
    task_type: str = "all"
    floor: float = 0.01
    device: str = "cuda"
    limit: int | None = None


class AttentionCollector:
    """Stream full attention outputs into diagonal + sparse response CSR."""

    def __init__(self, layers, heads, tokens, response_idx, floor):
        self.layers, self.heads, self.tokens = layers, heads, tokens
        self.response_idx, self.floor = response_idx, float(floor)
        self.diagonal = [None] * layers
        self.counts, self.columns, self.values = [], [], []

    def consume(self, layer, attention):
        self.diagonal[layer] = attention.diagonal(dim1=-2, dim2=-1).half().cpu()
        response = attention[:, self.response_idx :, :]
        source = torch.arange(self.tokens, device=attention.device)
        target = torch.arange(self.response_idx, self.tokens, device=attention.device)
        mask = (response.float() > self.floor) & (
            source[None, None, :] < target[None, :, None]
        )
        flat = mask.reshape(-1, self.tokens)
        nz = torch.nonzero(flat, as_tuple=False)
        self.counts.append(flat.sum(1, dtype=torch.int64).cpu())
        self.columns.append(nz[:, 1].to(torch.int32).cpu())
        self.values.append(response[mask].half().cpu())

    def finish(self):
        counts = torch.cat(self.counts)
        return (
            torch.stack(self.diagonal),
            torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))),
            torch.cat(self.columns),
            torch.cat(self.values),
        )


class AttentionExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config

    def run(self):
        if not 0.0 < self.config.floor <= 1.0:
            raise ValueError("floor must be in (0,1]")
        if self.config.limit is not None and self.config.limit < 1:
            raise ValueError("limit must be positive")
        output = Path(self.config.output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("output_dir must be empty")
        samples = load_ragtruth_samples(
            self.config.dataset_path,
            split=self.config.split,
            generator_model=self.config.generator_model,
            task_type=self.config.task_type,
        )
        if self.config.limit is not None:
            samples = samples[: self.config.limit]
        if not samples:
            raise ValueError("no RAGTruth samples matched extraction filters")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map={"": self.config.device},
        ).eval()
        (output / "attention").mkdir(parents=True, exist_ok=True)
        layers = len(model.model.layers)
        heads = int(model.config.num_attention_heads)
        rows, label_rows = [], []

        for item in tqdm(samples, desc=f"extract {self.config.split}"):
            token_ids, response_idx, positive_runs = tokenize_ragtruth_sample(
                tokenizer,
                prompt=item.prompt,
                response=item.response,
                positive_char_spans=item.positive_char_spans,
            )
            collector = AttentionCollector(
                layers, heads, len(token_ids), response_idx, self.config.floor
            )
            hooks = []
            for layer_id, layer in enumerate(model.model.layers):
                def hook(_module, _args, result, layer_id=layer_id):
                    collector.consume(layer_id, result[1][0])
                    result = list(result)
                    result[1] = None
                    return tuple(result)
                hooks.append(layer.self_attn.register_forward_hook(hook))
            try:
                ids = token_ids.unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_attentions=True,
                        use_cache=False,
                        return_dict=True,
                    )
            finally:
                for hook in hooks:
                    hook.remove()

            diagonal, row_ptr, columns, values = collector.finish()
            sample = AttentionSample(
                item.response_id,
                item.source_id,
                response_idx,
                token_ids,
                diagonal,
                row_ptr,
                columns,
                values,
                self.config.floor,
            )
            path = output / "attention" / f"{item.response_id}.npz"
            save_attention_sample(sample, path)
            rows.append(index_row(output, sample, path, metadata={
                "split": item.split,
                "task_type": item.task_type,
                "data_source": item.data_source,
                "generator_model": item.generator_model,
                "temperature": item.temperature,
                "quality": item.quality,
            }))
            label_rows.append({"sample_id": item.response_id, "positive_runs": positive_runs})

        labels_path = output / "labels.jsonl"
        labels_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
            encoding="utf-8",
        )
        return write_split_index(
            output,
            rows,
            attention_floor=self.config.floor,
            num_layers=layers,
            num_heads=heads,
            alignment="post_token_query_at_same_position",
            extra={
                "dataset": "RAGTruth",
                "split": self.config.split,
                "observer_model": Path(self.config.model_path).name,
                "generator_model": self.config.generator_model,
                "labels_sha256": sha256(labels_path),
            },
        )
