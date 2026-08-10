"""Extract the six canonical attention arrays directly from a causal LM."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import AttentionSample, save_attention_sample
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
    def __init__(self, layers, heads, tokens, response_idx, floor):
        self.layers, self.heads, self.tokens = layers, heads, tokens
        self.response_idx, self.floor = response_idx, floor
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
    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def run(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            device_map={"": self.config.device},
        ).eval()
        samples = load_ragtruth_samples(
            self.config.dataset_path,
            split=self.config.split,
            generator_model=self.config.generator_model,
            task_type=self.config.task_type,
        )
        if self.config.limit is not None:
            samples = samples[: self.config.limit]

        output = Path(self.config.output_dir)
        attention_dir = output / "attention"
        attention_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        L, H = len(model.model.layers), int(model.config.num_attention_heads)

        for item in tqdm(samples, desc=f"extract {self.config.split}"):
            token_ids, response_idx = tokenize_ragtruth_sample(
                tokenizer, prompt=item.prompt, response=item.response
            )
            collector = AttentionCollector(
                L, H, len(token_ids), response_idx, self.config.floor
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
                    model.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_attentions=True,
                        use_cache=False,
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
            relative = Path("attention") / f"{item.response_id}.npz"
            save_attention_sample(sample, output / relative)
            rows.append({
                "sample_id": item.response_id,
                "source_id": item.source_id,
                "path": relative.as_posix(),
            })

        manifest = {
            "attention_floor": self.config.floor,
            "num_layers": L,
            "num_heads": H,
            "count": len(rows),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (output / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return manifest
