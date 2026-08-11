"""Extract canonical attention, hidden-state and token-stat features in one pass."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import AttentionSample, index_row, save_attention_sample, write_split_index
from features import save_hidden_features, save_token_stats, teacher_forced_stats
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
    hidden_layers: tuple[int, ...] = ()
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
        if not 0 < self.config.floor <= 1:
            raise ValueError("floor must be in (0,1]")
        if self.config.limit is not None and (isinstance(self.config.limit, bool) or not isinstance(self.config.limit, int) or self.config.limit < 1):
            raise ValueError("limit must be a positive integer")
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
        if self.config.hidden_layers:
            (output / "hidden").mkdir(exist_ok=True)
        (output / "token_stats").mkdir(exist_ok=True)

        rows = []
        L, H = len(model.model.layers), int(model.config.num_attention_heads)
        hidden_layers = tuple(sorted(set(self.config.hidden_layers)))
        if any(layer < 0 or layer >= L for layer in hidden_layers):
            raise ValueError(f"hidden layers must be in [0,{L - 1}]")

        for item in tqdm(samples, desc=f"extract {self.config.split}"):
            token_ids, response_idx = tokenize_ragtruth_sample(
                tokenizer, prompt=item.prompt, response=item.response
            )
            collector = AttentionCollector(L, H, len(token_ids), response_idx, self.config.floor)
            hidden = {}
            hooks = []

            for layer_id, layer in enumerate(model.model.layers):
                def attention_hook(_module, _args, result, layer_id=layer_id):
                    collector.consume(layer_id, result[1][0])
                    result = list(result)
                    result[1] = None
                    return tuple(result)

                hooks.append(layer.self_attn.register_forward_hook(attention_hook))
                if layer_id in hidden_layers:
                    def hidden_hook(_module, _args, result, layer_id=layer_id):
                        hidden[layer_id] = result[0][0].detach().half().cpu()
                    hooks.append(layer.register_forward_hook(hidden_hook))

            try:
                ids = token_ids.unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    result = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_attentions=True,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                token_log_prob, entropy = teacher_forced_stats(result.logits, ids[0])
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
            attention_path = output / "attention" / f"{item.response_id}.npz"
            save_attention_sample(sample, attention_path)
            save_token_stats(
                output / "token_stats" / f"{item.response_id}.npz",
                token_ids,
                token_log_prob,
                entropy,
            )
            if hidden_layers:
                save_hidden_features(
                    output / "hidden" / f"{item.response_id}.npz",
                    token_ids,
                    hidden_layers,
                    torch.stack([hidden[layer] for layer in hidden_layers]),
                )
            rows.append(index_row(output, sample, attention_path, metadata={
                "split": item.split,
                "task_type": item.task_type,
                "data_source": item.data_source,
                "generator_model": item.generator_model,
                "temperature": item.temperature,
                "quality": item.quality,
            }))
            del result, token_log_prob, entropy

        return write_split_index(
            output, rows, attention_floor=self.config.floor, num_layers=L, num_heads=H,
            alignment="post_token_query_at_same_position", extra={
                "dataset": "RAGTruth",
                "split": self.config.split,
                "hidden_layers": list(hidden_layers),
                "observer_model": Path(self.config.model_path).name,
                "generator_model": self.config.generator_model,
            },
        )
