import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import save_feature


SYSTEM_PROMPT = "You are a helpful assistant."
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _name(value):
    return "".join(c for c in value.casefold() if c.isalnum())


def load_ragtruth(dataset_dir, split, generator_model, task="all"):
    dataset_dir = Path(dataset_dir)
    sources = {str(x["source_id"]): x for x in _read_jsonl(dataset_dir / "source_info.jsonl")}
    requested = _name(generator_model)

    rows = []
    for response in _read_jsonl(dataset_dir / "response.jsonl"):
        if response["split"] != split or _name(response["model"]) != requested:
            continue
        if str(response.get("quality", "")).casefold() != "good":
            continue
        source = sources[str(response["source_id"])]
        if task != "all" and source["task_type"].casefold() != task.casefold():
            continue
        rows.append((response, source))
    return rows


def encode_example(tokenizer, prompt, response):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = rendered + response
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    boundary = len(rendered)
    response_idx = next(i for i, (start, _) in enumerate(encoded["offset_mapping"]) if start >= boundary)
    return torch.tensor(encoded["input_ids"], dtype=torch.long), response_idx, encoded["offset_mapping"], boundary


def token_labels(offsets, boundary, labels):
    y = torch.zeros(len(offsets), dtype=torch.long)
    for span in labels:
        start = boundary + int(span["start"])
        end = boundary + int(span["end"])
        for i, (left, right) in enumerate(offsets):
            if right > start and left < end:
                y[i] = 1
    return y


def compress_attention(attentions, response_idx, floor, dtype):
    """Dense model attention -> response-only CSR + diagonal."""
    diagonal = []
    row_counts = []
    sources = []
    weights = []

    for attention in attentions:
        a = attention[0]
        heads, tokens, _ = a.shape
        diagonal.append(a.diagonal(dim1=-2, dim2=-1).to(dtype=dtype, device="cpu"))

        response = a[:, response_idx:, :]
        target = torch.arange(response_idx, tokens, device=a.device)
        source = torch.arange(tokens, device=a.device)
        keep = (response.float() > floor) & (source[None, None, :] < target[None, :, None])
        flat = keep.reshape(-1, tokens)
        where = torch.nonzero(flat, as_tuple=False)
        row_counts.append(flat.sum(1, dtype=torch.long).cpu())
        sources.append(where[:, 1].to(torch.int32).cpu())
        weights.append(response[keep].to(dtype=dtype, device="cpu"))

    counts = torch.cat(row_counts)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return (
        torch.stack(diagonal),
        row_ptr,
        torch.cat(sources),
        torch.cat(weights),
    )


def extract_ragtruth(
    model_path,
    dataset_dir,
    output_dir,
    split,
    generator_model="llama-2-7b-chat",
    task="all",
    floor=0.01,
    dtype="float16",
    device="cuda",
    hidden_layers=(),
    limit=None,
):
    """Extract minimal graph-building features and separate evaluation labels."""
    output_dir = Path(output_dir)
    feature_dir = output_dir / "features" / split
    label_dir = output_dir / "labels" / split
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    storage_dtype = DTYPES[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=storage_dtype,
        attn_implementation="eager",
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()

    rows = load_ragtruth(dataset_dir, split, generator_model, task)
    if limit is not None:
        rows = rows[:limit]

    for response, source in tqdm(rows, desc=f"extract {split}"):
        token_ids, response_idx, offsets, boundary = encode_example(
            tokenizer, source["prompt"], response["response"]
        )
        input_ids = token_ids.unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_attentions=True,
                output_hidden_states=bool(hidden_layers),
                use_cache=False,
                return_dict=True,
            )

        diagonal, row_ptr, source_index, attention_weight = compress_attention(
            outputs.attentions, response_idx, floor, storage_dtype
        )

        sample = {
            "sample_id": str(response["id"]),
            "source_id": str(response["source_id"]),
            "response_idx": response_idx,
            "token_ids": token_ids,
            "attention_diagonal": diagonal,
            "row_ptr": row_ptr,
            "source_index": source_index,
            "attention_weight": attention_weight,
            "attention_floor": float(floor),
        }

        if hidden_layers:
            model_layers = len(outputs.hidden_states) - 1
            resolved = [model_layers - 1 if layer == -1 else layer for layer in hidden_layers]
            sample["hidden_layers"] = torch.tensor(resolved, dtype=torch.long)
            sample["hidden_states"] = torch.stack([
                outputs.hidden_states[layer + 1][0].to(dtype=storage_dtype, device="cpu")
                for layer in resolved
            ])

        save_feature(sample, feature_dir / f"{response['id']}.pt")

        y_token = token_labels(offsets, boundary, response.get("labels", []))
        torch.save({
            "sample_id": str(response["id"]),
            "task": str(source["task_type"]),
            "y_token": y_token,
            "response_label": int(y_token[response_idx:].any()),
        }, label_dir / f"{response['id']}.pt")

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
