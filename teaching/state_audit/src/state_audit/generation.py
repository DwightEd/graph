"""Create answers and exact replay records; annotations never move to a new answer."""

import hashlib
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from transformers import GenerationConfig, set_seed

from .datasets import load_examples
from .models import input_tensor
from .storage import start_stage, write_json
from .tokenization import answer_record, encode_prompt, generated_offsets


def make_answer(model, tokenizer, example, options: dict, seed: int) -> dict:
    prompt = encode_prompt(tokenizer, example, options["template"])
    if options["mode"] == "replay":
        if example.response is None:
            raise ValueError(f"{example.id}: replay requires a response")
        encoded = tokenizer(example.response, add_special_tokens=False, return_offsets_mapping=True)
        ids, text, offsets = encoded["input_ids"], example.response, encoded["offset_mapping"]
    else:
        ids = sample_answer(model, tokenizer, prompt["prompt_ids"], options, seed)
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        offsets = generated_offsets(tokenizer, ids, text)
    if len(prompt["prompt_ids"]) + len(ids) > options["max_length"]:
        raise ValueError(f"{example.id}: exceeds --max-length; no silent truncation")
    return answer_record(tokenizer, example, prompt, ids, text, offsets, options["mode"], seed)


def sample_answer(model, tokenizer, prompt_ids: list[int], options: dict, seed: int) -> list[int]:
    available = options["max_length"] - len(prompt_ids)
    if available < options["max_new_tokens"]:
        raise ValueError("Prompt + max_new_tokens exceeds --max-length; raise it explicitly")
    set_seed(seed)
    settings = dict(
        max_new_tokens=options["max_new_tokens"],
        do_sample=options["temperature"] > 0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ),
    )
    if settings["do_sample"]:
        settings.update(temperature=options["temperature"], top_p=options["top_p"])
    ids = input_tensor(model, prompt_ids)
    with torch.inference_mode():
        output = model.generate(
            ids, attention_mask=torch.ones_like(ids), generation_config=GenerationConfig(**settings)
        )
    return output[0, len(prompt_ids) :].tolist()


def generate_run(
    model,
    tokenizer,
    data: Path,
    root: Path,
    options: dict,
    model_settings: dict,
    resume: bool = False,
) -> dict:
    if options["temperature"] < 0 or not 0 < options["top_p"] <= 1:
        raise ValueError("temperature >= 0 and top_p in (0, 1] are required")
    if min(options["max_new_tokens"], options["max_length"]) < 1:
        raise ValueError("Token limits must be positive")
    examples = load_examples(data)
    manifest = dict(
        schema_version=1,
        model=model_settings,
        generation=options,
        model_config=model.config.to_dict(),
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        samples=[
            dict(index=i, id=row.id, source_id=row.source_id) for i, row in enumerate(examples)
        ],
    )
    start_stage(root / "run.json", manifest, resume)
    for index, example in enumerate(tqdm(examples, desc="answers")):
        path = root / "samples" / f"{index:06d}" / "answer.json"
        if not (resume and path.exists()):
            record = make_answer(model, tokenizer, example, options, options["seed"] + index)
            write_json(path, record)
    return manifest
