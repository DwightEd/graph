"""Same input × sampling seed, with immutable token IDs and answer lineage."""

import hashlib
from dataclasses import asdict, dataclass

import torch
import transformers
from tqdm import tqdm
from transformers import GenerationConfig, set_seed

from .dataset import load_examples
from .intervention import intervene
from .storage import start_stage, write_json
from .tokenization import answer_record, encode_prompt, generated_offsets


@dataclass(frozen=True)
class GenerationOptions:
    mode: str = "generate"
    template: str = "chat"
    seed: int = 0
    samples: int = 1
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 128
    max_length: int = 1024

    def __post_init__(self):
        if self.samples < 1 or (self.mode == "replay" and self.samples != 1):
            raise ValueError("samples must be positive; replay uses exactly one saved response")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature >= 0 and top_p in (0, 1] are required")


def make_answer(model, tokenizer, example, options: GenerationOptions, seed: int) -> dict:
    prompt = encode_prompt(tokenizer, example, options.template)
    if options.mode == "replay":
        if example.response is None:
            raise ValueError(f"{example.id}: replay requires a response")
        encoded = tokenizer(example.response, add_special_tokens=False, return_offsets_mapping=True)
        ids, text, offsets = encoded["input_ids"], example.response, encoded["offset_mapping"]
        if len(prompt["prompt_ids"]) + len(ids) > options.max_length:
            raise ValueError(f"{example.id}: exceeds max_length; no silent truncation")
    else:
        ids = sample_answer(model, tokenizer, prompt["prompt_ids"], options, seed)
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        offsets = generated_offsets(tokenizer, ids, text)
    return answer_record(tokenizer, example, prompt, ids, text, offsets, options.mode, seed)


def sample_answer(model, tokenizer, prompt_ids, options, seed, operations=()) -> list[int]:
    if len(prompt_ids) + options.max_new_tokens > options.max_length:
        raise ValueError("Prompt + max_new_tokens exceeds max_length; raise it explicitly")
    set_seed(seed)
    settings = dict(
        max_new_tokens=options.max_new_tokens,
        do_sample=options.temperature > 0,
        eos_token_id=model.native.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Full-prefix execution keeps absolute operation positions stable during generation.
        use_cache=not bool(operations),
    )
    if settings["pad_token_id"] is None:
        settings["pad_token_id"] = tokenizer.eos_token_id
    if settings["do_sample"]:
        settings.update(temperature=options.temperature, top_p=options.top_p)
    ids = model.input_ids(prompt_ids)
    with torch.inference_mode(), intervene(model, operations):
        output = model.native.generate(
            ids, attention_mask=torch.ones_like(ids), generation_config=GenerationConfig(**settings)
        )
    return output[0, len(prompt_ids) :].tolist()


def sampling_inputs(examples, mode):
    """Avoid sampling the same official prompt once for every existing response."""
    if mode == "replay":
        return [(example, [example.id]) for example in examples]
    groups = {}
    for example in examples:
        evidence = tuple((span["id"], span["start"], span["end"]) for span in example.evidence)
        key = (example.source_id, example.prompt, evidence)
        if key not in groups:
            groups[key] = (example, [])
        groups[key][1].append(example.id)
    return list(groups.values())


def sampling_jobs(examples, options):
    jobs = []
    for example, parents in sampling_inputs(examples, options.mode):
        for draw in range(options.samples):
            identity = example.id if options.mode == "replay" else f"{example.id}:sample:{draw}"
            sample = dict(
                index=len(jobs),
                id=identity,
                source_id=example.source_id,
                parent_ids=parents,
                draw=draw,
                seed=options.seed + draw,
            )
            jobs.append((sample, example))
    return jobs


def generate_run(model, tokenizer, data, root, options, model_settings, resume=False) -> dict:
    jobs = sampling_jobs(load_examples(data), options)
    manifest = dict(
        schema_version=2,
        model=model_settings,
        generation=asdict(options),
        model_config=model.native.config.to_dict(),
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        samples=[sample for sample, _ in jobs],
    )
    start_stage(root / "run.json", manifest, resume)
    for sample, example in tqdm(jobs, desc="answers"):
        path = root / "samples" / f"{sample['index']:06d}" / "answer.json"
        if resume and path.exists():
            continue
        record = make_answer(model, tokenizer, example, options, sample["seed"])
        record.update(sample)
        write_json(path, record)
    return manifest
