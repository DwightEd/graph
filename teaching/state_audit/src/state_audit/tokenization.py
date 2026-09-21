"""One explicit alignment: response token t is predicted at query P + t - 1."""

import numpy as np

from .dataset import Example


def encode_prompt(tokenizer, example: Example, template: str) -> dict:
    text = example.prompt
    if template == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if text.count(example.prompt) != 1:
            raise ValueError("Chat template must preserve the prompt text exactly once")
    start = text.index(example.prompt)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = np.asarray(encoded["offset_mapping"])
    key_sources = np.full(len(offsets), -1, dtype=int)
    for index, span in enumerate(example.evidence):
        hits = (offsets[:, 0] < start + span["end"]) & (offsets[:, 1] > start + span["start"])
        if np.any(key_sources[hits] >= 0) or not np.any(hits):
            raise ValueError("Evidence overlaps at tokenizer boundaries or has no tokens")
        key_sources[hits] = index
    return dict(
        prompt_text=text,
        prompt_ids=encoded["input_ids"],
        prompt_offsets=offsets.tolist(),
        key_sources=key_sources.tolist(),
    )


def generated_offsets(tokenizer, response_ids: list[int], text: str) -> list | None:
    """Retokenization verifies offsets only; it NEVER replaces the generated IDs."""
    special = set(special_token_ids(tokenizer))
    ordinary = [token for token in response_ids if token not in special]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if ordinary != encoded["input_ids"]:
        return None
    offsets = iter(encoded["offset_mapping"])
    return [[0, 0] if token in special else list(next(offsets)) for token in response_ids]


def answer_record(
    tokenizer,
    example: Example,
    prompt: dict,
    response_ids: list[int],
    response: str,
    offsets: list | None,
    mode: str,
    seed: int,
) -> dict:
    ids = prompt["prompt_ids"] + response_ids
    if not prompt["prompt_ids"] or not response_ids:
        raise ValueError("Prompt and response must both contain at least one token")
    return dict(
        schema_version=2,
        id=example.id,
        source_id=example.source_id,
        metadata=example.metadata,
        mode=mode,
        seed=seed,
        **prompt,
        token_ids=ids,
        prompt_length=len(prompt["prompt_ids"]),
        response_ids=response_ids,
        response=response,
        response_offsets=offsets,
        token_strings=tokenizer.convert_ids_to_tokens(ids),
        special_token_ids=special_token_ids(tokenizer),
        evidence=example.evidence,
        labels=example.labels if mode == "replay" else None,
        labels_status="original_response" if mode == "replay" else "new_response_unreviewed",
    )


def special_token_ids(tokenizer) -> list[int]:
    """Some chat control tokens are marked only in the added-token table."""
    marked = {index for index, token in tokenizer.added_tokens_decoder.items() if token.special}
    return sorted(set(tokenizer.all_special_ids) | marked)
