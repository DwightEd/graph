"""Original token identities, explicit source deletion, and text-only intervals."""

import re

import numpy as np

BOUNDARY = re.compile(r'\n+|[.!?。！？]+["”’\')\]]*(?=\s|$)')
LIST_MARKER = re.compile(r'\s*(?:\d+[.)]|[-*•])\s*')


def unit_intervals(response, max_tokens):
    """Align punctuation to whole tokens; split long units without dropping text."""
    pieces = response["token_text"][response["prompt_length"]:]
    text = "".join(pieces)
    ends = np.cumsum([len(piece) for piece in pieces])
    boundaries, start = [0], 0
    for match in BOUNDARY.finditer(text):
        fragment = text[start:match.end()]
        if not fragment.strip() or LIST_MARKER.fullmatch(fragment):
            continue
        stop = int(np.searchsorted(ends, match.end(), side="left")) + 1
        if stop > boundaries[-1]:
            boundaries.append(stop)
            start = int(ends[stop - 1])
    if boundaries[-1] != len(pieces):
        boundaries.append(len(pieces))
    units = []
    for left, right in zip(boundaries, boundaries[1:]):
        for begin in range(left, right, max_tokens):
            units.append(dict(start=begin, stop=min(begin + max_tokens, right),
                              text_start=left, text_stop=right))
    return units


def prepare_views(response, sources, max_tokens):
    """Delete only saved source keys in the prompt; never retokenize the answer."""
    ids, prompt = response["token_ids"], response["prompt_length"]
    groups = np.asarray(sources["group_ids"])
    if not 0 < prompt < len(ids) or len(response["token_text"]) != len(ids):
        raise ValueError(f"{response['id']}: invalid original token/prompt alignment")
    if groups.shape != (len(ids),) or (groups < 0).any():
        raise ValueError(f"{response['id']}: source groups differ from original tokens")
    source_mask = groups[:prompt] < len(sources["blocks"])
    kept = np.flatnonzero(~source_mask).tolist()
    if not kept:
        raise ValueError(f"{response['id']}: source deletion removed the entire prompt")
    return dict(prompt_with_source=ids[:prompt],
                prompt_without_source=[ids[index] for index in kept],
                kept_prompt_positions=kept,
                removed_prompt_positions=np.flatnonzero(source_mask).tolist(),
                answer_ids=ids[prompt:], units=unit_intervals(response, max_tokens))


def query_positions(views):
    count = len(views["answer_ids"])
    starts = np.empty(count, dtype=np.int64)
    unit_id = np.empty(count, dtype=np.int64)
    for index, unit in enumerate(views["units"]):
        selected = slice(unit["start"], unit["stop"])
        starts[selected], unit_id[selected] = unit["start"], index
    target = np.arange(count)
    result = dict(target=target, token_id=np.asarray(views["answer_ids"]), unit_id=unit_id)
    for condition in ("with_source", "without_source"):
        last_prompt = len(views[f"prompt_{condition}"]) - 1
        result[f"full_query_{condition}"] = last_prompt + target
        result[f"local_query_{condition}"] = last_prompt + target - starts
    return result
