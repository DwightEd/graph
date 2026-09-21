"""Pair reviewed normal/error answers from exactly the same prompt, never from seed alone."""

from itertools import product
from pathlib import Path

from .dataset.jsonl import read_jsonl, validate
from .dataset.schema import Example
from .storage import read_json, write_json


def import_reviews(root: Path, path: Path):
    """Review JSONL: sample (integer), exact response text, character labels."""
    for row in read_jsonl(path):
        directory = root / "samples" / f"{row['sample']:06d}"
        answer = read_json(directory / "answer.json")
        if row["response"] != answer["response"]:
            raise ValueError(f"Sample {row['sample']}: review belongs to a different response")
        example = Example(
            answer["id"],
            answer["source_id"],
            answer["prompt_text"],
            [],
            answer["response"],
            row["labels"],
        )
        validate(example)
        write_json(directory / "review.json", row)


def load_answer(directory: Path) -> dict:
    answer = read_json(directory / "answer.json")
    review_path = directory / "review.json"
    if review_path.exists():
        review = read_json(review_path)
        answer = dict(answer, labels=review["labels"], labels_status="reviewed_response")
    return answer


def pair_answers(root: Path, output: Path) -> dict:
    groups = {}
    unreviewed = 0
    for sample in read_json(root / "run.json")["samples"]:
        answer = load_answer(root / "samples" / f"{sample['index']:06d}")
        if answer["labels"] is None:
            unreviewed += 1
            continue
        key = (answer["source_id"], tuple(answer["prompt_ids"]))
        group = groups.setdefault(key, {"normal": [], "error": []})
        side = "error" if answer["labels"] else "normal"
        group[side].append(sample["index"])
    pairs = []
    for (source, _), sides in groups.items():
        for normal, error in product(sides["normal"], sides["error"]):
            pairs.append(dict(source_id=source, normal=normal, error=error))
    result = dict(
        purpose="reviewed_response_pairs_not_automatic_fact_verification",
        unreviewed=unreviewed,
        pairs=pairs,
    )
    write_json(output, result)
    return result
