import json
from dataclasses import replace

import pytest

from state_audit.capture import CaptureSpec, capture_run
from state_audit.dataset import load_examples
from state_audit.generation import generate_run, make_answer, sample_answer
from state_audit.operations import Delete, Target
from state_audit.pairing import import_reviews, pair_answers
from state_audit.storage import read_json
from state_audit.tokenization import encode_prompt


def test_resampling_deduplicates_prompt_preserves_lineage_and_resumes(tiny_run, tmp_path):
    model, tokenizer, _, data, options, settings = tiny_run
    options = replace(options, mode="generate", samples=3, temperature=0.8)
    root = tmp_path / "resampled"
    manifest = generate_run(model, tokenizer, data, root, options, settings)
    assert len(manifest["samples"]) == 3  # two original answers shared one prompt
    assert [row["seed"] for row in manifest["samples"]] == [7, 8, 9]
    paths = [root / "samples" / f"{index:06d}" / "answer.json" for index in range(3)]
    answers = [read_json(path) for path in paths]
    assert all(answer["parent_ids"] == ["good", "bad"] for answer in answers)
    assert all(answer["labels"] is None for answer in answers)
    assert len({answer["id"] for answer in answers}) == 3
    expected = make_answer(model, tokenizer, load_examples(data)[0], options, 8)
    assert answers[1]["response_ids"] == expected["response_ids"]
    times = [path.stat().st_mtime_ns for path in paths]
    generate_run(model, tokenizer, data, root, options, settings, resume=True)
    assert [path.stat().st_mtime_ns for path in paths] == times
    capture_run(model, root, CaptureSpec(layers=(0,), representations=("residual_after",)))
    pairs = pair_answers(root, root / "pairs.json")
    assert pairs["unreviewed"] == 3 and pairs["pairs"] == []


def test_review_text_identity_and_pairing(tiny_run, tmp_path):
    _, _, root, *_ = tiny_run
    report = pair_answers(root, root / "pairs.json")
    assert report["pairs"] == [dict(source_id="coat", normal=0, error=1)]
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps(dict(sample=0, response="different", labels=[])) + "\n")
    with pytest.raises(ValueError, match="different response"):
        import_reviews(root, reviews)
    response = read_json(root / "samples" / "000001" / "answer.json")["response"]
    reviews.write_text(json.dumps(dict(sample=1, response=response, labels=[])) + "\n")
    import_reviews(root, reviews)
    assert pair_answers(root, root / "pairs.json")["pairs"] == []


def test_intervened_generation_uses_full_prefix_and_keeps_attention_masks(tiny_run):
    model, tokenizer, _, data, options, _ = tiny_run
    example = load_examples(data)[0]
    prompt = encode_prompt(tokenizer, example, options.template)["prompt_ids"]
    target = Target("attention", (0,), heads=(0, 1, 2), keys=(0,))
    baseline = sample_answer(model, tokenizer, prompt, options, seed=1)
    sham = sample_answer(
        model, tokenizer, prompt, options, seed=1, operations=[Delete(target, strength=0)]
    )
    assert baseline == sham
    changed = sample_answer(model, tokenizer, prompt, options, seed=1, operations=[Delete(target)])
    assert len(changed) > 0
