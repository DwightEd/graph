"""Cached, bounded frozen-reader calls; responses are predictions, never gold."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch
from transformers import GenerationConfig

from route_graph.json_framing import parse_framed_json


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json_once(destination, value):
    """Publish a complete file without replacing any existing artifact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


class FrozenReader:
    def __init__(self, model, tokenizer, cache, model_identity, context_limit=8192):
        if model.training:
            raise ValueError("reader requires eval mode")
        if not isinstance(model_identity, dict) or not all(
            key in model_identity for key in ("model", "tokenizer", "code_sha256")
        ):
            raise ValueError("reader requires model/tokenizer/code execution identity")
        self.model, self.tokenizer = model, tokenizer
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.identity = model_identity
        self.limit = context_limit
        self.calls = 0
        self.outcomes = Counter()

    def _record(self, saved, *, cache_hit=False):
        """Account for returned requests, including failed and cached requests."""
        self.outcomes["returned_requests"] += 1
        self.outcomes["cache_hits"] += int(cache_hit)
        prediction = saved["prediction"]
        if "reader_error" in prediction:
            self.outcomes["failure_" + prediction["reader_error"]] += 1
        elif saved["request"]["labels"]:
            self.outcomes["label_logits"] += 1
        else:
            self.outcomes["json_" + saved["json_framing"]["status"]] += 1
        return prediction

    @torch.no_grad()
    def ask(self, instruction, payload, max_new_tokens=1536, labels=None):
        """The exact instruction/payload/settings identify an immutable call.

        No conversational state is carried between calls. With labels, score
        the specified single-token labels at the first response position.
        Otherwise parse one intact bounded JSON root, recording any terminal
        framing punctuation or fence separately and losslessly. Invalid content
        or context yields an explicit failed prediction. Raw output remains;
        no prompt repairs evidence and no partial root is completed.
        """
        if self.model.training:
            raise ValueError("reader requires eval mode")
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        generation = self.model.generation_config.to_dict()
        generation.update(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        request = {
            "model": self.identity,
            "instruction": instruction,
            "payload": payload,
            "max_new_tokens": max_new_tokens,
            "labels": labels,
            "context_limit": self.limit,
            "thinking": False,
            "do_sample": False,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "parser_revision": "intact-first-root-terminal-punctuation@1",
            "dtype": str(self.model.dtype),
            "torch_version": str(torch.__version__),
            "generation_config": generation,
        }
        request = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        key = digest(request)
        destination = self.cache / f"{key}.json"
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved["request"] != request:
                raise ValueError("reader cache identity mismatch")
            return self._record(saved, cache_hit=True)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        count = inputs["input_ids"].shape[1]
        raw = None
        framing = {"status": "not_generated"}
        if count + (1 if labels else max_new_tokens) > self.limit:
            prediction = {"reader_error": "context_limit", "input_tokens": count}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            self.calls += 1
            if labels:
                tokens = [
                    self.tokenizer(label, add_special_tokens=False)["input_ids"]
                    for label in labels
                ]
                if any(len(token) != 1 for token in tokens):
                    raise ValueError("reader labels must each be one token")
                logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
                chosen = logits[[token[0] for token in tokens]]
                probabilities = chosen.softmax(-1).cpu().tolist()
                prediction = {
                    "labels": list(labels),
                    "logits": chosen.cpu().tolist(),
                    "probabilities": probabilities,
                    "normalization": "conditional_on_enumerated_labels",
                }
            else:
                output = self.model.generate(
                    **inputs,
                    generation_config=GenerationConfig.from_dict(generation),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = output[0, count:]
                raw = self.tokenizer.decode(generated, skip_special_tokens=True)
                try:
                    parsed = parse_framed_json(raw)
                    prediction, framing = parsed["prediction"], parsed["framing"]
                except (ValueError, TypeError) as error:
                    prediction = {"reader_error": "invalid_json"}
                    framing = {"status": "unrecovered", "reason": str(error)}
                if (
                    len(generated) >= max_new_tokens
                    and "reader_error" not in prediction
                ):
                    prediction = {
                        "reader_error": "generation_limit",
                        "partial": prediction,
                    }
        saved = {
            "request": request,
            "prediction": prediction,
            "raw_output": raw,
            "json_framing": framing,
        }
        try:
            write_json_once(destination, saved)
        except FileExistsError:
            published = json.loads(destination.read_text())
            if published != saved:
                raise ValueError(
                    "concurrent reader cache prediction mismatch"
                ) from None
            return self._record(published, cache_hit=True)
        return self._record(saved)
