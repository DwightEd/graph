"""Read-only identity/offset binding for already-computed canonical results.

A cache folder declares a partition, not an answer identity. RAGTruth's official
split must agree. Missing token alignment is recovered only with exact saved-ID
verification, never by guessing an offset from a token count.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def input_path(settings, record):
    cache = settings.get("cache")
    if not cache:
        return None
    cache = Path(cache).expanduser()
    if cache.suffix == ".npz":
        return cache
    relative = record.get("cache") or Path(record["file"]).relative_to("samples")
    return cache / relative


def prepare_record(settings, saved):
    """Resolve only cache-side identity before opening any annotation file."""
    record = dict(saved)
    path = input_path(settings, saved)
    partition = next((p.name for p in path.parents if p.name in ("train", "test")), "") if path else ""
    if partition and record.get("split") and partition != record["split"]:
        raise ValueError("saved split conflicts with input cache directory: " + record["file"])
    record["split"] = record.get("split") or partition
    name = Path(record.get("cache") or record["file"]).stem
    rid = str(record.get("id") or name)
    # Existing reader used path.stem when six-field caches had no embedded ID.
    if rid == name and name.startswith("attention_"):
        rid = name.removeprefix("attention_")
    record["id"] = rid
    record["split_origin"] = "saved" if saved.get("split") else ("input_cache_directory" if partition else "annotation")
    return record


def read_sources(annotations_path, explicit=None):
    """Read the existing source_info list, ID map, or official JSONL file."""
    parent = Path(annotations_path).parent
    path = Path(explicit) if explicit else next((p for p in (parent / "source_info.json", parent / "source_info.jsonl") if p.is_file()), None)
    if path is None:
        return {}, None
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [rows] if "id" in rows else [dict(value, id=key) for key, value in rows.items()]
    return {str(row["id"]): row for row in rows}, str(path)


def resolve_tokenizer(settings, record, explicit=None):
    """Use a declared observer tokenizer, never the answer-generator field."""
    if explicit:
        return str(Path(explicit).expanduser())
    configs = [(settings, Path.cwd())]
    path = input_path(settings, record)
    if path:
        for parent in path.parents:
            for name in ("manifest.json", "settings.json"):
                file = parent / name
                if file.is_file():
                    configs.append((json.loads(file.read_text(encoding="utf-8")), parent))
    for stamp in settings.get("index_files", []):
        file = Path(stamp[0]).parent / "settings.json"
        if file.is_file():
            configs.append((json.loads(file.read_text(encoding="utf-8")), file.parent))
    for config, parent in configs:
        for key in ("tokenizer_path", "tokenizer_name_or_path", "tokenizer", "model_path", "model_name_or_path", "model"):
            value = config.get(key)
            if isinstance(value, str) and value:
                local = Path(value).expanduser()
                if local.is_dir():
                    return str(local.resolve())
                if not local.is_absolute() and (parent / local).is_dir():
                    return str((parent / local).resolve())
                # A declared Hugging Face ID is allowed only from the local cache.
                if not local.is_absolute() and len(value.split("/")) == 2:
                    return value
    return None


def verified_offsets(tokenizer, token_ids, prompt_length, response):
    """Re-encode with the original tokenizer and require identical token IDs.

    Try the response-only layout first, then the saved prompt context. An EOS
    suffix is accepted only if every extra saved ID is a declared special token;
    these control tokens receive empty character spans, never fake text spans.
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    p = int(prompt_length)
    expected = token_ids[p:].tolist()
    special = set(tokenizer.all_special_ids)

    def aligned(encoded, prefix_length=0, start=0):
        ids = encoded["input_ids"]
        target = expected if start == 0 else token_ids.tolist()
        if len(ids) > len(target) or target[:len(ids)] != ids:
            return None
        extra = target[len(ids):]
        if any(token not in special for token in extra):
            return None
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64).reshape(-1, 2)[start:]
        offsets = np.clip(offsets - prefix_length, 0, len(response))
        if extra:
            offsets = np.vstack((offsets, np.full((len(extra), 2), len(response), dtype=np.int64)))
        return offsets if len(offsets) == len(expected) else None

    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = aligned(encoded)
    if offsets is not None:
        return offsets
    prefix = tokenizer.decode(token_ids[:p].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(prefix + response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = aligned(encoded, len(prefix), p)
    if offsets is None:
        raise ValueError("saved token_ids do not match the annotated response with this tokenizer; no offsets were fabricated")
    return offsets


class EvaluationBinding:
    """Bind missing metadata in memory; never modify NPZs or resume settings."""

    def __init__(self, settings, tokenizer=None):
        self.settings = settings
        self.tokenizer_path = tokenizer
        self.tokenizers = {}

    def bind(self, record, annotation, arrays, sources):
        result = dict(record)
        official_split = annotation["split"]
        if result.get("split") and result["split"] != official_split:
            raise ValueError("cache/annotation split mismatch: " + result["id"])
        if result.get("source_id") and str(annotation["source_id"]) != result["source_id"]:
            raise ValueError("response/source identity mismatch: " + result["id"])
        digest = hashlib.sha256(annotation["response"].encode()).hexdigest()
        saved_digest = result.get("response_sha256")
        if saved_digest and saved_digest != digest:
            raise ValueError("response identity mismatch: " + result["id"])
        offsets = arrays["offsets"] if "offsets" in arrays else None
        result["alignment_origin"] = "saved_offsets"
        if offsets is None or not saved_digest:
            if "token_ids" not in arrays:
                raise ValueError("saved identity and offsets are required, or saved token_ids plus the original tokenizer: " + result["file"])
            path = resolve_tokenizer(self.settings, record, self.tokenizer_path)
            if path is None:
                missing = "offsets" if offsets is None else "response identity"
                raise ValueError(f"{result['file']}: no identity-bound alignment; saved identity and offsets are required (missing {missing}). Set TOKENIZER (or --tokenizer) to the original observer tokenizer directory. Scores are saved; do not rerun analysis.")
            if path not in self.tokenizers:
                from transformers import AutoTokenizer
                self.tokenizers[path] = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            verified = verified_offsets(self.tokenizers[path], arrays["token_ids"], int(arrays["prompt_length"]), annotation["response"])
            if offsets is None:
                offsets = verified
                result["alignment_origin"] = "exact_token_id_verified_offsets"
            elif not np.array_equal(offsets, verified):
                raise ValueError("saved offsets conflict with exact tokenization: " + result["id"])
            result["verified_tokenizer"] = path
        result.update(split=official_split, source_id=str(annotation["source_id"]), response_sha256=digest)
        source = sources.get(result["source_id"], {})
        result["task"] = result.get("task") or source.get("task_type", source.get("task", ""))
        result["generator"] = result.get("generator") or annotation.get("model", "")
        return result, np.asarray(offsets)
