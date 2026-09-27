"""Label-free text views and exact character-to-token projection."""

import json
import re

import numpy as np


def sentences(text):
    """Partition all characters, retaining decimals and original offsets."""
    starts = [0]
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z*0-9])|\n+|\\n", text):
        starts.append(match.end())
    stops = starts[1:] + [len(text)]
    return [dict(start=start, stop=stop, text=text[start:stop])
            for start, stop in zip(starts, stops) if stop > start]


def flatten_source(value, path=()):
    """Lossless JSON leaf paths: retain types, list indices and empty containers."""
    if isinstance(value, dict) and value:
        return [line for key, item in value.items()
                for line in flatten_source(item, (*path, key))]
    if isinstance(value, list) and value:
        return [line for index, item in enumerate(value)
                for line in flatten_source(item, (*path, index))]
    return [json.dumps(dict(path=path, value=value), ensure_ascii=False)]


def parse_claims(raw, answer, evidence):
    """Keep exact unique answer quotes; invalid outputs remain counted, never dropped."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as error:
        return [], [dict(reason="invalid_json", detail=str(error))]
    if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
        return [], [dict(reason="invalid_schema")]
    claims, invalid = [], []
    for item in parsed["claims"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str)
                for key in ("quote", "claim", "evidence", "alternative")):
            invalid.append(dict(reason="invalid_claim_schema", item=item))
            continue
        quote = item["quote"]
        matches = list(re.finditer(re.escape(quote), answer)) if quote.strip() else []
        if len(matches) != 1 or not item["claim"].strip():
            invalid.append(dict(reason="nonunique_or_missing_quote", item=item))
            continue
        match = matches[0]
        claims.append(dict(**item, start=match.start(), stop=match.end(),
                           evidence_exact=bool(item["evidence"]) and item["evidence"] in evidence))
    return claims, invalid


def project_scores(offsets, sentence_rows, sentence_scores, claims, claim_scores):
    """Every original token gets a score; claim overlaps use max, gaps use baseline."""
    offsets = np.asarray(offsets)
    direct = np.zeros(len(offsets), dtype=np.float64)
    for unit, score in zip(sentence_rows, sentence_scores, strict=True):
        overlap = (offsets[:, 0] < unit["stop"]) & (offsets[:, 1] > unit["start"])
        direct[overlap] = score
    refined = np.full(len(offsets), -np.inf)
    for claim, score in zip(claims, claim_scores, strict=True):
        overlap = (offsets[:, 0] < claim["stop"]) & (offsets[:, 1] > claim["start"])
        refined[overlap] = np.maximum(refined[overlap], score)
    covered = np.isfinite(refined)
    refined[~covered] = direct[~covered]
    return direct, refined, covered
