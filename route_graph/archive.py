"""Recover complete responses without inventing a missing capture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from route_graph.data import read_responses, text_digest

VIEWS = ("observed", "null", "residual", "signal")
FIXED = ("source_id", "split", "task", "generator", "response_sha256")
FIELDS = {
    "schema",
    "response_id",
    *FIXED,
    "token_index",
    "token_count",
    "char_span",
    "token_id",
    "prompt_tokens",
    "predictor_index",
    "candidate_ids",
    "entropy",
    "negative_margin",
    *VIEWS,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_response(rows: list[dict], record: dict) -> None:
    """Check complete causal token metadata and finite aligned raw features."""
    first = rows[0]
    count = first["token_count"]
    if type(count) is not int or count < 1:
        raise ValueError("invalid token_count")
    if [r["token_index"] for r in rows] != list(range(count)):
        raise ValueError("incomplete or unordered response")
    widths = set()
    for row in rows:
        if set(row) != FIELDS:
            raise ValueError("unexpected feature fields; labels are not allowed")
        for key in (
            "token_id",
            "token_index",
            "token_count",
            "prompt_tokens",
            "predictor_index",
        ):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("invalid token identity or position type")
        candidates = row["candidate_ids"]
        if (
            not isinstance(candidates, list)
            or len(candidates) < 2
            or any(type(c) is not int or c < 0 for c in candidates)
            or len(set(candidates)) != len(candidates)
            or len(candidates) != len(first["candidate_ids"])
        ):
            raise ValueError("invalid candidate IDs")
        if (
            not isinstance(row["char_span"], list)
            or len(row["char_span"]) != 2
            or any(type(v) is not int for v in row["char_span"])
        ):
            raise ValueError("invalid character interval type")
        if (
            row["schema"] != "route-graph/features@1"
            or row["response_id"] != record["id"]
        ):
            raise ValueError("unexpected feature schema or response ID")
        if any(row[key] != record[key] for key in FIXED):
            raise ValueError("input identity mismatch")
        if (
            row["token_count"] != count
            or row["prompt_tokens"] != first["prompt_tokens"]
        ):
            raise ValueError("inconsistent response metadata")
        if (
            row["prompt_tokens"] < 1
            or row["predictor_index"] != row["prompt_tokens"] + row["token_index"] - 1
        ):
            raise ValueError("invalid predictor index")
        left, right = row["char_span"]
        if not 0 <= left < right <= len(record["response"]):
            raise ValueError("invalid character interval")
        if not np.isfinite([row["entropy"], row["negative_margin"]]).all():
            raise ValueError("nonfinite scalar")
        for view in VIEWS:
            values = np.asarray(row[view], dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError("invalid feature vector")
            widths.add(len(values))
        if not np.allclose(
            np.asarray(row["observed"]) - row["null"],
            row["residual"],
            atol=1e-10,
            rtol=1e-8,
        ):
            raise ValueError("residual does not equal observed minus null")
    if len(widths) != 1:
        raise ValueError("feature widths differ")


class FeatureArchive:
    """One atomic NPZ per complete response, with a raw-input audit index."""

    def __init__(self, directory: Path):
        self.directory = directory

    def recover(self, features: Path, inputs: Path) -> dict:
        records = {r["id"]: r for r in read_responses(inputs)}
        self.directory.mkdir(parents=True, exist_ok=True)
        stat = features.stat()
        identity = {
            "features": str(features.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "input_sha256": digest(inputs),
        }
        settings = self.directory / "settings.json"
        if settings.exists() and json.loads(settings.read_text()) != identity:
            raise ValueError("archive inputs changed; choose a new directory")
        atomic_json(settings, identity)
        index, rejected, seen = [], [], set()
        rows, current, raw_hash = [], None, hashlib.sha256()
        total_hash, line_count = hashlib.sha256(), 0

        def finish():
            if current in seen:
                raise ValueError("response appears in disjoint blocks")
            seen.add(current)
            if current not in records:
                raise ValueError("feature response absent from input")
            try:
                validate_response(rows, records[current])
            except (ValueError, KeyError, TypeError) as error:
                rejected.append(
                    {"response_id": current, "rows": len(rows), "reason": str(error)}
                )
                return
            if index and len(rows[0]["residual"]) != index[0]["width"]:
                raise ValueError("feature widths differ across responses")
            index.append(self._save(rows, raw_hash.hexdigest()))

        with (
            features.open("rb") as stream,
            tqdm(
                total=stat.st_size,
                unit="B",
                unit_scale=True,
                desc="recover raw features",
            ) as progress,
        ):
            for raw in stream:
                line_count += 1
                total_hash.update(raw)
                progress.update(len(raw))
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    # Only a torn final line is recoverable. Interior corruption fails.
                    if stream.read(1):
                        raise ValueError(
                            f"invalid JSON inside capture at line {line_count}"
                        ) from None
                    rejected.append(
                        {"line": line_count, "reason": "truncated final JSON line"}
                    )
                    break
                key = row["response_id"]
                if current is not None and key != current:
                    finish()
                    rows, raw_hash = [], hashlib.sha256()
                current = key
                rows.append(row)
                raw_hash.update(raw)
            if rows:
                finish()
        if (
            features.stat().st_mtime_ns != stat.st_mtime_ns
            or features.stat().st_size != stat.st_size
        ):
            raise ValueError("capture changed during recovery")
        report = {
            "schema": "route-graph/recovered@1",
            "provenance": "original capture manifest unavailable; layout and checkpoint hashes unknown",
            "original_capture_complete": False,
            "labels_used": False,
            "features_sha256": total_hash.hexdigest(),
            "input_sha256": identity["input_sha256"],
            "planned_responses": len(records),
            "complete_responses": len(index),
            "complete_tokens": sum(r["tokens"] for r in index),
            "raw_lines": line_count,
            "responses": index,
            "rejected": rejected,
            "missing_response_ids": sorted(set(records) - seen),
        }
        temporary = self.directory / "index.partial"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(self.directory / "index.json")
        return report

    def _save(self, rows, raw_sha):
        key = text_digest(rows[0]["response_id"])[:24]
        path = self.directory / f"{key}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["raw_sha256"] != raw_sha or digest(path) != old["sha256"]:
                raise ValueError("completed archive response changed")
            return old
        # An interrupted unmarked partial belongs to this archive only.
        partial = path.with_suffix(".partial")
        metadata = [{k: v for k, v in r.items() if k not in VIEWS} for r in rows]
        with partial.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=json.dumps(metadata),
                **{
                    v: np.asarray([r[v] for r in rows], dtype=np.float64) for v in VIEWS
                },
            )
        partial.replace(path)
        entry = {
            "response_id": rows[0]["response_id"],
            "source_id": rows[0]["source_id"],
            "split": rows[0]["split"],
            "tokens": len(rows),
            "width": len(rows[0]["residual"]),
            "file": path.name,
            "raw_sha256": raw_sha,
            "sha256": digest(path),
        }
        atomic_json(marker, entry)
        return entry

    def responses(self):
        report = json.loads((self.directory / "index.json").read_text())
        if (
            report["schema"] != "route-graph/recovered@1"
            or report["labels_used"] is not False
        ):
            raise ValueError("invalid recovered archive")
        for entry in report["responses"]:
            if entry["width"] != report["responses"][0]["width"]:
                raise ValueError("archive feature widths differ")
            path = self.directory / entry["file"]
            if path.parent != self.directory or digest(path) != entry["sha256"]:
                raise ValueError("archive file identity mismatch")
            with np.load(path, allow_pickle=False) as saved:
                rows = json.loads(str(saved["metadata"]))
                if (
                    len(rows) != entry["tokens"]
                    or [r["token_index"] for r in rows] != list(range(entry["tokens"]))
                    or any(set(r) != FIELDS - set(VIEWS) for r in rows)
                    or any(
                        r["response_id"] != entry["response_id"]
                        or r["source_id"] != entry["source_id"]
                        or r["split"] != entry["split"]
                        for r in rows
                    )
                ):
                    raise ValueError("archive token metadata mismatch")
                for view in VIEWS:
                    if saved[view].shape != (entry["tokens"], entry["width"]):
                        raise ValueError("archive feature dimensions mismatch")
                    for row, vector in zip(rows, saved[view], strict=True):
                        row[view] = vector
            yield rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = FeatureArchive(args.output).recover(args.features, args.input)
    print(json.dumps({k: v for k, v in result.items() if k != "responses"}, indent=2))


if __name__ == "__main__":
    main()
