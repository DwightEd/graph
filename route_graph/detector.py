"""Source-balanced reference distances with fixed, causal nuisance strata."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from route_graph.capture import file_digest
from route_graph.data import read_jsonl, text_digest, write_jsonl

VIEWS = ("residual", "observed", "null", "signal")


def _stratum(row: dict) -> tuple:
    return (
        row["task"],
        row["generator"],
        int(row["prompt_tokens"]).bit_length() - 1,
        (int(row["token_index"]) + 1).bit_length() - 1,
    )


class SourceReference:
    """Shared-scale deviation and supplementary distance to distinct sources."""

    def __init__(self, neighbors: int = 3, per_source: int = 8) -> None:
        if neighbors < 1 or per_source < 1:
            raise ValueError("neighbors and per_source must be positive")
        self.neighbors, self.per_source = neighbors, per_source

    def fit(self, rows: list[dict]) -> SourceReference:
        if not rows:
            raise ValueError("empty reference set")
        self.sources = {row["source_id"] for row in rows}
        grouped = defaultdict(lambda: defaultdict(list))
        widths = set()
        for row in rows:
            for view in VIEWS:
                values = np.asarray(row[view], dtype=float)
                if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                    raise ValueError(
                        "reference features must be finite nonempty vectors"
                    )
                widths.add(len(values))
            grouped[_stratum(row)][row["source_id"]].append(row)
        if len(widths) != 1:
            raise ValueError("reference feature dimensions differ")
        self.width = widths.pop()
        self.profiles = {}
        for key, by_source in grouped.items():
            quota = min(self.per_source, min(map(len, by_source.values())))
            selected = []
            for source in sorted(by_source):
                ordered = sorted(
                    by_source[source],
                    key=lambda r: text_digest(f"{r['response_id']}:{r['token_index']}"),
                )
                selected.append(ordered[:quota])
            profile = {
                "source_ids": sorted(by_source),
                "examples_per_source": quota,
                "reference_ids": [
                    [f"{r['response_id']}:{r['token_index']}" for r in group]
                    for group in selected
                ],
            }
            values_by_view = {
                view: np.array([[r[view] for r in group] for group in selected])
                for view in VIEWS
            }
            # One scale and active-coordinate mask for all representations. Constant
            # axes cannot turn harmless float32 roundoff into huge anomaly scores.
            scale = np.max(
                [
                    values.reshape(-1, self.width).std(axis=0)
                    for values in values_by_view.values()
                ],
                axis=0,
            )
            active = scale > 1e-6
            profile.update(active_mask=active, shared_scale=scale)
            for view, values in values_by_view.items():
                flat = values.reshape(-1, self.width)
                center = np.median(flat, axis=0)
                profile[view] = {
                    "center": center,
                    "reference": (values[..., active] - center[active]) / scale[active],
                }
            self.profiles[key] = profile
        return self

    def score(self, row: dict) -> dict[str, float]:
        if row["source_id"] in self.sources:
            raise ValueError("fit and score must be source-disjoint")
        key = _stratum(row)
        profile = self.profiles.get(key)
        if profile is None or len(profile["source_ids"]) < self.neighbors:
            raise ValueError(
                f"insufficient reference stratum {key}; need {self.neighbors} sources"
            )
        result = {}
        for view in VIEWS:
            values = np.asarray(row[view], dtype=float)
            if values.shape != (self.width,) or not np.isfinite(values).all():
                raise ValueError(
                    "score features must be finite and match reference dimensions"
                )
            fitted = profile[view]
            active = profile["active_mask"]
            if not active.any():
                result[view] = result[f"{view}_knn"] = 0.0
                continue
            standardized = (values[active] - fitted["center"][active]) / profile[
                "shared_scale"
            ][active]
            distances = np.sqrt(
                np.mean((fitted["reference"] - standardized) ** 2, axis=-1)
            )
            by_source = distances.min(axis=1)
            result[view] = float(np.mean(standardized**2))
            result[f"{view}_knn"] = float(np.sort(by_source)[: self.neighbors].mean())
        return result


class RouteDetector:
    """Load one completed capture, freeze train-source references, score test tokens."""

    def __init__(
        self,
        features_dir: Path,
        output_dir: Path,
        neighbors: int = 3,
        per_source: int = 8,
    ) -> None:
        self.features_dir, self.output_dir = features_dir, output_dir
        self.neighbors, self.per_source = neighbors, per_source

    def run(self) -> dict:
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        manifest = json.loads(
            (self.features_dir / "manifest.json").read_text(encoding="utf-8")
        )
        path = self.features_dir / "features.jsonl"
        if (
            manifest["schema"] != "route-graph/capture@1"
            or manifest["labels_used"] is not False
            or file_digest(path) != manifest["features_sha256"]
        ):
            raise ValueError(
                "capture schema, label boundary or feature digest mismatch"
            )
        rows = read_jsonl(path)
        expected = {
            "schema",
            "response_id",
            "source_id",
            "split",
            "task",
            "generator",
            "response_sha256",
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
        identities = set()
        by_response = defaultdict(list)
        for row in rows:
            if set(row) != expected or row["schema"] != "route-graph/features@1":
                raise ValueError("unexpected feature fields or schema")
            identity = (row["response_id"], row["token_index"])
            if identity in identities:
                raise ValueError("duplicate feature token")
            identities.add(identity)
            if row["split"] not in {"train", "test"}:
                raise ValueError("unexpected split")
            if any(len(row[view]) != len(manifest["feature_names"]) for view in VIEWS):
                raise ValueError("feature manifest dimension mismatch")
            if (
                row["prompt_tokens"] < 1
                or row["token_count"] < 1
                or row["predictor_index"]
                != row["prompt_tokens"] + row["token_index"] - 1
            ):
                raise ValueError("invalid predictor metadata in capture")
            by_response[row["response_id"]].append(row)
        if len(rows) != manifest["tokens"]:
            raise ValueError("incomplete feature capture")
        if len(by_response) != manifest["responses"]:
            raise ValueError("complete capture requires every response in the manifest")
        fixed = (
            "source_id",
            "split",
            "task",
            "generator",
            "response_sha256",
            "prompt_tokens",
            "token_count",
        )
        for tokens in by_response.values():
            first = tokens[0]
            if sorted(row["token_index"] for row in tokens) != list(
                range(first["token_count"])
            ) or any(row[key] != first[key] for row in tokens for key in fixed):
                raise ValueError(
                    "complete capture requires aligned full token streams in both splits"
                )
        fit = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        if not test:
            raise ValueError("no test tokens")
        reference = SourceReference(self.neighbors, self.per_source).fit(fit)
        scores = []
        for row in test:
            values = reference.score(row)
            values.update(
                entropy=row["entropy"],
                negative_margin=row["negative_margin"],
                position=row["token_index"],
            )
            scores.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {*VIEWS, "entropy", "negative_margin"}
                }
            )
            scores[-1].update(schema="route-graph/score@1", scores=values)
            scores[-1]["reference_active_features"] = int(
                reference.profiles[_stratum(row)]["active_mask"].sum()
            )
        self.output_dir.mkdir(parents=True)
        write_jsonl(self.output_dir / "scores.jsonl", scores)
        profiles = [
            {"stratum": key, **profile} for key, profile in reference.profiles.items()
        ]
        report = {
            "schema": "route-graph/reference@1",
            "fit_sources": sorted(reference.sources),
            "fit_tokens": len(fit),
            "scored_tokens": len(scores),
            "neighbors": self.neighbors,
            "per_source_cap": self.per_source,
            "score_direction": "larger_is_more_anomalous",
            "standardization": "per-view median; shared maximum within-view std; active std > 1e-6",
            "inactive_stratum_policy": "zero scores with reference_active_features=0",
            "labels_used": False,
            "features_sha256": manifest["features_sha256"],
            "profiles": profiles,
        }
        (self.output_dir / "reference.json").write_text(
            json.dumps(
                report, indent=2, default=lambda value: value.tolist(), allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "fit_sources": len(reference.sources),
            "fit_tokens": len(fit),
            "scored_tokens": len(scores),
            "output": str(self.output_dir),
        }
