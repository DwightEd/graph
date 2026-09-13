"""Exploratory frozen baseline on an explicitly incomplete recovered capture."""

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from route_graph.archive import FIELDS, VIEWS, FeatureArchive, atomic_json, digest
from route_graph.data import write_jsonl
from route_graph.detector import SourceReference, _stratum
from route_graph.evaluation import RouteEvaluator


def score_archive(archive: Path, output: Path, neighbors=3, per_source=8):
    if output.exists():
        raise FileExistsError(output)
    groups = list(FeatureArchive(archive).responses())
    fit = [r for group in groups for r in group if r["split"] == "train"]
    test = [group for group in groups if group[0]["split"] == "test"]
    reference = SourceReference(neighbors, per_source).fit(fit)
    if any(r["source_id"] in reference.sources for group in test for r in group):
        raise ValueError("fit/test source overlap")
    # Eligibility is determined from unlabeled metadata before reading annotations.
    supported, excluded = [], []
    for group in test:
        unavailable = sorted(
            {
                _stratum(r)
                for r in group
                if _stratum(r) not in reference.profiles
                or len(reference.profiles[_stratum(r)]["source_ids"]) < neighbors
            }
        )
        if unavailable:
            excluded.append(
                {
                    "response_id": group[0]["response_id"],
                    "source_id": group[0]["source_id"],
                    "tokens": len(group),
                    "unavailable_strata": unavailable,
                }
            )
        else:
            supported.append(group)
    output.mkdir(parents=True)
    report = {
        "schema": "route-graph/recovered-scores@1",
        "archive_index_sha256": digest(archive / "index.json"),
        "labels_used": False,
        "neighbors": neighbors,
        "per_source": per_source,
        "fit_sources": len(reference.sources),
        "fit_tokens": len(fit),
        "complete_test_responses": len(test),
        "eligible_test_responses": len(supported),
        "eligible_test_tokens": sum(map(len, supported)),
        "excluded_test_responses": excluded,
        "original_capture_provenance": "unknown layout and checkpoint digests; exploratory only",
    }
    atomic_json(output / "coverage.json", report)
    if not supported:
        raise ValueError("no complete test response has full reference coverage")

    def scored_rows():
        for group in tqdm(supported, desc="score recovered responses"):
            for row in group:
                scores = reference.score(row)
                scores.update(
                    entropy=row["entropy"],
                    negative_margin=row["negative_margin"],
                    position=row["token_index"],
                )
                metadata = {
                    k: row[k] for k in FIELDS - {*VIEWS, "entropy", "negative_margin"}
                }
                yield dict(
                    metadata,
                    schema="route-graph/score@1",
                    scores=scores,
                    reference_active_features=int(
                        reference.profiles[_stratum(row)]["active_mask"].sum()
                    ),
                )

    write_jsonl(output / "scores.jsonl", scored_rows())
    report["scores_sha256"] = digest(output / "scores.jsonl")
    atomic_json(output / "complete.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--archive", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.command == "score":
        result = score_archive(args.archive, args.output)
    else:
        # Reuse the same label joining and source-balanced metrics as mainline.
        completed = json.loads((args.scores.parent / "complete.json").read_text())
        if completed["labels_used"] is not False or completed[
            "scores_sha256"
        ] != digest(args.scores):
            raise ValueError("scores do not match the completed scoring run")
        result = RouteEvaluator(
            args.scores, args.labels, args.output, args.bootstrap, 20260912
        ).run()
        result.update(
            scores_sha256=completed["scores_sha256"],
            archive_index_sha256=completed["archive_index_sha256"],
            labels_sha256=digest(args.labels),
        )
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
