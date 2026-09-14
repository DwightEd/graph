"""Label-free fit/calibrate/score and a separate frozen evaluation command.

Roster: JSON list of {path, response_id, source_id, split}. Paths are relative to
the roster; split is reference/calibration/test. Explicit source IDs are required.
"""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .data import ResponseCache
from .information_flow import RoutingResidual, SourceCalibration, build_flow_graph


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda a: a.tolist()) + "\n", encoding="utf-8")


def read_roster(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("roster must be a nonempty list")
    seen, sources, paths = set(), {}, set()
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("response_id", "source_id", "path", "split")):
            raise ValueError("roster needs explicit string identities, path and split")
        if row["split"] not in ("reference", "calibration", "test"):
            raise ValueError("invalid roster split")
        source, rid = row["source_id"], row["response_id"]
        if rid in seen or (source in sources and sources[source] != row["split"]):
            raise ValueError("duplicate response or source leakage between splits")
        cache_path = (path.parent / row["path"]).resolve()
        if cache_path in paths:
            raise ValueError("same cache reused under multiple identities")
        seen.add(rid); paths.add(cache_path); sources[source] = row["split"]
        row["path"] = str(cache_path)
    if set(sources.values()) != {"reference", "calibration", "test"}:
        raise ValueError("all three source-disjoint splits are required")
    return rows


def score_roster(roster, output, floor=0., attribute="hidden", alpha=0.05, projection_dim=64, max_attribute_mib=1024):
    rows = read_roster(roster)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be new or empty; frozen runs are never overwritten")
    if not np.isfinite(max_attribute_mib) or max_attribute_mib <= 0:
        raise ValueError("positive finite max_attribute_mib required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    output.mkdir(parents=True, exist_ok=True)
    code = {str(Path(__file__).with_name(name).resolve()): _hash(Path(__file__).with_name(name))
            for name in ("data.py", "information_flow.py", "flow_run.py", "flow_evaluate.py")}
    inputs = [{k: row[k] for k in ("path", "response_id", "source_id", "split")} for row in rows]
    content_sources = {}
    for row in inputs:
        row["sha256"] = _hash(row["path"])
        prior = content_sources.setdefault(row["sha256"], row["source_id"])
        if prior != row["source_id"]:
            raise ValueError("identical cache content assigned to different sources")
    _write(output / "inputs.json", {"rows": inputs, "code": code,
                                   "environment": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "scipy", "scikit-learn")}},
                                   "config": dict(floor=floor, attribute=attribute, alpha=alpha,
                                                  projection_dim=projection_dim, max_attribute_mib=max_attribute_mib)})
    graphs, identities = [], {}
    attribute_bytes = 0
    for row in inputs:
        record = ResponseCache().load(row["path"])
        with np.load(row["path"], allow_pickle=False) as data:
            if "source_id" in data and str(data["source_id"].item()) != row["source_id"]:
                raise ValueError("cache source_id disagrees with roster")
        record.response_id, record.source_id = row["response_id"], row["source_id"]
        graph = build_flow_graph(record, floor=floor, attribute=attribute)
        attribute_bytes += graph.x.nbytes
        if attribute_bytes > max_attribute_mib * 1024 ** 2:
            raise ValueError("node attribute memory budget exceeded; reduce the frozen roster or increase its explicit budget")
        graphs.append(graph)
        ids = record.token_ids
        if ids is not None and (ids.ndim != 1 or len(ids) != len(graph.x) or ids.dtype.kind not in "iu"):
            raise ValueError("token_ids must identify all physical nodes")
        identities[graph.response_id] = None if ids is None else ids[graph.start:]
        if _hash(row["path"]) != row["sha256"]:
            raise ValueError("input changed during loading")
    groups = {role: [g for g, row in zip(graphs, inputs) if row["split"] == role]
              for role in ("reference", "calibration", "test")}
    predictions, calibrations = {}, {}
    for mode in ("graph", "no_neighbors", "mass_matched_uniform", "permuted_weights"):
        model = RoutingResidual(projection_dim=projection_dim, mode=mode).fit(groups["reference"])
        model.save(output / f"{mode}.npz")
        cal = SourceCalibration([model.score(g)["score"] for g in groups["calibration"]],
                                [g.source_id for g in groups["calibration"]], alpha=alpha)
        calibrations[mode] = cal.summary()
        mode_predictions = []
        for g in groups["test"]:
            result = model.score(g)
            p = cal.p_values(result["score"])
            result.update(response_id=g.response_id, source_id=g.source_id,
                          token_indices=np.arange(g.start, len(g.x)), token_ids=identities[g.response_id],
                          p_value=p, alarm=p <= alpha)
            mode_predictions.append(result)
        predictions[mode] = mode_predictions
    _write(output / "scores.json", predictions)
    _write(output / "calibration.json", calibrations)
    if any(_hash(path) != digest for path, digest in code.items()):
        raise ValueError("scorer code changed during run; refusing freeze")
    artifacts = {p.name: _hash(p) for p in output.iterdir() if p.is_file()}
    _write(output / "freeze.json", {"status": "scores_frozen_before_labels", "artifacts": artifacts,
                                    "code": code, "timing": "post-observation physical token query i",
                                    "claim": "engineering artifact; natural hallucination efficacy unverified"})
    return {"responses": {k: len(v) for k, v in groups.items()}, "output": str(output)}


def verify_freeze(output):
    output = Path(output)
    freeze = json.loads((output / "freeze.json").read_text())
    for name, digest in freeze["artifacts"].items():
        if Path(name).name != name or _hash(output / name) != digest:
            raise ValueError(f"frozen artifact changed: {name}")
    for path, digest in freeze["code"].items():
        if _hash(path) != digest:
            raise ValueError(f"frozen code changed: {path}")
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    score = sub.add_parser("score")
    score.add_argument("--roster", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--floor", type=float, default=0.)
    score.add_argument("--attribute", choices=("diagonal", "hidden"), default="hidden")
    score.add_argument("--alpha", type=float, default=0.05)
    score.add_argument("--projection-dim", type=int, default=64)
    score.add_argument("--max-attribute-mib", type=float, default=1024)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--annotations", required=True)
    args = parser.parse_args(argv)
    if args.phase == "score":
        result = score_roster(args.roster, args.output, args.floor, args.attribute, args.alpha, args.projection_dim, args.max_attribute_mib)
    else:
        verify_freeze(args.output)
        from .flow_evaluate import evaluate_frozen
        result = evaluate_frozen(args.output, args.annotations)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
