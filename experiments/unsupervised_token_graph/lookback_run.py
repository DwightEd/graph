"""Freeze label-free lookback structure measurements for attention caches."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CacheDataset
from .reanchor import ReanchorAnalyzer


def read_metadata(path):
    if path is None:
        return {}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {Path(row["trace"]).stem: row for row in rows}


def run(cache_root, output, metadata=None, window=10, horizon_low=10, horizon_high=100):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    meta = read_metadata(metadata)
    analyzer = ReanchorAnalyzer(window, horizon_low, horizon_high)
    rows = []
    for record in CacheDataset(cache_root):
        result = analyzer.run(record.attention, record.prompt_length)
        identity = meta.get(record.response_id, {})
        np.savez_compressed(output / f"{record.response_id}.npz",
                            waad=result.waad, fai=result.fai,
                            local_mass=result.local_mass,
                            remote_history_mass=result.remote_history_mass,
                            prompt_mass=result.prompt_mass,
                            evidence_entropy=result.evidence_entropy,
                            evidence_concentration=result.evidence_concentration,
                            distribution_shift=result.distribution_shift,
                            lookback_ratio=result.lookback_ratio,
                            valid_history=result.valid_history,
                            event=result.event,
                            event_strength=result.event_strength)
        rows.append({"response_id": record.response_id,
                     "source_id": identity.get("source_id", record.source_id),
                     "seed": identity.get("seed"),
                     "tokens": len(result.waad),
                     "valid_history": int(result.valid_history.sum()),
                     "events": int(result.event.sum()),
                     "event_rate": float(result.event.mean())})
    (output / "summary.json").write_text(json.dumps({"responses": rows,
        "config": {"window": window, "horizon_low": horizon_low, "horizon_high": horizon_high,
                    "labels_read": False}}, indent=2), encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--horizon-low", type=int, default=10)
    parser.add_argument("--horizon-high", type=int, default=100)
    args = parser.parse_args(argv)
    rows = run(args.cache, args.output, args.metadata, args.window, args.horizon_low, args.horizon_high)
    print(json.dumps({"complete": True, "responses": len(rows), "labels_read": False}))


if __name__ == "__main__":
    main()