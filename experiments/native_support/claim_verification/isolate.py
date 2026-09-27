"""Frozen context-removal comparison: same reader/source/claims, no answer context."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .run import save_json, score_claims
from .runtime import Reader
from .text import project_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    for row in manifest["records"]:
        if not (args.output / "responses" / row["id"] / "scores.npz").exists():
            raise ValueError("Complete all original predictions before this paired comparison")
    reader = Reader(manifest["model"], args.batch_size)
    started = perf_counter()
    for row in manifest["records"]:
        directory = args.output / "responses" / row["id"]
        if (directory / "isolated_scores.npz").exists():
            continue
        audit = json.loads((directory / "audit.json").read_text())
        claims = audit["claims"]
        direct, direct_orders = score_claims(reader, row["source"], "", [unit["text"] for unit in row["sentences"]])
        atomic, atomic_orders = score_claims(reader, row["source"], "", [claim["claim"] for claim in claims])
        baseline, refined, covered = project_scores(row["offsets"], row["sentences"], direct, claims, atomic)
        save_json(directory / "isolated_audit.json", dict(direct_scores=direct, atom_scores=atomic,
                  direct_label_orders=direct_orders, atom_label_orders=atomic_orders,
                  context_removed=True, interpretation="input ablation, not generator causal mechanism"))
        np.savez_compressed(directory / "isolated_scores.tmp.npz", token_id=np.asarray(row["token_ids"]),
                            direct_isolated=baseline, atomic_isolated=refined, claim_covered=covered)
        (directory / "isolated_scores.tmp.npz").replace(directory / "isolated_scores.npz")
        print(json.dumps(dict(id=row["id"], seconds=round(perf_counter() - started, 2))), flush=True)
    save_json(args.output / "isolation_completed.json", dict(answers=len(manifest["records"]),
              seconds=perf_counter() - started, labels_used=False))


if __name__ == "__main__":
    main()
