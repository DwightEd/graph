"""Reproducible CPU engineering example. Synthetic labels are NOT RAGTruth.

python -m examples.token_flow_demo --output outputs/token_flow_demo_new
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.unsupervised_token_graph.flow_run import score_roster
from experiments.unsupervised_token_graph.flow_evaluate import evaluate_frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cache = root / "cache"
    cache.mkdir()
    roster, annotations = [], []
    for i in range(39):
        role = "reference" if i < 16 else "calibration" if i < 35 else "test"
        rng = np.random.default_rng(20260914 + i)
        n, start, d = 20, 4, 6
        x = rng.normal(size=(n, d))
        a = np.zeros((1, 1, n, n))
        for q in range(n):
            if q < start:
                a[0, 0, q, q] = 1
                continue
            j = int(rng.integers(start))
            if q == start:
                a[0, 0, q, j] = 0.8
                x[q] = 0.8 * x[j]
            else:
                a[0, 0, q, j] = 0.6
                a[0, 0, q, q - 1] = 0.2
                x[q] = 0.6 * x[j] + 0.2 * x[q - 1]
            a[0, 0, q, q] = 0.2
            x[q] += rng.normal(scale=0.03, size=d)
        injected = role == "test" and i % 2 == 1
        if injected:
            x[start + 5] += 4  # controlled state corruption, not semantic falsity
        np.savez_compressed(cache / f"{i}.npz", attention=a, hidden=x, hidden_schema=json.dumps({'observer_model': 'synthetic', 'observer_revision': 'v1', 'tokenizer': 'integer', 'tokenizer_revision': 'v1', 'layer': '0', 'state_location': 'synthetic-node'}), response_idx=start,
                            token_ids=np.arange(n), source_id=str(i))
        roster.append(dict(path=f"cache/{i}.npz", response_id=str(i), source_id=str(i), split=role))
        if role == "test":
            annotations.append(dict(id=str(i), token_ids=list(range(start, n)),
                                    offsets=[[2 * j, 2 * j + 1] for j in range(n - start)],
                                    labels=[dict(start=10, end=11)] if injected else []))
    roster_path = root / "roster.json"
    roster_path.write_text(json.dumps(roster, indent=2) + "\n")
    # Scoring completes and freezes before the evaluation annotation is written.
    score_roster(roster_path, root / "run", attribute="hidden")
    label_path = root / "synthetic_labels.jsonl"
    label_path.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    report = evaluate_frozen(root / "run", label_path)
    print(json.dumps({"scope": "synthetic engineering only; not natural hallucination performance",
                      "output": str(root), "test_tokens": report["methods"]["graph"]["all"]["pooled"]["tokens"]}))


if __name__ == "__main__":
    main()
