"""Command-line entry point for out-of-fold unsupervised graph experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from research_dataset import ResearchDataset
from unsupervised_evaluation import compare_variants, evaluate_records
from unsupervised_experiment import (
    AllDataEvaluator,
    LearnedEmbeddingVisualizer,
    UnsupervisedGraphMethod,
)


class _LimitedDataset:
    """Expose a deterministic prefix of a canonical dataset to the evaluator."""

    def __init__(self, dataset, limit: int):
        self.dataset = dataset
        self.manifest = dataset.manifest
        self.sample_ids = dataset.sample_ids[:limit]

    def __getitem__(self, sample_id):
        return self.dataset[sample_id]

    def labels(self):
        return self.dataset.labels()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run source-grouped out-of-fold unsupervised graph evaluation."
    )
    parser.add_argument("--canonical-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--density-steps", type=int, default=75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def _save_results(records, output_dir: Path):
    embeddings = np.stack([np.asarray(row["embedding"]) for row in records])
    np.savez_compressed(
        output_dir / "results.npz",
        embedding=embeddings,
        score=np.asarray([row["score"] for row in records]),
        nll=np.asarray([row["nll"] for row in records]),
        label=np.asarray([row["label"] for row in records], dtype=np.int64),
        fold=np.asarray([row["fold"] for row in records], dtype=np.int64),
        sample_id=np.asarray([row["sample_id"] for row in records], dtype=str),
        source_id=np.asarray([row["source_id"] for row in records], dtype=str),
        token_index=np.asarray([row["token_index"] for row in records], dtype=np.int64),
        task_type=np.asarray([row["task_type"] for row in records], dtype=str),
        data_source=np.asarray([row["data_source"] for row in records], dtype=str),
        generator_model=np.asarray([row["generator_model"] for row in records], dtype=str),
    )


def _run_variant(
    dataset, args, *, num_channels, output_name, graph_variant, message_steps, output_dir
):
    evaluator = AllDataEvaluator(dataset, folds=args.folds, seed=args.seed)
    visualizer = LearnedEmbeddingVisualizer(random_state=args.seed)
    visualization_fold = args.seed % args.folds
    visualization_train = None

    with tqdm(total=args.folds, desc=f"{output_name} folds", unit="fold") as progress:
        def fit_fold(train_samples, heldout_samples, fold):
            nonlocal visualization_train
            method = UnsupervisedGraphMethod(
                num_channels=num_channels,
                embedding_dim=args.embedding_dim,
                message_passing_steps=message_steps,
                graph_variant=graph_variant,
                epochs=args.epochs,
                fit_steps=args.density_steps,
                seed=args.seed + fold,
            )
            method.fit(train_samples, progress=True)
            if fold == visualization_fold:
                visualization_train = np.concatenate(
                    list(method.embed(train_samples).values()), axis=0
                )
            outputs = method.score(heldout_samples)
            progress.update(1)
            return outputs

        records = evaluator.run(fit_fold)
    evaluated = evaluator.evaluate(records)
    variant_dir = output_dir / output_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    _save_results(evaluated, variant_dir)
    report = evaluate_records(evaluated, seed=args.seed)
    report.save(variant_dir)
    visualization_records = [
        row for row in evaluated if row["fold"] == visualization_fold
    ]
    projection = visualizer.plot_fold(
        visualization_train,
        visualization_records,
        variant_dir / f"embedding_fold_{visualization_fold}.png",
    )
    np.savez_compressed(
        variant_dir / f"embedding_fold_{visualization_fold}.npz",
        coordinates=projection["coordinates"],
        sample_id=np.asarray([row["sample_id"] for row in visualization_records], dtype=str),
        token_index=np.asarray([row["token_index"] for row in visualization_records]),
        score=np.asarray([row["score"] for row in visualization_records]),
        label=np.asarray([row["label"] for row in visualization_records]),
    )
    return evaluated, report


def main(argv=None):
    args = parse_args(argv)
    dataset = ResearchDataset(args.canonical_split, device=args.device)
    if args.limit is not None:
        dataset = _LimitedDataset(dataset, args.limit)
    source_count = len({dataset[sample_id].source_id for sample_id in dataset.sample_ids})
    minimum_sources = args.folds + 1
    if source_count < minimum_sources:
        raise ValueError(
            f"at least {minimum_sources} source groups are required for OOF evaluation "
            "and fold-local calibration"
        )
    num_channels = int(dataset.manifest["num_layers"]) * int(dataset.manifest["num_heads"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = str(dataset.manifest.get("split", "")).lower()
    fit_scope_warning = (
        "This command fits its unsupervised models on each OOF training fold of the "
        f"provided {split or 'unspecified'} split. Pass the intended analysis population "
        "explicitly; running a test archive is transductive evaluation."
    )
    reports = {}
    variant_records = {}
    evaluated = None
    for name, steps in (("full", args.message_steps), ("no_message", 0),
                        ("rewired", args.message_steps), ("channel_mean", args.message_steps)):
        records, report = _run_variant(
            dataset,
            args,
            num_channels=num_channels,
            output_name=name,
            graph_variant="full" if name == "no_message" else name,
            message_steps=steps,
            output_dir=output_dir,
        )
        reports[name] = report
        variant_records[name] = records
        if name == "full":
            evaluated = records
    report = reports["full"]
    summary = {
        "canonical_split": str(Path(args.canonical_split).resolve()),
        "records": len(evaluated),
        "positive_labels": int(sum(row["label"] for row in evaluated)),
        "token_metrics": report.metrics["token"]["overall"],
        "answer_metrics": report.metrics["answer"]["overall"],
        "folds": args.folds,
        "embedding_dim": args.embedding_dim,
        "message_steps": args.message_steps,
        "epochs": args.epochs,
        "density_steps": args.density_steps,
        "seed": args.seed,
        "limit": args.limit,
        "fit_scope_warning": fit_scope_warning,
        "variants": {
            name: {
                "token": value.metrics["token"]["overall"],
                "answer": value.metrics["answer"]["overall"],
            }
            for name, value in reports.items()
        },
        "full_minus_ablation": compare_variants(variant_records, seed=args.seed),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
