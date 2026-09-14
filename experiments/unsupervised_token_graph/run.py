"""Prepare, fit, score, and post-freeze evaluate token graphs."""

import argparse
import hashlib
import json
from pathlib import Path

from .config import Config
from .data import CacheDataset
from .evaluate import evaluate_predictions
from .graph import TokenGraph
from .model import GraphAutoencoder
from .score import calibrate_source_budget, score_graph


def split_sources(records, fraction):
    sources = sorted({record.source_id for record in records}, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    cut = max(1, min(len(sources) - 1, int(len(sources) * fraction)))
    reference = set(sources[:cut])
    return [{"response_id": record.response_id, "source_id": record.source_id,
             "split": "reference" if record.source_id in reference else "calibration"}
            for record in records]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--phase", choices=("fit", "score", "evaluate"), default="fit")
    parser.add_argument("--attention-floor", type=float, default=Config.attention_floor)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = list(CacheDataset(args.cache))
    if not records:
        raise FileNotFoundError(
            f"no .npz attention caches found under {args.cache}; "
            "expected compressed files with attention/prompt_length or formal CSR fields"
        )
    graphs = [TokenGraph.from_cache(record, args.attention_floor) for record in records]
    config = Config(attention_floor=args.attention_floor, epochs=args.epochs)
    splits = split_sources(records, config.reference_fraction)
    model_path = output / "model.pt"
    if args.phase == "fit":
        reference = [graph for graph, split in zip(graphs, splits) if split["split"] == "reference"]
        model = GraphAutoencoder(reference[0].x.shape[1], epochs=config.epochs, seed=config.seed)
        history = model.fit(reference)
        import torch
        torch.save(model, model_path)
        (output / "fit.json").write_text(json.dumps({"loss": history, "responses": len(reference), "splits": splits}), encoding="utf-8")
        return
    import torch
    model = torch.load(model_path, weights_only=False)
    predictions = [score_graph(model, graph) for graph in graphs]
    (output / "scores.json").write_text(json.dumps(predictions, default=lambda value: value.tolist()), encoding="utf-8")
    if args.phase == "score":
        calibration_records = [split for split in splits if split["split"] == "calibration"]
        calibration_predictions = [prediction for prediction, split in zip(predictions, splits) if split["split"] == "calibration"]
        calibration = calibrate_source_budget(calibration_predictions, calibration_records, config.alarm_budget)
        (output / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    else:
        result = evaluate_predictions(predictions, args.annotations)
        (output / "evaluation.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
