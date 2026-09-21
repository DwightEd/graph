"""Reuse each model across datasets. Give each model/dataset its own output directory."""

import argparse
from pathlib import Path

from state_audit.capture import CaptureSpec, capture_run
from state_audit.generation import GenerationOptions, generate_run
from state_audit.model import load_model


def run(args):
    options = GenerationOptions(samples=args.samples, temperature=0.8, max_length=args.max_length)
    for model_index, checkpoint in enumerate(args.models):
        model, tokenizer = load_model(checkpoint, device=args.device, dtype=args.dtype)
        settings = dict(
            name=checkpoint,
            revision=model.native.config._commit_hash or "main",
            device=args.device,
            dtype=args.dtype,
        )
        for data_index, data in enumerate(args.data):
            output = args.output / f"model_{model_index}" / f"dataset_{data_index}"
            generate_run(model, tokenizer, data, output, options, settings, args.resume)
            capture_run(
                model, output, CaptureSpec(representations=("residual_after",)), args.resume
            )
        del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
