"""Workflows only: load → generate → capture; analyses remain separate commands."""

from dataclasses import fields
from pathlib import Path

from .storage import read_json


def generate_and_capture(args, full: bool):
    from .capture import capture_run
    from .generation import GenerationOptions, generate_run
    from .model import load_model

    name = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    model, tokenizer = load_model(name, args.revision, args.device, args.dtype)
    model_settings = dict(
        name=name,
        revision=model.native.config._commit_hash or args.revision,
        device=args.device,
        dtype=args.dtype,
    )
    options = GenerationOptions(
        **{field.name: getattr(args, field.name) for field in fields(GenerationOptions)}
    )
    generate_run(model, tokenizer, args.data, args.output, options, model_settings, args.resume)
    if full:
        spec = capture_spec(args)
        capture_run(model, args.output, spec, args.resume)


def capture_spec(args):
    from .capture import CaptureSpec

    representations = args.representations
    if representations is None:
        representations = CaptureSpec().representations
    layers = None if args.layers is None else tuple(args.layers)
    return CaptureSpec(layers, tuple(representations), args.scope)


def load_run_model(root: Path, device: str | None):
    from .model import load_model

    settings = read_json(root / "run.json")["model"]
    return load_model(
        settings["name"], settings["revision"], device or settings["device"], settings["dtype"]
    )[0]


def audit_settings(args) -> dict:
    return dict(
        window=args.window,
        minimum_mass=args.minimum_mass,
        minimum_rise=args.minimum_rise,
        roles=args.roles,
    )
