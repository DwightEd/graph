"""The complete teaching workflow, readable from top to bottom."""

from pathlib import Path

from .audit import audit_run
from .storage import read_json


def generate_and_capture(args, full: bool):
    # Lazy imports keep offline auditing usable with only NumPy installed.
    from .capture import capture_run
    from .generation import generate_run
    from .models import load_model

    name = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    model, tokenizer = load_model(name, args.revision, args.device, args.dtype)
    model_settings = dict(
        name=name,
        revision=model.config._commit_hash or args.revision,
        device=args.device,
        dtype=args.dtype,
    )
    options = {
        key: getattr(args, key)
        for key in (
            "mode",
            "template",
            "seed",
            "temperature",
            "top_p",
            "max_new_tokens",
            "max_length",
        )
    }
    generate_run(model, tokenizer, args.data, args.output, options, model_settings, args.resume)
    if full:
        capture_run(model, args.output, args.layers, args.resume)
        del model
        audit_run(args.output, args.output / "audit", audit_settings(args))


def load_run_model(root: Path, device: str | None):
    from .models import load_model

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
