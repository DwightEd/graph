"""Reuse a prepared observation run without loading its large attention inputs."""

import hashlib
import json
from pathlib import Path

DATA_FIELDS = (
    "train_cache",
    "test_cache",
    "dataset",
    "index",
    "source_info",
    "tokenizer",
    "tasks",
    "generators",
    "layers",
    "heads",
    "special_token_ids",
    "limit",
)
PATH_FIELDS = ("train_cache", "test_cache", "dataset", "index", "source_info")


def inherit_observations(args):
    source = args.observations_from.resolve()
    if source == args.output.resolve():
        raise ValueError("Reused observations require a different output directory")
    settings = json.loads((source / "settings.json").read_text())
    if settings["observation_mode"] != "ordinary_key_submass":
        raise ValueError("Cross-term audit needs ordinary-key submass observations")
    manifest = source / "observations/manifest.json"
    args.observation_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in DATA_FIELDS:
        value = settings[name]
        setattr(args, name, Path(value) if value is not None and name in PATH_FIELDS else value)
    return settings["excluded_token_ids"]


def link_observations(args):
    source = (args.observations_from / "observations").resolve()
    destination = args.output / "observations"
    if destination.exists():
        if destination.resolve() != source:
            raise ValueError("Existing observations do not match --observations-from")
    else:
        destination.symlink_to(source, target_is_directory=True)
