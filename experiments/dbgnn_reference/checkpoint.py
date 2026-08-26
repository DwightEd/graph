"""Rewrite a locally produced DBGNN checkpoint into weights-only-safe values."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch


def plain_value(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return plain_value(value.tolist())
    if isinstance(value, dict):
        return {plain_value(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(plain_value(item) for item in value)
    if isinstance(value, list):
        return [plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def make_portable(path) -> None:
    """Convert NumPy scalar metadata while preserving tensors and model state."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = plain_value(payload)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".pt",
        delete=False,
    ) as file:
        temporary = Path(file.name)

    torch.save(payload, temporary)
    temporary.replace(path)

    # The next pipeline stage uses this exact loading mode.
    torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a locally generated DBGNN checkpoint loadable with weights_only=True"
    )
    parser.add_argument("--path", required=True)
    arguments = parser.parse_args()
    make_portable(arguments.path)
    print(f"checkpoint ready: {arguments.path}")


if __name__ == "__main__":
    main()
