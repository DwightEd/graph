"""Immutable raw-value artifacts for the measured native graph."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from route_graph.frozen_reader import digest, write_json_once


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def donor_writer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def save(captures, branch, scale):
        metadata = {
            "branch": branch,
            "input_scale": scale,
            "layers": {
                str(layer): {
                    "keys": capture.keys,
                    "input_ids": capture.input_ids,
                    "dtype": str(capture.values.dtype),
                    "shape": list(capture.values.shape),
                    "values_sha256": capture.values_sha256,
                }
                for layer, capture in captures.items()
            },
        }
        name = digest(metadata)
        destination = directory / f"{name}.npz"
        manifest = directory / f"{name}.json"
        arrays = {
            str(layer): capture.values.contiguous().view(torch.uint8).numpy()
            for layer, capture in captures.items()
        }
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(
                prefix=name, suffix=".partial", dir=directory
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.link(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        # Validate existing or newly saved arrays against native byte digests.
        with np.load(destination, allow_pickle=False) as saved:
            for layer, capture in captures.items():
                if (
                    hashlib.sha256(saved[str(layer)].tobytes()).hexdigest()
                    != capture.values_sha256
                ):
                    raise ValueError("donor artifact differs from actual native values")
        record = {
            **metadata,
            "npz_sha256": file_sha256(destination),
            "storage": "raw_uint8_native_dtype_shape_in_manifest",
        }
        if not manifest.exists():
            write_json_once(manifest, record)
        elif digest(json.loads(manifest.read_text())) != digest(record):
            raise ValueError("donor manifest differs from actual native capture")
        return {
            "manifest": str(manifest),
            "values": str(destination),
            "npz_sha256": record["npz_sha256"],
            "manifest_sha256": file_sha256(manifest),
        }

    return save


def verify_donor_artifact(ref, root):
    manifest, values = Path(ref["manifest"]).resolve(), Path(ref["values"]).resolve()
    if not manifest.is_relative_to(root.resolve()) or not values.is_relative_to(
        root.resolve()
    ):
        raise ValueError("donor artifact escapes the current run")
    if (
        file_sha256(manifest) != ref["manifest_sha256"]
        or file_sha256(values) != ref["npz_sha256"]
    ):
        raise ValueError("donor artifact checksum mismatch")
    metadata = json.loads(manifest.read_text())
    if metadata["npz_sha256"] != ref["npz_sha256"]:
        raise ValueError("donor manifest and values disagree")
    with np.load(values, allow_pickle=False) as arrays:
        if set(arrays.files) != set(metadata["layers"]):
            raise ValueError("donor layer set mismatch")
        for layer, info in metadata["layers"].items():
            if (
                hashlib.sha256(arrays[layer].tobytes()).hexdigest()
                != info["values_sha256"]
            ):
                raise ValueError("donor layer byte digest mismatch")
