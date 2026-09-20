"""Read one native head without loading the language model."""

import argparse
import json
from pathlib import Path

import numpy as np

from state_audit.measurements import source_masks
from state_audit.storage import read_arrays, read_json


def inspect_head(root: Path, sample: int, layer: int, head: int, target: int):
    directory = root / "samples" / f"{sample:06d}"
    answer = read_json(directory / "answer.json")
    trace = read_arrays(directory / "trace" / f"layer_{layer:03d}.npz")
    weights = read_arrays(root / "weights" / f"layer_{layer:03d}.npz")["output_projection"]
    heads, _, _ = trace["attention"].shape
    kv_head = head // (heads // trace["value"].shape[0])
    width = trace["value"].shape[-1]
    projection = weights[:, head * width : (head + 1) * width]
    attention = trace["attention"][head, target]
    messages = attention[:, None] * trace["value"][kv_head]
    writes = messages @ projection.T
    np.testing.assert_allclose(
        messages.sum(0), trace["head_readout"][target, head], atol=2e-3, rtol=2e-2
    )
    masks = source_masks(answer, len(attention))
    keys = np.flatnonzero(masks["ordinary"])
    keys = keys[np.argsort(attention[keys])[-5:][::-1]]
    edges = [
        dict(
            key=int(key),
            token=answer["token_strings"][key],
            attention=float(attention[key]),
            write_norm=float(np.linalg.norm(writes[key])),
        )
        for key in keys
    ]
    return dict(
        query=int(trace["queries"][target]),
        target=target,
        layer=layer,
        head=head,
        retained_ordinary_mass=float(attention[masks["ordinary"]].sum()),
        edges=edges,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--target", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_head(args.run, args.sample, args.layer, args.head, args.target),
            ensure_ascii=False,
            indent=2,
        )
    )
