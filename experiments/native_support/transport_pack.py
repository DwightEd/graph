"""Pack existing value_transport captures for a head/time audit; no model runs.

All heads and signed candidate effects are retained. Dense attention/value-energy
arrays are replaced by exact source-group totals and top-8 historical addresses.
The latter are a partial edge view; retained mass is saved explicitly.
"""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def grouped_reads(attention, energy, groups, group_count):
    # Keep predictor self separate from ordinary source/history addresses.
    read_groups = groups.copy()
    read_groups[-1] = group_count
    masks = [read_groups == group for group in range(group_count + 1)]
    magnitude = np.sqrt(np.maximum(energy, 0))
    return {
        "head_group_attention": np.stack(
            [attention[..., mask].sum(-1) for mask in masks], axis=-1,
        ),
        "head_group_message_norm": np.stack(
            [magnitude[..., mask].sum(-1) for mask in masks], axis=-1,
        ),
    }


def history_addresses(attention, prompt, query):
    history = attention[..., prompt:query]  # Actual history keys; exclude self.
    count = min(8, history.shape[-1])
    if count:
        index = np.argpartition(history, -count, axis=-1)[..., -count:]
        mass = np.take_along_axis(history, index, axis=-1)
        order = np.argsort(-mass, axis=-1)
        index = np.take_along_axis(index, order, axis=-1)
        mass = np.take_along_axis(mass, order, axis=-1)
    else:
        index = np.empty((*attention.shape[:-1], 0), dtype=np.int32)
        mass = np.empty(index.shape, dtype=attention.dtype)
    return {
        "history_top_key": (index + prompt).astype(np.int32),
        "history_top_attention": mass,
        "history_total_attention": history.sum(-1),
        "history_top_retained_attention": mass.sum(-1),
    }


def compact_arrays(saved, prompt):
    """Retain signed actions; only dense addresses/energy need compaction."""
    packed = {name: saved[name] for name in saved
              if name not in ("attention", "edge_value_energy")}
    if "attention" in saved:
        attention = saved["attention"]
        packed.update(grouped_reads(attention, saved["edge_value_energy"],
                                    saved["group_ids"], len(saved["root_positive"])))
        packed.update(history_addresses(attention, prompt, int(saved["query"])))
    return packed


def compact_capture(path, prompt):
    with np.load(path, allow_pickle=False) as saved:
        packed = compact_arrays(saved, prompt)
    buffer = io.BytesIO()
    np.savez(buffer, **packed)
    return buffer.getvalue()


def pack_schema():
    return {
        "purpose": "head_and_temporal_audit_of_existing_value_transport",
        "model_run": False, "scores_changed": False, "heads_filtered": False,
        "head_group_axes": ["layer", "head", "source_blocks+history+other+special+self"],
        "root_and_signed_head_groups": "unchanged: source_blocks+history+other+special",
        "history_top_key": "absolute input key; answer token index = key - prompt_length",
        "query_alignment": "P+t-1; predictor self excluded from history edge tables",
        "dense_edges": "omitted; group totals exact, top-8 addresses incomplete",
        "candidate_alignment": "IDs vary by target; candidate rank is not a stable semantic axis",
    }


def copy_metadata(output, packed):
    for name in ("settings.json", "annotations.json"):
        packed.write(output / name, name)
    value = output / "value_transport"
    for path in sorted(value.rglob("*")):
        relative = path.relative_to(value)
        metadata = path.suffix in (".json", ".csv")
        scores = relative.parts[0] == "responses" and path.suffix == ".npz"
        if path.is_file() and (metadata or scores):
            packed.write(path, str(path.relative_to(output)))


def pack_response(output, packed, index, response):
    directory = output / "value_transport" / "capture" / f"{index:04d}"
    total = len(response["token_ids"]) - response["prompt_length"]
    for target in range(total):
        path = directory / f"token_{target:06d}.npz"
        content = compact_capture(path, response["prompt_length"])
        packed.writestr(str(path.relative_to(output)), content)
        if (target + 1) % 25 == 0 or target + 1 == total:
            print(f"\r{response['id']}: {target + 1}/{total}", end="", flush=True)
    print()
    return total


def pack(output, archive):
    settings = json.loads((output / "settings.json").read_text())
    schema = pack_schema()
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as packed:
        copy_metadata(output, packed)
        count = 0
        for index, response in enumerate(settings["responses"]):
            count += pack_response(output, packed, index, response)
        schema["captured_tokens"] = count
        packed.writestr("audit_pack_schema.json", json.dumps(schema, indent=2))
    print(f"{archive.resolve()} ({archive.stat().st_size / 2**20:.1f} MiB; {count} tokens)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("value_transport_head_review.zip"))
    args = parser.parse_args(argv)
    pack(args.output, args.archive)


if __name__ == "__main__":
    main()
