"""Discover routing preferences on reference TRAIN sources only; freeze head masks."""

import csv
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..fixed_graph.reference import split_sources
from ..head_geometry.pipeline import read_json
from ..offline_span.data import write_json
from .blocks import ordinary_blocks, sample_swaps
from .metrics import choose_masks
from .native import capture


def choose_records(records, roles, budget, seed):
    sources = defaultdict(list)
    for row in records:
        if row["split"] == "train" and roles[row["source_id"]] == "reference":
            sources[row["source_id"]].append(row)
    random = np.random.default_rng(seed)
    names = sorted(sources)
    random.shuffle(names)
    return [sources[source][0] for source in names[:budget]]


def query_plan(saved, args, excluded):
    targets = np.flatnonzero(saved["coverage"])
    chosen = np.unique(np.linspace(0, len(targets) - 1, min(len(targets), args.probe_queries)).astype(int))
    positions, swaps = [], []
    for target in targets[chosen]:
        query = int(saved["prompt_length"] + target - 1)
        blocks = ordinary_blocks(saved["token_ids"], query, excluded, args.block_width)
        pairs = sample_swaps(blocks, args.swaps, [args.seed, query])
        if len(pairs):
            positions.append(query)
            swaps.append(pairs)
    return positions, swaps


def save_capture(path, records, positions, swaps):
    arrays = dict(positions=np.asarray(positions))
    arrays.update({f"swaps_{index}": pairs for index, pairs in enumerate(swaps)})
    for layer, rows in records.items():
        for key in rows[0]["scores"]:
            arrays[f"L{layer}__{key}"] = np.stack([row["scores"][key] for row in rows])
        for index, row in enumerate(rows):
            arrays[f"L{layer}__before_{index}"] = row["before"]
            arrays[f"L{layer}__after_{index}"] = row["after"]
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval().requires_grad_(False)


def profile(args, excluded):
    manifest = read_json(args.output / "observations/manifest.json")
    roles = split_sources(manifest["records"], args.seed)
    selected = choose_records(manifest["records"], roles, args.probe_sources, args.seed)
    root = args.output / "profile"
    root.mkdir(exist_ok=True)
    measured, skipped, model = [], [], None
    for number, row in enumerate(tqdm(selected, desc="TRAIN key-content swaps")):
        path = root / f"{number:04d}.npz"
        with np.load(args.output / "observations" / row["file"], allow_pickle=False) as saved:
            positions, swaps = query_plan(saved, args, excluded)
            token_ids = saved["token_ids"][:max(positions) + 1] if positions else []
        if not positions:
            skipped.append(row["id"])
            continue
        if not (args.resume and path.exists()):
            if model is None:
                model = load_model(args)
            records = capture(model, token_ids, positions, swaps, excluded, args.temperature)
            save_capture(path, records, positions, swaps)
        measured.append(dict(row, profile_file=path.name))
    del model
    write_json(root / "manifest.json", dict(records=measured, skipped=skipped, roles=roles,
               mode="frozen_layer_key_content_swaps", labels_used=False, complete=True))
    return summarize(args)


def head_summary(args, manifest, channels):
    keys = ("positional", "symbolic", "gap", "contrast", "swapped_mass", "reconstruction_error")
    source_values = defaultdict(lambda: defaultdict(list))
    for row in manifest["records"]:
        with np.load(args.output / "profile" / row["profile_file"], allow_pickle=False) as saved:
            for key in keys:
                reduce = np.nanmax if key == "reconstruction_error" else np.nanmean
                values = [reduce(saved[f"L{layer}__{key}"][:, head]) for layer, head in channels]
                source_values[row["source_id"]][key].append(values)
    if not source_values:
        raise ValueError("No eligible reference source: ordinary blocks are required")
    result = {key: [] for key in keys}
    for measurements in source_values.values():
        for key in keys:
            result[key].append(np.nanmean(measurements[key], axis=0))
    arrays = {key: np.asarray(values) for key, values in result.items()}
    return arrays


def summarize(args):
    manifest = read_json(args.output / "profile/manifest.json")
    observations = read_json(args.output / "observations/manifest.json")
    channels = np.asarray(observations["channels"])
    values = head_summary(args, manifest, channels)
    means = {key: np.nanmean(value, axis=0) for key, value in values.items()}
    error = float(np.nanmax(values["reconstruction_error"]))
    if error > args.reconstruction_atol:
        raise ValueError(f"Native Q/K reconstruction failed: {error:g}")
    valid_sources = np.isfinite(values["gap"]).sum(0)
    agreement = np.mean(np.sign(values["gap"]) == np.sign(means["gap"]), axis=0)
    identifiable = ((valid_sources >= 2) & (means["contrast"] >= .1)
                    & (means["swapped_mass"] >= .01) & (agreement >= .75)
                    & np.isfinite(means["gap"]))
    layers, heads = np.unique(channels[:, 0]), np.unique(channels[:, 1])
    masks = choose_masks(layers, heads, means["gap"], identifiable,
                         args.drop_fraction, args.seed, args.random_controls)
    np.savez_compressed(args.output / "profile/priors.npz", channels=channels, **masks,
                        **{f"source_{key}": value for key, value in values.items()})
    rows = []
    for index, (layer, head) in enumerate(channels):
        rows.append(dict(layer=layer, head=head, identifiable=identifiable[index],
            sources=valid_sources[index], source_direction_agreement=agreement[index],
            **{key: value[index] for key, value in means.items()},
            **{name: bool(mask[index]) for name, mask in masks.items()}))
    write_csv(args.output / "profile/heads.csv", rows)
    summary = dict(sources=len(values["gap"]), heads=len(channels), max_reconstruction_error=error,
        identifiable_heads=int(identifiable.sum()), removed={name: int((~mask).sum()) for name, mask in masks.items()},
        labels_used=False, interpretation="routing preference, not semantic correctness or causal importance")
    write_json(args.output / "profile/summary.json", summary)
    print(json.dumps(summary), flush=True)
    plot_profile(args.output / "profile", channels, means)
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_profile(root, channels, means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers, heads = np.unique(channels[:, 0]), np.unique(channels[:, 1])
    figure, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    for axis, key in zip(axes, ("positional", "symbolic", "gap")):
        matrix = means[key].reshape(len(layers), len(heads))
        options = dict(cmap="coolwarm", vmin=-1, vmax=1) if key == "gap" else dict(vmin=0, vmax=1)
        shown = axis.imshow(matrix, aspect="auto", **options)
        axis.set(title=key, xlabel="physical head", ylabel="physical layer")
        axis.set_xticks(np.arange(len(heads)), heads, fontsize=6)
        axis.set_yticks(np.arange(len(layers)), layers, fontsize=6)
        figure.colorbar(shown, ax=axis, shrink=.7)
    figure.savefig(root / "head_roles.png", dpi=160)
    plt.close(figure)
