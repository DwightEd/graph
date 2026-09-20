"""Independent full-input block permutations on explicit entity/color facts.

This is a transfer check for local routing preferences. It never updates the
natural TRAIN prior or uses RAGTruth labels. Unequal token lengths are retained.
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ..offline_span.data import write_json
from .metrics import swap_scores
from .profile import load_model, write_csv


NAMES = ("Alice", "Bob", "Carol", "David", "Elena", "Farah", "George", "Helen")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")


def binding_input(tokenizer, order, target):
    blocks = [f"\n{NAMES[index]} likes the color {COLORS[index]}." for index in order]
    content = "Use only these statements." + "".join(blocks)
    content += f"\nWhat color does {NAMES[target]} like? Answer with the color only."
    text = tokenizer.apply_chat_template([dict(role="user", content=content)],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = np.asarray(encoded["offset_mapping"])
    ranges, cursor = [], 0
    for block in blocks:
        start = text.index(block, cursor)
        end = start + len(block)
        indices = np.flatnonzero((offsets[:, 1] > start) & (offsets[:, 0] < end))
        ranges.append((int(indices[0]), int(indices[-1]) + 1))
        cursor = end
    ranges = np.asarray(ranges)
    if np.any(ranges[1:, 0] < ranges[:-1, 1]):
        raise ValueError("Tokenizer straddles block boundaries; use explicit disjoint blocks")
    return np.asarray(encoded["input_ids"]), ranges


def capture_blocks(model, token_ids, blocks, excluded):
    ordinary = ~np.isin(token_ids, excluded)
    records, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        def hook(module, inputs, output, layer_index=index):
            row = output[1][0, :, -1].float()
            valid = torch.as_tensor(ordinary, device=row.device)
            row = row * valid
            row = row / row.sum(-1, keepdim=True)
            means = [row[:, start:end].mean(-1) for start, end in blocks]
            records[layer_index] = torch.stack(means, -1).cpu().numpy()
        handles.append(layer.self_attn.register_forward_hook(hook))
    ids = torch.as_tensor(np.asarray(token_ids)[None], device=next(model.parameters()).device)
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return records


def run_binding(args, excluded):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    model = load_model(args)
    root = args.output / "binding"
    root.mkdir(exist_ok=True)
    random = np.random.default_rng(args.seed)
    candidates = np.column_stack(np.triu_indices(len(NAMES), 1))
    chosen = random.choice(len(candidates), min(args.swaps, len(candidates)), replace=False)
    pairs = candidates[chosen]
    rows = []
    for target in tqdm((0, 3, 7), desc="full-input binding permutations"):
        path = root / f"target_{target}.npz"
        if not (args.resume and path.exists()):
            arrays = binding_worlds(model, tokenizer, target, pairs, excluded)
            partial = path.with_suffix(".partial.npz")
            np.savez_compressed(partial, **arrays)
            partial.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            for layer in range(model.config.num_hidden_layers):
                values = swap_scores(saved[f"L{layer}__before"], saved[f"L{layer}__after"], args.temperature)
                for head in range(model.config.num_attention_heads):
                    rows.append(dict(target_block=target, layer=layer, head=head,
                                     **{name: float(value[head]) for name, value in values.items()}))
    write_csv(root / "heads.csv", rows)
    compare_prior(args, rows)
    write_json(root / "summary.json", dict(mode="full_input_fact_block_permutations",
        targets=[0, 3, 7], swaps=pairs.tolist(), facts=list(zip(NAMES, COLORS)),
        labels_used=False, prior_updated=False, changed_query_and_contextual_states=True))
    return rows


def compare_prior(args, rows):
    path = args.output / "profile/priors.npz"
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as saved:
        gaps = np.nanmean(saved["source_gap"], axis=0)
        priors = {tuple(channel): gap for channel, gap in zip(saved["channels"], gaps)}
    comparisons = []
    for channel, gap in priors.items():
        observed = [row["gap"] for row in rows if (row["layer"], row["head"]) == channel]
        transfer = float(np.mean(observed))
        comparisons.append(dict(layer=channel[0], head=channel[1], natural_local_gap=gap,
            binding_full_input_gap=transfer, same_direction=bool(gap * transfer > 0),
            both_clear=bool(abs(gap) > .05 and abs(transfer) > .05)))
    write_csv(args.output / "binding/transfer.csv", comparisons)


def binding_worlds(model, tokenizer, target, pairs, excluded):
    token_ids, blocks = binding_input(tokenizer, np.arange(len(NAMES)), target)
    baseline = capture_blocks(model, token_ids, blocks, excluded)
    before, after = {}, {layer: [] for layer in baseline}
    for first, second in pairs:
        order = np.arange(len(NAMES))
        order[first], order[second] = order[second], order[first]
        changed_ids, changed_blocks = binding_input(tokenizer, order, target)
        changed = capture_blocks(model, changed_ids, changed_blocks, excluded)
        for layer in baseline:
            after[layer].append(changed[layer][:, [first, second]])
    arrays = dict(pairs=pairs, target=target, token_ids=token_ids, blocks=blocks)
    for layer, values in baseline.items():
        before[layer] = np.stack([values[:, pair] for pair in pairs])
        arrays[f"L{layer}__before"] = before[layer]
        arrays[f"L{layer}__after"] = np.stack(after[layer])
    return arrays
