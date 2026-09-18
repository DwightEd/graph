"""Functional confirmation on held-out RAGTruth positions with the Llama-3.1 observer."""

from pathlib import Path
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.path_conflict.data import token_text_offsets, quote_tokens
from experiments.unsupervised_token_graph.span_audit.inputs import AuditInputs
from experiments.unsupervised_token_graph.span_audit.matching import match_controls

from .native import forward_target, prepare_target


GROUPS = ("source", "other_prompt", "history", "query_self")


def source_strings(record):
    """Flatten source_info text for QA/Summary/Data2txt without assuming one schema."""
    strings = []

    def visit(value):
        if isinstance(value, str) and len(value.strip()) >= 3:
            strings.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record.get("source_info"))
    return strings


def source_token_ids(tokenizer, prompt_ids, source_record):
    text, offsets = token_text_offsets(tokenizer, prompt_ids)
    positions = []
    for string in source_strings(source_record):
        if text.count(string) == 1:
            positions.extend(quote_tokens(text, offsets, string).tolist())
    return np.unique(positions).astype(int)


def target_groups(answer, tokenizer, source_record, position, recent):
    prompt = answer.prompt_length
    prefix_length = prompt + position
    query = prefix_length - 1
    prompt_ids = answer.token_ids[:prompt].tolist()
    source = source_token_ids(tokenizer, prompt_ids, source_record)
    other_prompt = np.setdiff1d(np.arange(prompt, dtype=int), source)
    history = np.arange(prompt, query, dtype=int)
    return dict(
        source=source,
        other_prompt=other_prompt,
        history=history,
        query_self=np.array([query], dtype=int),
        recent_history=np.arange(max(prompt, query - recent), query, dtype=int),
        remote_history=np.arange(prompt, max(prompt, query - recent), dtype=int),
        all_context=np.arange(prefix_length, dtype=int),
    )


def target_rows(pair):
    rows = [
        ("error_onset", pair.error.start),
        ("control_onset", pair.control.start),
    ]
    if pair.error.length > 1:
        offset = max(1, pair.error.length // 2)
        rows.extend([
            ("error_continuation", pair.error.start + offset),
            ("control_continuation", pair.control.start + offset),
        ])
    return rows


def choose_pairs(inputs, args):
    candidates = []
    for response_id in inputs.selected_ids("test", args.tasks):
        answer = inputs.load_answer(response_id)
        pairs = match_controls(
            answer, args.position_gap, args.repetition_gap, args.entropy_gap, args.window
        )
        if pairs:
            candidates.append((answer.source_id, answer, pairs[0]))

    candidates.sort(key=lambda item: (item[0], item[1].response_id))
    selected = []
    seen = set()
    for source_id, answer, pair in candidates:
        if source_id in seen:
            continue
        selected.append((answer, pair))
        seen.add(source_id)
        if len(selected) == args.confirm_pairs:
            break
    return selected


def run_target(model, tokenizer, source_record, answer, role, position, heads, args):
    prefix_ids = answer.token_ids[:answer.prompt_length + position].tolist()
    target = int(answer.response_ids[position])
    groups = target_groups(answer, tokenizer, source_record, position, args.recent_window)
    groups["selected_heads"] = heads
    prepared = prepare_target(model, prefix_ids)
    baseline_logp, baseline = forward_target(model, prepared, target, groups)

    local = {
        (row["layer"], row["head"], row["source_group"]): row
        for row in baseline.local
    }
    rows = []
    for layer, head in heads:
        for group in GROUPS:
            intervention = dict(layer=layer, head=head, source_group=group)
            changed_logp, _ = forward_target(
                model, prepared, target, groups, intervention
            )
            local_row = local[(layer, head, group)]
            final_support = baseline_logp - changed_logp
            rows.append(dict(
                response_id=answer.response_id,
                source_id=answer.source_id,
                task=answer.task,
                generator=answer.generator,
                role=role,
                position=position,
                layer=layer,
                head=head,
                source_group=group,
                attention_mass=local_row["attention_mass"],
                local_support=local_row["local_support"],
                final_support=final_support,
                baseline_logp=baseline_logp,
                changed_logp=changed_logp,
                downstream_reversal=local_row["local_support"] * final_support < 0,
            ))
    del prepared
    target_row = dict(
        response_id=answer.response_id,
        source_id=answer.source_id,
        task=answer.task,
        generator=answer.generator,
        role=role,
        position=position,
        target_id=target,
        source_tokens=len(groups["source"]),
    )
    return rows, target_row


def confirm_dataset(args):
    output = Path(args.output)
    selected = pd.read_csv(output / "selected_heads.csv")
    heads = [
        tuple(row)
        for row in selected[["layer", "head"]].head(args.confirm_heads).to_numpy(int)
    ]

    inputs = AuditInputs(
        args.test_cache, args.dataset, args.index, args.tokenizer, args.feature_root
    )
    pairs = choose_pairs(inputs, args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    rows = []
    targets = []
    completed = set()
    effects_path = output / "functional_effects.csv.gz"
    targets_path = output / "confirm_targets.csv"
    if args.resume and effects_path.exists() and targets_path.exists():
        rows = pd.read_csv(effects_path).to_dict("records")
        targets = pd.read_csv(targets_path).to_dict("records")
        completed = {
            (str(row["response_id"]), str(row["role"]), int(row["position"]))
            for row in targets
        }

    for answer, pair in tqdm(pairs, desc="RAGTruth functional confirm", unit="pair"):
        source_record = inputs.sources[answer.source_id]
        for role, position in target_rows(pair):
            key = (answer.response_id, role, position)
            if key in completed:
                continue
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(model.device)
            result, target_row = run_target(
                model, tokenizer, source_record, answer, role, position, heads, args
            )
            if torch.cuda.is_available():
                target_row["peak_cuda_gib"] = (
                    torch.cuda.max_memory_allocated(model.device) / 1024 ** 3
                )
            rows.extend(result)
            targets.append(target_row)
            completed.add(key)
            pd.DataFrame(rows).to_csv(effects_path, index=False)
            pd.DataFrame(targets).to_csv(targets_path, index=False)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_confirmation_summary(output)


def write_confirmation_summary(output):
    table = pd.read_csv(Path(output) / "functional_effects.csv.gz")
    source = table.groupby(
        ["source_id", "role", "layer", "head", "source_group"], as_index=False
    ).agg(
        final_support=("final_support", "mean"),
        local_support=("local_support", "mean"),
        attention_mass=("attention_mass", "mean"),
        reversal=("downstream_reversal", "mean"),
    )
    summary = source.groupby(
        ["role", "layer", "head", "source_group"], as_index=False
    ).agg(
        final_support=("final_support", "mean"),
        local_support=("local_support", "mean"),
        attention_mass=("attention_mass", "mean"),
        reversal_rate=("reversal", "mean"),
        sources=("source_id", "nunique"),
    )
    summary.to_csv(Path(output) / "functional_summary.csv", index=False)
