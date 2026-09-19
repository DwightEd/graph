"""Unified evidence-to-target and target-to-source functional audit."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import DEFAULT_SAMPLES, inventory, compile_cases, write_json
from .focused_inputs import candidate_panels, source_rows
from .flow_inputs import add_flow_groups
from .flow_plan import HeadTrial, select_heads, SCAN_GROUPS
from .native import Intervention
from .scoring import evaluate_candidates
from .cooperation import select_pairs, run_pairs, save_pairs


PROTOCOL = "evidence_target_flow_v2"
WRITE_IDENTITY = ("case_id", "source_id", "side", "panel", "seed", "trace", "query")


def attach_identity(frame, identity):
    for key in WRITE_IDENTITY:
        frame[key] = identity[key]
    return frame


def save_capture(path, identity, variant, score, run, trial=None):
    from .main import save_run

    record = dict(identity, variant=variant, **score)
    if trial is not None:
        record.update(layer=trial.layer, head=trial.head, source_group=trial.source_group)
    save_run(path, record, run.trajectory, run.writes, run.changes)


def baseline_panel(model, tokenizer, case, side, probe, output):
    identity = dict(
        case_id=case["case_id"],
        source_id=case["source_id"],
        side=side,
        panel=probe["panel"],
        seed=probe["seed"],
        trace=probe["trace"],
        query=len(probe["prefix_ids"]) - 1,
        candidates=probe["candidate_texts"],
    )
    score, run = evaluate_candidates(model, probe)
    stem = "_".join([case["case_id"], side, probe["panel"]])
    path = output / "baseline" / (stem + ".npz")
    if not path.exists():
        save_capture(path, identity, "full", score, run)

    writes = attach_identity(pd.DataFrame(run.writes), identity)
    return identity, score, writes, source_rows(tokenizer, probe, identity)


def final_head_trials(model, probe, identity, baseline_score, heads, output):
    rows = []
    trajectories = []
    stem = "_".join([identity["case_id"], identity["side"], identity["panel"]])

    for layer, head in tqdm(heads, desc=stem + " heads", leave=False):
        for group in SCAN_GROUPS:
            trial = HeadTrial(layer, head, group)
            path = output / "runs" / (stem + "_" + trial.name + ".npz")
            action = Intervention(layer, (group,), "query", (head,))

            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    record = json.loads(str(saved["record"]))
                    trajectory = json.loads(str(saved["trajectory"]))
            else:
                score, run = evaluate_candidates(model, probe, (action,))
                save_capture(path, identity, trial.name, score, run, trial)
                record = dict(
                    identity,
                    variant=trial.name,
                    layer=layer,
                    head=head,
                    source_group=group,
                    **score,
                )
                trajectory = run.trajectory

            record["final_support"] = baseline_score["next_margin"] - record["next_margin"]
            record["final_sequence_support"] = baseline_score["sequence_margin"] - record["sequence_margin"]
            rows.append(record)
            trajectories.extend(
                dict(
                    identity,
                    variant=trial.name,
                    layer_cut=layer,
                    head_cut=head,
                    source_group=group,
                    **step,
                )
                for step in trajectory
            )
    return rows, trajectories


def setup(args, output):
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    inventory_dir = output / "inventory"
    inventory_dir.mkdir(exist_ok=True)
    settings, samples, prompts = inventory(args.samples, inventory_dir, cases)
    config = dict(
        protocol=PROTOCOL,
        samples=args.samples,
        actual_model=args.model or settings["model"],
        dtype=args.dtype,
        recent_window=args.recent_window,
        top_k=args.flow_top_k,
        max_heads=args.flow_max_heads,
        pairs_per_group=args.pairs_per_group,
        cases=cases,
        sampling_settings=settings,
    )
    config_path = output / "flow_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Flow protocol/settings changed; use a new output directory")
    write_json(config_path, config)
    return cases, samples, prompts, config


def run_flow(args):
    from .flow_report import report_flow

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report_flow(output, args.supervised_head_roles)
        return

    cases, samples, prompts, config = setup(args, output)
    if args.stage == "inventory":
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .main import verify_replay

    tokenizer = AutoTokenizer.from_pretrained(
        config["actual_model"], local_files_only=True, use_fast=True
    )
    compiled = compile_cases(args.samples, tokenizer, samples, prompts, cases, output)
    model = AutoModelForCausalLM.from_pretrained(
        config["actual_model"],
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    for name in ("baseline", "runs"):
        (output / name).mkdir(exist_ok=True)

    all_writes = []
    all_sources = []
    probes = []

    for entry in compiled:
        case = entry["case"]
        for side, native in entry["sides"].items():
            verify_replay(model, native, args.replay_atol, args.message_rtol)
            for original in candidate_panels(tokenizer, native, case, args.recent_window):
                probe = add_flow_groups(original)
                identity, score, writes, sources = baseline_panel(
                    model, tokenizer, case, side, probe, output
                )
                all_writes.append(writes)
                all_sources.extend(sources)
                probes.append((identity, score, probe))

    writes = pd.concat(all_writes, ignore_index=True)
    writes.to_csv(output / "baseline_head_sources.csv.gz", index=False)
    pd.DataFrame(all_sources).to_csv(output / "source_tokens.csv.gz", index=False)

    heads = select_heads(writes, args.flow_top_k, args.flow_max_heads)
    write_json(output / "selected_heads.json", [dict(layer=l, head=h) for l, h in heads])

    effects = []
    trajectories = []
    pair_rows = []
    for identity, score, probe in probes:
        panel_writes = writes
        for field in ("case_id", "side", "panel"):
            panel_writes = panel_writes[panel_writes[field] == identity[field]]
        pairs = select_pairs(panel_writes, heads, args.pairs_per_group)
        stem = "_".join([identity["case_id"], identity["side"], identity["panel"]])
        write_json(output / (stem + "_pair_plan.json"), pairs)
        rows, paths = final_head_trials(model, probe, identity, score, heads, output)
        effects.extend(rows)
        trajectories.extend(paths)
        pair_rows.extend(run_pairs(model, probe, identity, score, pairs, output))

    pd.DataFrame(effects).to_csv(output / "head_interventions.csv", index=False)
    pd.DataFrame(trajectories).to_csv(output / "propagation.csv.gz", index=False)
    save_pairs(pair_rows, output)
    report_flow(output, args.supervised_head_roles)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "run", "report"], default="run")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    parser.add_argument("--output", default="outputs/evidence_target_flow_v2")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--flow-top-k", type=int, default=3)
    parser.add_argument("--flow-max-heads", type=int, default=12)
    parser.add_argument("--pairs-per-group", type=int, default=2)
    parser.add_argument("--supervised-head-roles")
    parser.add_argument("--replay-atol", type=float, default=.25)
    parser.add_argument("--message-rtol", type=float, default=.02)
    args = parser.parse_args(argv)
    run_flow(args)


if __name__ == "__main__":
    cli()
