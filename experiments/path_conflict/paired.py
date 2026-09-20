"""Same-question paired head audit: inventory -> cached controls -> finite experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import DEFAULT_SAMPLES, write_json
from .paired_inputs import compile_pair, inventory_pairs
from .paired_plan import choose_paired_heads, head_units, joint_plan, target_head_scores
from .paired_report import report_pairs, review_previous


PROTOCOL = "paired_head_transport_v1"


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Paired experiment changed: {path}; use a new output directory")
    write_json(path, value)


def run_phase(model, probe, heads, pairs, onset_units, identity, directory, args):
    from .paired_trials import (baseline, single_effects, conditional_effects,
        adaptation_effects, carried_effects, save_rows)

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "worlds").mkdir(exist_ok=True)
    units = head_units(heads, probe)
    probe = dict(probe, capture_units=units)
    freeze(directory / "context.json", dict(prefix_ids=probe["prefix_ids"],
        candidates=probe["candidates"], units=units, readout=probe["readout"],
        position=probe["position"], claim_span=probe["claim_span"]))
    full, writes = baseline(model, probe, directory)
    singles, rows, errors = single_effects(model, probe, units, args.doses, full, writes,
                                          directory, args.seed, args.sham_atol)
    save_rows(rows, identity, directory, "effects")
    rows = conditional_effects(model, probe, units, pairs, args.doses, full, singles,
                              errors, directory, args.sham_atol)
    save_rows(rows, identity, directory, "interactions")
    rows = adaptation_effects(model, probe, units, pairs, full, singles, writes,
                             directory, args.mediation_pairs, args.sham_atol)
    save_rows(rows, identity, directory, "adaptation")
    rows = carried_effects(model, probe, onset_units, full, directory, args.sham_atol)
    save_rows(rows, identity, directory, "persistence")
    rows = [dict(unit=unit["unit"], source_tokens=len(unit["sources"]),
                 measured=bool(unit["sources"]), readout=probe["readout"]) for unit in units]
    save_rows(rows, identity, directory, "coverage")


def discover_pair(model, probes, directory, args):
    from .target_gradients import trace_target

    plan_path = directory / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    tables = []
    for side in ("supported", "unsupported"):
        edges, arrays, metadata = trace_target(model, probes[side]["onset"], args.edge_top_k)
        edges.to_csv(directory / (side + "_discovery.csv"), index=False)
        np.savez_compressed(directory / (side + "_discovery.npz"), metadata=json.dumps(metadata), **arrays)
        tables.append(target_head_scores(arrays))
    heads = choose_paired_heads(tables, model.config.num_hidden_layers, model.config.num_attention_heads,
                                args.heads, args.control_heads, args.seed)
    plan = dict(heads=heads, pairs=joint_plan(heads), selection="paired_onset_exploratory")
    write_json(plan_path, plan)
    return plan


def audit_pair(model, tokenizer, samples, case, args):
    from .main import verify_replay

    directory = Path(args.output) / "pairs" / case["case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    probes = compile_pair(args.samples, tokenizer, samples, case)
    freeze(directory / "reviewed_case.json", case)
    for side, phases in probes.items():
        freeze(directory / (side + "_onset.json"), dict(prefix_ids=phases["onset"]["prefix_ids"],
                                                       candidates=phases["onset"]["candidates"]))
    replay_path = directory / "replay.json"
    if not replay_path.exists():
        replay = {side: verify_replay(model, dict(phases["onset"], record_writes=True),
                  args.replay_atol, args.message_rtol)[0] for side, phases in probes.items()}
        write_json(replay_path, replay)
    plan = discover_pair(model, probes, directory, args)
    for side, phases in probes.items():
        onset_units = head_units(plan["heads"], phases["onset"])
        for phase, probe in phases.items():
            identity = dict(case_id=case["case_id"], source_id=case["source_id"], side=side,
                            phase=phase, position=probe["position"], seed=probe["seed"])
            run_phase(model, probe, plan["heads"], plan["pairs"], onset_units, identity,
                      directory / side / phase, args)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "review":
        review_previous(args.review_input, output, args.minimum_effect)
        return
    if args.stage == "report":
        report_pairs(output, args.minimum_effect, args.bootstrap)
        return
    cases = json.loads(Path(args.cases).read_text())
    settings, samples, table = inventory_pairs(args.samples, cases, output)
    config = dict(protocol=PROTOCOL, sampling=settings, cases=cases,
                  settings={key: value for key, value in vars(args).items()
                            if key not in ("stage", "review_input", "cases")})
    freeze(output / "paired_config.json", config)
    print(table[["case_id", "side", "seed", "trace_available", "history_status"]].to_string(index=False))
    if args.stage == "inventory":
        return
    from .paired_screen import screen_pairs
    screen_pairs(args.samples, samples, cases, output, args.vocabulary_size)
    if args.stage == "screen":
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = args.model or settings["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="eager").to(args.device).eval().requires_grad_(False)
    if model.config.model_type != "llama" or model.config.vocab_size != args.vocabulary_size:
        raise ValueError("This audit requires native Llama and the declared vocabulary size")
    for case in tqdm(cases, desc="same-question claim pairs", unit="pair"):
        audit_pair(model, tokenizer, samples, case, args)
    report_pairs(output, args.minimum_effect, args.bootstrap)


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "screen", "run", "report", "review"], default="run")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("paired_cases.json")))
    parser.add_argument("--output", default="outputs/paired_head_transport_v1")
    parser.add_argument("--review-input")
    parser.add_argument("--model", help="Optional relocation of the SAME generator weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--control-heads", type=int, default=1)
    parser.add_argument("--edge-top-k", type=int, default=2)
    parser.add_argument("--mediation-pairs", type=int, default=2)
    parser.add_argument("--doses", type=float, nargs="+", default=[.25, 1.])
    parser.add_argument("--vocabulary-size", type=int, default=128256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sham-atol", type=float, default=1e-5)
    parser.add_argument("--replay-atol", type=float, default=.25)
    parser.add_argument("--message-rtol", type=float, default=.02)
    parser.add_argument("--minimum-effect", type=float, default=.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.stage == "review" and not args.review_input:
        parser.error("--stage review requires the extracted v3 --review-input directory")
    if args.heads < 1 or args.control_heads < 0 or args.mediation_pairs < 0:
        parser.error("heads >= 1; control-heads and mediation-pairs >= 0")
    if 1. not in args.doses or any(not 0 < dose <= 1 for dose in args.doses):
        parser.error("Doses must be in (0,1] and include 1")
    run(args)


if __name__ == "__main__":
    cli()
