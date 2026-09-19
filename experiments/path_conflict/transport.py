"""Target-first native transport: discover messages, test interactions, trace repair."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import DEFAULT_SAMPLES, compile_cases, inventory, write_json
from .focused_inputs import candidate_panels
from .target_gradients import trace_target
from .transport_plan import select_units, select_message_pairs
from .transport_trials import baseline_world, single_worlds, joint_worlds, mediation_worlds


PROTOCOL = "target_transport_v3"


def freeze_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Transport inputs/settings changed: {path}. Use a new output directory.")
    write_json(path, value)


def discover(model, probe, directory, top_k, objective):
    path = directory / "discovery.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
        return pd.read_csv(directory / "discovery_edges.csv", float_precision="round_trip"), metadata
    edges, arrays, metadata = trace_target(model, probe, top_k, objective=objective)
    edges["linear_metric"] = objective
    edges.to_csv(directory / "discovery_edges.csv", index=False)
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
    return edges, metadata


def describe_edges(edges, tokenizer, probe):
    """Attach text and reviewed roles AFTER discovery; roles never select units."""
    edges = edges.copy()
    ids = probe["prefix_ids"]
    primitive = ("scope", "supported_value", "value_source", "other_prompt",
                 "query_self", "recent_history", "remote_history")
    labels = {index: [] for index in range(len(ids))}
    for name in primitive:
        for index in probe["groups"][name]:
            labels[int(index)].append(name)
    for field in ("source", "receiver"):
        edges[field + "_token"] = [tokenizer.decode([int(ids[index])]) for index in edges[field]]
    edges["source_roles"] = ["|".join(labels[index]) for index in edges.source]
    edges["edge_is_self"] = edges.source == edges.receiver
    edges["writes_at_target"] = edges.receiver == len(ids) - 1
    return edges


def save_table(rows, directory, name, identity):
    frame = pd.DataFrame(rows)
    for key, value in identity.items():
        frame[key] = value
    frame.to_csv(directory / name, index=False)


def run_panel(model, tokenizer, original, identity, args):
    stem = "_".join(identity[key] for key in ("case_id", "side", "panel"))
    directory = Path(args.output) / "panels" / stem
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "worlds").mkdir(exist_ok=True)
    probe = dict(original, record_writes=False)
    freeze_json(directory / "context.json", dict(
        identity=identity, prefix_ids=list(map(int, probe["prefix_ids"])),
        candidates=[list(map(int, ids)) for ids in probe["candidates"]]))
    edges, metadata = discover(model, probe, directory, args.edge_top_k, args.objective)
    units = select_units(edges, args.max_units, args.control_units, args.seed)
    pairs = select_message_pairs(units, args.max_pairs)
    freeze_json(directory / "plan.json", dict(units=units, pairs=pairs, doses=args.doses))
    described = describe_edges(edges, tokenizer, probe)
    save_table(described, directory, "edges.csv", identity)
    probe["capture_units"] = units
    baseline, writes, reference = baseline_world(model, probe, directory)
    metadata["prefix_replay_margin_error"] = metadata["next_margin"] - baseline["next_margin"]
    metadata["objective_replay_error"] = metadata["objective_value"] - baseline[args.objective]
    write_json(directory / "gradient_check.json", metadata)
    singles, rows = single_worlds(model, probe, units, args.doses, baseline, reference, directory, args.seed)
    save_table(rows, directory, "interventions.csv", identity)
    rows = joint_worlds(model, probe, units, pairs, args.doses, baseline, singles, reference, directory)
    save_table(rows, directory, "interactions.csv", identity)
    rows = mediation_worlds(model, probe, units, pairs, baseline, singles, writes, reference, directory)
    save_table(rows, directory, "mediation.csv", identity)


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "flow_config.json").exists():
        raise ValueError("This directory contains v1/v2 results; use a new v3 output directory")
    cases = json.loads(Path(args.cases).read_text())
    directory = output / "inventory"
    directory.mkdir(exist_ok=True)
    settings, samples, prompts = inventory(args.samples, directory, cases)
    config = dict(protocol=PROTOCOL, model=args.model or settings["model"], dtype=args.dtype,
                  samples=args.samples, cases=cases, sampling_settings=settings,
                  max_units=args.max_units, control_units=args.control_units,
                  max_pairs=args.max_pairs, edge_top_k=args.edge_top_k, doses=args.doses,
                  recent_window=args.recent_window, seed=args.seed,
                  selection="final_gate_derivative_before_role_annotation",
                  objective=args.objective, task="reviewed_candidate_contrast_not_detector_score")
    freeze_json(output / "transport_config.json", config)
    return config, cases, samples, prompts


def run_transport(args):
    from .transport_report import report_transport

    if args.stage == "report":
        report_transport(Path(args.output))
        return
    config, cases, samples, prompts = prepare(args)
    if args.stage == "inventory":
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .main import verify_replay

    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True, use_fast=True)
    compiled = compile_cases(args.samples, tokenizer, samples, prompts, cases, Path(args.output))
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager").to(args.device).eval().requires_grad_(False)
    if model.config.model_type != "llama":
        raise ValueError("Transport v3 is validated for native Llama eager attention only")
    for entry in tqdm(compiled, desc="reviewed cases"):
        case = entry["case"]
        for side, native in entry["sides"].items():
            verify_replay(model, native, args.replay_atol, args.message_rtol)
            for probe in candidate_panels(tokenizer, native, case, args.recent_window):
                identity = dict(case_id=case["case_id"], source_id=case["source_id"],
                                side=side, panel=probe["panel"])
                run_panel(model, tokenizer, probe, identity, args)
    report_transport(Path(args.output))


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["inventory", "run", "report"], default="run")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    parser.add_argument("--output", default="outputs/target_transport_v3")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--control-units", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--edge-top-k", type=int, default=2)
    parser.add_argument("--objective", choices=["sequence_margin", "next_margin"], default="sequence_margin")
    parser.add_argument("--doses", type=float, nargs="+", default=[.25, 1.])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recent-window", type=int, default=10)
    parser.add_argument("--replay-atol", type=float, default=.25)
    parser.add_argument("--message-rtol", type=float, default=.02)
    args = parser.parse_args(argv)
    if args.max_units < 1 or args.edge_top_k < 1 or min(args.max_pairs, args.control_units) < 0:
        parser.error("Require max-units and edge-top-k >= 1, controls and pairs >= 0")
    if 1. not in args.doses or any(not 0 < dose <= 1 for dose in args.doses):
        parser.error("Doses must be in (0,1] and include 1 for restoration comparisons")
    run_transport(args)


if __name__ == "__main__":
    cli()
