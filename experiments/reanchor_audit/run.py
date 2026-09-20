"""Foreground, resumable audit of routes preceding reviewed factual claims."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .graph import analyze_graph, flow_checks
from .inputs import compile_probe, freeze, inventory, model_path
from .selection import describe_edges, make_plan, ranked_edges, select_events


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-input", type=Path, default=Path("outputs/paired_head_transport_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reanchor_audit_v1"))
    parser.add_argument("--stage", choices=["inventory", "discover", "run", "report"], default="run")
    parser.add_argument("--model", help="Original generator weights, if the saved local path moved")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--control-radius", type=int, default=10)
    parser.add_argument("--doses", type=float, nargs="+", default=[.25, 1.])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sham-atol", type=float, default=1e-5)
    parser.add_argument("--message-rtol", type=float, default=.02)
    return parser.parse_args()


def save_npz(path, arrays):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def route_checks(graph, flow, tolerance):
    checks = flow_checks(flow)
    reconstruction = float(graph["reconstruction_error"].max())
    receiver = np.arange(graph["source"].shape[2])[None, None, :, None]
    future_mass = float(graph["attention"][graph["source"] > receiver].sum())
    valid = reconstruction <= tolerance and future_mass == 0.
    valid &= all(row["reachable"] and row["max_layer_mass_error"] < 1e-8
                 and abs(row["sink_flow"] - 1.) < 1e-8 for row in checks)
    return dict(flow=checks, reconstruction_max_relative_error=reconstruction,
        future_attention_mass=future_mass, route_checks_ok=bool(valid),
        retained_mass_quantiles=np.quantile(graph["retained_mass"], [0, .1, .5, .9, 1]).tolist())


def discover(model, tokenizer, probe, roles, token_text, directory, args):
    from .capture import capture_graph

    path = directory / "routes.npz"
    if not path.exists():
        special = np.isin(probe["prefix_ids"], tokenizer.all_special_ids)
        graph = capture_graph(model, probe, args.top_k, args.window, special)
        save_npz(path, graph)
    graph = read_npz(path)
    flow = analyze_graph(graph)
    save_npz(directory / "flow.npz", flow)
    checks = route_checks(graph, flow, args.message_rtol)
    freeze(directory / "route_checks.json", checks)
    if not checks["route_checks_ok"]:
        raise ValueError(f"Route reconstruction/conservation failed: {directory}")
    edges = ranked_edges(graph, flow)
    selected = select_events(edges, args.events, len(probe["prefix_ids"]) - 1, args.control_radius)
    describe_edges(edges, token_text, roles, probe["prompt_length"]).to_csv(
        directory / "edges.csv.gz", index=False, compression="gzip")
    describe_edges(selected, token_text, roles, probe["prompt_length"]).to_csv(directory / "events.csv", index=False)
    plan = make_plan(selected, edges, roles)
    freeze(directory / "plan.json", plan)
    return graph, plan


def run_case(model, tokenizer, item, args):
    case = item["case"]
    directory = args.output / "cases" / case["case_id"] / item["side"]
    (directory / "worlds").mkdir(parents=True, exist_ok=True)
    probe, roles, token_text = compile_probe(item, tokenizer)
    context = dict(case_id=case["case_id"], source_id=case["source_id"], side=item["side"],
        prefix_ids=probe["prefix_ids"], candidates=probe["candidates"], token_text=token_text,
        prompt_length=probe["prompt_length"], roles={key: value.tolist() for key, value in roles.items()},
        history_status=case["history_status"][item["side"]], reviewed_case=case)
    freeze(directory / "context.json", context)
    graph, plan = discover(model, tokenizer, probe, roles, token_text, directory, args)
    if args.stage == "discover" or not plan:
        return
    from .trials import audit_trials

    record = audit_trials(model, probe, plan, directory, args.doses, args.random_seeds, args.sham_atol)
    first_logp = [record["correct_first_logp"], record["wrong_first_logp"]]
    error = float(np.max(np.abs(graph["candidate_first_logp"] - first_logp)))
    freeze(directory / "baseline_checks.json", dict(capture_first_logp_error=error,
        baseline_ok=error <= args.sham_atol, full_sequence_margin=record["measure"]))
    if error > args.sham_atol:
        raise ValueError(f"Capture changed candidate readout: {directory}")


def load_model(path, args):
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(path, dtype=getattr(torch, args.dtype),
        attn_implementation="eager", local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    from .report import report

    if args.stage == "report":
        print(json.dumps(report(args.output), ensure_ascii=False))
        return
    cases = inventory(args.paired_input, args.output)
    if not cases:
        raise ValueError("No reviewed paired onset contexts found")
    if args.stage == "inventory":
        print(json.dumps(report(args.output), ensure_ascii=False))
        return
    settings = {key: value for key, value in vars(args).items() if key not in ("stage", "output", "paired_input")}
    settings["model"] = model_path(args)
    freeze(args.output / "audit_config.json", settings)
    model, tokenizer = load_model(settings["model"], args)
    for item in tqdm(cases, desc="paired reanchor audit", unit="side"):
        run_case(model, tokenizer, item, args)
        report(args.output, package=False)
    print(json.dumps(report(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
