"""Run a foreground, resumable message audit using the standalone state_audit package."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from state_audit.storage import read_json, start_stage, write_json
from tqdm import tqdm

from .inputs import compile_groups, describe_case, read_cases, read_paired_cases
from .protocol import random_groups, run_protocol
from .trials import Trials

DEFAULT_PLAN = Path(__file__).with_name("heads_llama32.json")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--cases", type=Path, help="Portable cases or teaching answer references")
    inputs.add_argument("--paired-input", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--model", help="Checkpoint override, e.g. after relocating local weights")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("outputs/head_interactions_v2"))
    parser.add_argument("--stage", choices=("run", "prepare", "report"), default="run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--query-offsets", nargs="+", type=int, default=[0])
    parser.add_argument("--random-seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--gain", type=float, default=1.5)
    parser.add_argument("--atol", type=float, default=0.01, help="Per-token sham tolerance, nats")
    parser.add_argument(
        "--minimum-effect", type=float, default=0.05, help="Descriptive effect floor, nats"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def input_model(args):
    if args.cases is None and args.paired_input is None:
        args.paired_input = Path("outputs/paired_head_transport_v1")
    if args.model:
        return args.model
    if args.cases is not None:
        return read_json(args.cases)["model"]
    return read_json(args.paired_input / "paired_config.json")["sampling"]["model"]


def load_inputs(args, tokenizer):
    if args.cases is not None:
        return read_cases(args.cases, tokenizer)
    return read_paired_cases(args.paired_input, tokenizer)


def panel_groups(groups, model, seeds):
    yield "selected", groups
    for seed in seeds:
        yield f"random_{seed}", random_groups(groups, model, seed)


def donor_messages(model, case, groups, directory):
    trials = Trials(model, case, groups, directory)
    return trials.evaluate("baseline")[1]


def run_panel(model, tokenizer, case, cases, groups, plan, shift, panel, directory, args):
    path = directory / "complete.json"
    if args.resume and path.exists():
        return
    context = describe_case(case, groups, tokenizer)
    context.update(
        panel=panel,
        query_offset=shift,
        groups={name: [asdict(site) for site in sites] for name, sites in groups.items()},
    )
    write_json(directory / "context.json", context)
    donor = None
    if case.donor is not None:
        donor_case = cases[case.donor]
        donor_groups = compile_groups(plan, donor_case, model, shift)
        if panel != "selected":
            donor_groups = random_groups(donor_groups, model, int(panel.removeprefix("random_")))
        donor = donor_messages(model, donor_case, donor_groups, directory / "donor")
        write_json(
            directory / "donor/context.json", describe_case(donor_case, donor_groups, tokenizer)
        )
    with tqdm(desc="conditions", unit="trial", leave=False) as progress:
        trials = Trials(model, case, groups, directory, progress)
        result = run_protocol(trials, groups, args.atol, args.gain, donor, not args.no_compensation)
    write_json(directory / "results.json", result)
    write_json(path, dict(valid=result["valid"], trials=len(trials.records)))


def execute(model, tokenizer, cases, plan, args):
    lookup = {case.id: case for case in cases}
    total = len(cases) * len(args.query_offsets) * (1 + len(args.random_seeds))
    with tqdm(total=total, desc="head interactions", unit="panel") as progress:
        for index, case in enumerate(cases):
            for shift in args.query_offsets:
                groups = compile_groups(plan, case, model, shift)
                for panel, selected in panel_groups(groups, model, args.random_seeds):
                    directory = args.output / "cases" / f"{index:06d}" / f"query_{shift}" / panel
                    progress.set_postfix_str(f"{case.id} q{shift:+d} {panel}")
                    run_panel(
                        model,
                        tokenizer,
                        case,
                        lookup,
                        selected,
                        plan,
                        shift,
                        panel,
                        directory,
                        args,
                    )
                    progress.update()


def freeze_settings(model, cases, plan, args, name):
    settings = dict(
        protocol_version=2,
        source_restore="native_contraction_source_replacement",
        candidate_shape="equal_length_causal_tail_padding",
        purpose="candidate_conditioned_mechanism_audit_not_detector_evaluation",
        model=name,
        revision=args.revision,
        family=model.native.config.model_type,
        model_config=model.native.config.to_dict(),
        dtype=args.dtype,
        plan=plan,
        cases=[asdict(case) for case in cases],
        query_offsets=args.query_offsets,
        random_seeds=args.random_seeds,
        gain=args.gain,
        atol=args.atol,
        minimum_effect=args.minimum_effect,
        compensation=not args.no_compensation,
        readout="sum_logp(supported)-sum_logp(rival); mean margin is a separate sensitivity view",
    )
    start_stage(args.output / "settings.json", settings, args.resume)


def main():
    args = arguments()
    from .report import report

    if args.stage == "report":
        print(json.dumps(report(args.output, args.bootstrap), ensure_ascii=False))
        return
    name = input_model(args)
    if args.stage == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, revision=args.revision, use_fast=True)
        cases = load_inputs(args, tokenizer)
        write_json(
            args.output / "prepared_cases.json", dict(model=name, cases=[asdict(c) for c in cases])
        )
        print(json.dumps(dict(cases=len(cases), model=name, stage="prepared_no_model_forward")))
        return
    from state_audit.model import load_model

    model, tokenizer = load_model(name, args.revision, args.device, args.dtype)
    cases, plan = load_inputs(args, tokenizer), read_json(args.plan)
    freeze_settings(model, cases, plan, args, name)
    execute(model, tokenizer, cases, plan, args)
    print(json.dumps(report(args.output, args.bootstrap), ensure_ascii=False))


if __name__ == "__main__":
    main()
