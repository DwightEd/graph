"""Command-line entry points for the mechanism audit."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common.ragtruth_alignment import TASK_TYPES

from .audit import audit_target, save_audit
from .dataset import (
    ManifestAuditCorpus,
    ManifestLabelSource,
    RagTruthAuditCorpus,
)
from .flow import FlowSignal
from .mechanism_plot import save_mechanism_figure
from .route_plan import RouteBudget
from .sample_scan import render_sample_scan
from .subset import (
    CohortPlan,
    MechanismPlan,
    SubsetAuditRunner,
    SubsetRunConfig,
    TargetPlan,
)
from .subset_report import evaluate_subset_split
from .target_selection import REANCHOR_POLICIES
from .worlds import load_world

MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def selected_splits(args: argparse.Namespace) -> tuple[str, ...]:
    return ("train", "test") if args.split == "all" else (args.split,)


def subset_output_root(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return (
        Path(__file__).resolve().parent
        / "outputs"
        / args.model.name
        / (
            "native_mechanism_all"
            if args.command == "audit-all"
            else "native_mechanism_v3"
        )
    )


def subset_split_output(args: argparse.Namespace, split: str) -> Path:
    return subset_output_root(args) / split


def split_manifest_path(path: Path, split: str) -> Path:
    """Resolve one split from either a JSON file or a train/test directory."""

    return path / f"{split}.json" if path.is_dir() else path


def audit_corpus_from_args(args: argparse.Namespace, tokenizer, split: str):
    """Construct the only external data dependency used by the audit runner."""

    if args.dataset_manifest is not None:
        corpus = ManifestAuditCorpus.open(
            split_manifest_path(args.dataset_manifest, split)
        )
        if corpus.identity.split.casefold() != split.casefold():
            raise ValueError(
                f"dataset split {corpus.identity.split!r} differs from requested "
                f"split {split!r}"
            )
        return corpus
    return RagTruthAuditCorpus.open(
        args.cache / split,
        args.source_info,
        split=split,
        model_id=str(args.model.resolve()),
        tokenizer=tokenizer,
        sample_ids=tuple(args.sample_id or ()),
    )


def load_model(path: Path, device: str, dtype: str):
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            torch_dtype=DTYPE[dtype],
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def number(value: object) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def confirmation_rate(value: object) -> str:
    return "not-run" if value is None else number(value)


def corridor(args: argparse.Namespace) -> dict:
    """Audit every fixed target in one aligned clean/counterfactual pair."""

    world = load_world(args.pair)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    if Path(world.tokenizer_id).name != Path(tokenizer.name_or_path).name:
        raise ValueError("paired-world tokenizer differs from the loaded tokenizer")

    output = args.output or args.pair.parent / "etcc_outputs"
    output.mkdir(parents=True, exist_ok=True)
    signal = FlowSignal(args.flow_signal)
    reports = {}
    for target in world.targets:
        result = audit_target(
            model,
            world,
            target,
            signal,
            carrier_scope=args.carrier_scope,
            coverage=args.edge_coverage,
            gradient_steps=args.gradient_steps,
            query_chunk=args.query_chunk,
            root_screen_limit=args.root_screen_limit,
            carrier_limit=args.carrier_limit,
            materialize_messages=args.materialize_messages,
        )
        destination = output / (
            f"{world.sample_id}_q{target.query_position}"
            f"_a{target.positive_token_id}_b{target.negative_token_id}"
            f"_{signal.value}.npz"
        )
        save_audit(
            destination,
            world,
            result,
            model_id=str(args.model),
            model_dtype=args.dtype,
            coverage=args.edge_coverage,
            gradient_steps=args.gradient_steps,
            carrier_scope=args.carrier_scope,
            query_chunk=args.query_chunk,
            root_screen_limit=args.root_screen_limit,
            carrier_limit=args.carrier_limit,
            materialize_messages=args.materialize_messages,
        )
        effect = result.effect
        key = (
            f"q{target.query_position}"
            f"_a{target.positive_token_id}_b{target.negative_token_id}"
        )
        reports[key] = {
            "output": destination,
            "pair_effect": effect.pair_effect,
            "selected_root_unit_id": result.selected_root_unit_id,
            "selected_root_confirmed": result.selected_root_confirmed,
            "corridor_confirmed": result.corridor_confirmed,
            "necessity": effect.necessity,
            "sufficiency": effect.sufficiency,
            "mediated_sufficiency": effect.mediated_sufficiency,
        }
        print(
            f"q={target.query_position} signal={signal.value} "
            f"pair={number(effect.pair_effect)} "
            f"root={result.selected_root_unit_id} "
            f"root_ok={result.selected_root_confirmed} "
            f"corridor_ok={result.corridor_confirmed} "
            f"necessity={number(effect.necessity)} "
            f"rescue={number(effect.sufficiency)} "
            f"mediated={number(effect.mediated_sufficiency)} "
            f"output={destination}"
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model, tokenizer
    clear_memory()
    return reports


def _load_token_labels(path: Path | None) -> Sequence[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("--tokens-json must contain a JSON array of token strings")
    return value


def render_artifact(
    artifact_path: Path,
    output_path: Path | None = None,
    token_labels: Sequence[str] | None = None,
) -> Path:
    """Render one label-free mechanism figure from a native audit NPZ."""

    destination = output_path or artifact_path.with_suffix(".mechanism.png")
    with np.load(artifact_path, allow_pickle=False) as stored:
        artifact = {name: stored[name] for name in stored.files}
    return save_mechanism_figure(destination, artifact, token_labels)


def mechanism_plot(args: argparse.Namespace) -> Path:
    destination = render_artifact(
        args.artifact,
        args.output,
        _load_token_labels(args.tokens_json),
    )
    print(f"mechanism figure: {destination}")
    return destination


def _artifact_token_labels(artifact: Path, tokenizer) -> list[str]:
    with np.load(artifact, allow_pickle=False) as stored:
        token_ids = stored["token_ids"].astype(np.int64).tolist()
    return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]


def _render_subset(output: Path, tokenizer) -> int:
    artifacts = sorted((output / "audits").glob("**/*.npz"))
    for artifact in tqdm(
        artifacts,
        desc=f"{output.name} plots",
        unit="figure",
        dynamic_ncols=True,
    ):
        render_artifact(
            artifact,
            token_labels=_artifact_token_labels(artifact, tokenizer),
        )
    return len(artifacts)


def _render_scans(output: Path, tokenizer) -> int:
    scans = sorted((output / "scans").glob("**/*.npz"))
    for scan in tqdm(scans, desc=f"{output.name} timelines", unit="figure"):
        destination = scan.with_suffix(".timeline.png")
        if (
            not destination.is_file()
            or destination.stat().st_mtime < scan.stat().st_mtime
        ):
            render_sample_scan(scan, destination, tokenizer)
    return len(scans)


def subset_config_from_args(
    args: argparse.Namespace,
    tokenizer,
    split: str,
    available_tasks: tuple[str, ...] = TASK_TYPES,
) -> SubsetRunConfig:
    """Translate one validated CLI request into the subset service contract."""

    return SubsetRunConfig(
        model_id=str(args.model.resolve()),
        model_dtype=args.dtype,
        tokenizer_id=Path(tokenizer.name_or_path).name,
        cohort=CohortPlan(
            tasks=available_tasks if args.task == "all" else (args.task,),
            samples_per_task=args.samples_per_task,
            sample_ids=tuple(args.sample_id or ()),
            seed=args.selection_seed,
        ),
        targets=TargetPlan(
            count=args.targets_per_sample,
            policy=args.target_policy,
            max_response_tokens=(
                None if args.max_response_tokens == 0 else args.max_response_tokens
            ),
        ),
        mechanism=MechanismPlan(
            signal=FlowSignal(args.flow_signal),
            carrier_scope=args.carrier_scope,
            coverage=args.edge_coverage,
            query_chunk=args.query_chunk,
            route_budget=RouteBudget(
                edges_per_head=args.edges_per_head,
                max_rows=args.max_route_rows,
                root_candidates=args.root_candidates,
                hub_candidates=args.hub_candidates,
                corridor_edges=args.corridor_edges,
                confirm=args.confirm,
            ),
            local_window=args.local_window,
        ),
        scan_only=args.scan_only,
    )


def route_plan_summary(args: argparse.Namespace) -> str:
    """Return the visible execution plan for one subset invocation."""

    mode = (
        "structure-only"
        if args.scan_only
        else "candidate-discovery+confirmation"
        if args.confirm
        else "candidate-discovery"
    )
    return (
        f"route plan: mode={mode} target_policy={args.target_policy} "
        f"samples_per_task={args.samples_per_task or 'all'} "
        f"response_tokens={args.max_response_tokens or 'full'} "
        f"target_rows={args.targets_per_sample} carrier_scope={args.carrier_scope} "
        f"edges_per_head={args.edges_per_head} "
        f"max_route_rows={args.max_route_rows} "
        f"root_candidates={args.root_candidates} "
        f"hub_candidates={args.hub_candidates} "
        f"corridor_edges={args.corridor_edges}"
    )


def audit_subset(args: argparse.Namespace) -> dict:
    """Run the label-free native audit, optionally rendering every target."""

    print(route_plan_summary(args), flush=True)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    reports = {}
    for split in selected_splits(args):
        output = subset_split_output(args, split)
        corpus = audit_corpus_from_args(args, tokenizer, split)
        available_tasks = tuple(dict.fromkeys(row.task_type for row in corpus.records))
        config = subset_config_from_args(
            args,
            tokenizer,
            split,
            available_tasks,
        )
        counts = SubsetAuditRunner(
            model,
            tokenizer,
            corpus,
            output,
            config,
        ).run()
        reports[split] = counts
        confirmation = (
            f"corridors_confirmed={counts['confirmed']}"
            if config.mechanism.route_budget.confirm
            else "exact_confirmation=not-requested"
        )
        print(
            f"subset {split}: samples={counts['samples']} "
            f"targets={counts['targets']} resumed={counts['resumed']} "
            f"{confirmation}"
        )
        if args.plot:
            print(f"mechanism figures: {_render_subset(output, tokenizer)}")
        if args.plot or args.command == "audit-all":
            print(f"sample timeline figures: {_render_scans(output, tokenizer)}")
        if args.evaluate:
            dataset_path = (
                split_manifest_path(args.dataset_manifest, split)
                if args.dataset_manifest is not None
                else args.cache / split
            )
            label_source = (
                ManifestLabelSource.open(
                    split_manifest_path(args.label_manifest, split)
                )
                if args.label_manifest is not None
                else None
            )
            evaluation = evaluate_subset_split(
                dataset_path,
                output,
                label_source=label_source,
                plot=args.plot or args.command == "audit-all",
            )
            _print_evaluation(split, evaluation)
    del model, tokenizer
    clear_memory()
    return reports


def evaluate_subset(args: argparse.Namespace) -> dict:
    """Join hallucination labels only after mechanism capture is frozen."""

    reports = {}
    for split in selected_splits(args):
        dataset_path = (
            split_manifest_path(args.dataset_manifest, split)
            if args.dataset_manifest is not None
            else args.cache / split
        )
        label_source = (
            ManifestLabelSource.open(split_manifest_path(args.label_manifest, split))
            if args.label_manifest is not None
            else None
        )
        report = evaluate_subset_split(
            dataset_path,
            subset_split_output(args, split),
            label_source=label_source,
            plot=args.plot,
        )
        reports[split] = report
        _print_evaluation(split, report)
    return reports


def _print_evaluation(split: str, report: dict) -> None:
    print(f"\n=== NATIVE MECHANISM {split.upper()} ===")
    for task, groups in report["groups"].items():
        coverage = report["full_scan_coverage"].get(task)
        if coverage is not None:
            print(
                f"{task:9s} scanned_samples={coverage['scanned_samples']} "
                f"tokens={coverage['scanned_response_tokens']}/"
                f"{coverage['full_response_tokens']} "
                f"nonhallucinated={coverage['scanned_nonhallucinated_tokens']} "
                f"hallucinated={coverage['scanned_hallucinated_tokens']}"
            )
        if not groups["all"]["targets"]:
            print(f"{task:9s} functional_targets=0; functional AUROC/AUPRC not run")
            continue
        confirmation = groups["all"]["confirmation_rate"]
        metrics = groups["raw_axis_evaluation"]
        route = metrics["route_origin_competition"]
        switch = metrics["temporal_switch_score"]
        adoption = metrics["evidence_adoption"]
        print(
            f"{task:9s} clean={groups['clean']['targets']} "
            f"hallucinated={groups['hallucinated']['targets']} "
            f"route_auc={number(route['auroc'])} "
            f"switch_auc(raw/neg)={number(switch['auroc'])}/"
            f"{number(switch['negated_auroc'])} "
            f"adoption_auc={number(adoption['auroc'])} "
            f"n(route/switch/adopt)={route['evaluated_targets']}/"
            f"{switch['evaluated_targets']}/{adoption['evaluated_targets']} "
            "root_ok="
            f"{confirmation_rate(confirmation['selected_root_confirmed'])} "
            "corridor_ok="
            f"{confirmation_rate(confirmation['corridor_confirmed'])} "
            "chain_ok="
            f"{confirmation_rate(confirmation['full_chain_confirmed'])}"
        )


def add_model(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")


def add_corridor(command: argparse.ArgumentParser) -> None:
    command.add_argument("--pair", type=Path, required=True)
    add_model(command)
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--flow-signal",
        choices=tuple(signal.value for signal in FlowSignal),
        default=FlowSignal.MESSAGE.value,
        help="candidate routing: target-specific true message or raw attention",
    )
    command.add_argument("--carrier-scope", choices=("response", "all"), default="all")
    command.add_argument("--edge-coverage", type=float, default=0.95)
    command.add_argument("--gradient-steps", type=int, default=1)
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument("--root-screen-limit", type=int, default=8)
    command.add_argument("--carrier-limit", type=int, default=3)
    command.add_argument("--materialize-messages", action="store_true")


def add_subset(command: argparse.ArgumentParser, *, evaluation: bool = False) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "dataset-neutral aligned input JSON; for --split all pass a "
            "directory containing train.json and test.json"
        ),
    )
    command.add_argument(
        "--label-manifest",
        type=Path,
        help=(
            "separate post-capture label JSON; for --split all pass a directory"
        ),
    )
    command.add_argument("--output", type=Path)
    command.add_argument("--split", choices=("train", "test", "all"), default="test")
    if evaluation:
        command.add_argument(
            "--plot", action="store_true", help="render cohort comparisons"
        )
        return

    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument(
        "--task",
        default="all",
        help="task name from the selected corpus, or all",
    )
    command.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="samples per task; 0 includes every sample without label filtering",
    )
    command.add_argument("--sample-id", action="append")
    command.add_argument("--selection-seed", type=int, default=2026)
    command.add_argument("--targets-per-sample", type=int, default=1)
    command.add_argument(
        "--target-policy",
        choices=(
            "uncertain",
            "low-margin",
            "evenly-spaced",
            "all",
            *REANCHOR_POLICIES,
        ),
        default="reanchor",
        help=(
            "reanchor freezes strongest per-head exact W_O(A V) local-to-prompt/"
            "remote-response flips; reanchor-window spends the target-row budget "
            "on event centers and +/-1 context; no-event runs fall back to "
            "evenly-spaced rows"
        ),
    )
    command.add_argument(
        "--max-response-tokens",
        type=int,
        default=128,
        help="causal pilot horizon; 0 keeps the full response",
    )
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument(
        "--flow-signal",
        choices=tuple(signal.value for signal in FlowSignal),
        default=FlowSignal.MESSAGE.value,
        help="transport screen: exact message norm or raw attention",
    )
    command.add_argument(
        "--carrier-scope", choices=("response", "all"), default="response"
    )
    command.add_argument("--edge-coverage", type=float, default=0.9)
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument(
        "--edges-per-head",
        type=int,
        default=2,
        help=(
            "per head-row top-k for each capture branch and cap in the frozen "
            "corridor; capture union is at most 2k"
        ),
    )
    command.add_argument(
        "--max-route-rows",
        type=int,
        default=256,
        help="hard limit on represented destination token rows",
    )
    command.add_argument(
        "--root-candidates",
        dest="root_candidates",
        type=int,
        default=4,
    )
    command.add_argument(
        "--hub-candidates",
        dest="hub_candidates",
        type=int,
        default=8,
    )
    command.add_argument("--corridor-edges", type=int, default=64)
    command.add_argument(
        "--confirm",
        action="store_true",
        help="run exact interventions only for the frozen candidate budget",
    )
    command.add_argument("--local-window", type=int, default=10)
    command.add_argument(
        "--plot",
        action="store_true",
        help="render a label-free mechanism figure for every completed target",
    )
    command.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="join labels and compute cohort comparisons after each split finishes",
    )
    command.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "save full per-head timelines without target gradients/interventions; "
            "rerun the same output without this flag to add selected-target function"
        ),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Head-resolved evidence routing and causal mechanism audit"
    )
    commands = root.add_subparsers(dest="command", required=True)

    corridor_command = commands.add_parser(
        "corridor", help="audit an aligned clean/counterfactual pair"
    )
    add_corridor(corridor_command)
    corridor_command.set_defaults(handler=corridor)

    subset_command = commands.add_parser(
        "subset", help="capture a label-free RAGTruth mechanism subset"
    )
    add_subset(subset_command)
    subset_command.set_defaults(handler=audit_subset)

    all_command = commands.add_parser(
        "audit-all",
        help="audit every sample and full response; save timelines and compare label cohorts",
    )
    add_subset(all_command)
    all_command.set_defaults(
        handler=audit_subset,
        split="all",
        samples_per_task=0,
        max_response_tokens=0,
        targets_per_sample=3,
        target_policy="reanchor-window",
        evaluate=True,
    )

    subset_evaluate_command = commands.add_parser(
        "subset-evaluate", help="join labels after subset capture is complete"
    )
    add_subset(subset_evaluate_command, evaluation=True)
    subset_evaluate_command.set_defaults(handler=evaluate_subset)

    plot_command = commands.add_parser(
        "mechanism-plot",
        help="render one native schema-3 audit without reading labels",
    )
    plot_command.add_argument("--artifact", type=Path, required=True)
    plot_command.add_argument("--output", type=Path)
    plot_command.add_argument(
        "--tokens-json", type=Path, help="optional JSON array of display token strings"
    )
    plot_command.set_defaults(handler=mechanism_plot)
    return root


def validate_args(args: argparse.Namespace) -> None:
    for name in ("query_chunk", "gradient_steps", "local_window"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "edges_per_head",
        "max_route_rows",
        "root_candidates",
        "corridor_edges",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "hub_candidates") and args.hub_candidates < 0:
        raise ValueError("--hub-candidates cannot be negative")
    for name in ("root_screen_limit", "carrier_limit"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if hasattr(args, "samples_per_task") and args.samples_per_task < 0:
        raise ValueError("--samples-per-task cannot be negative (0 means all)")
    if hasattr(args, "targets_per_sample") and args.targets_per_sample < 1:
        raise ValueError("--targets-per-sample must be positive")
    if hasattr(args, "max_response_tokens") and args.max_response_tokens < 0:
        raise ValueError("--max-response-tokens cannot be negative")
    if hasattr(args, "edge_coverage") and not 0 < args.edge_coverage <= 1:
        raise ValueError("--edge-coverage must lie in (0,1]")
    if (
        args.command in {"subset", "audit-all"}
        and args.sample_id
        and args.split == "all"
    ):
        raise ValueError("--sample-id requires one concrete --split")
    if getattr(args, "dataset_manifest", None) is not None:
        if args.split == "all" and args.dataset_manifest.suffix:
            raise ValueError(
                "--dataset-manifest must be a directory when --split is all"
            )
        if getattr(args, "evaluate", True) and args.label_manifest is None:
            raise ValueError(
                "manifest-dataset evaluation requires a separate --label-manifest"
            )
    if getattr(args, "label_manifest", None) is not None:
        if args.dataset_manifest is None:
            raise ValueError("--label-manifest requires --dataset-manifest")
        if args.split == "all" and args.label_manifest.suffix:
            raise ValueError(
                "--label-manifest must be a directory when --split is all"
            )


def main() -> None:
    command_parser = parser()
    args = command_parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        command_parser.error(str(error))
    args.handler(args)


if __name__ == "__main__":
    main()
