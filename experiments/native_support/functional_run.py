"""Capture original answers directly, report complete coverage, and auto-pack."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from state_audit.storage import read_json, start_stage, write_arrays, write_json
from tqdm import tqdm

from .choice_cache import CaptureReader
from .functional_report import report
from .functional_units import capture_units
from .inputs import validate_tokenizer


def protocol(args, settings):
    return dict(version=3, model=settings["model"], responses=settings["responses"],
        response_ids=args.response_ids, start_target=args.start_target, stop_target=args.stop_target,
        max_units=args.max_units, max_unit_tokens=args.max_unit_tokens, dtype=args.dtype,
        purpose="native_observed_answer_function_audit",
        selection="explicit_range_and_unit_cap_only; all_text_types_included",
        unit_boundary="punctuation_then_budget_chunks; not_semantic_or_reanchor_boundaries",
        objective="sum_observed_log_probability_over_saved_unit",
        gradients="native_QK_RMS_SwiGLU; unit_objective_not_independent_token_gradients",
        root_readout="gradient_dot_embedding; local_sensitivity_not_additive_ledger",
        head_readout="receiver_OV_message_sensitivity; residual_plus_current_FFN_mediated",
        head_query_scope="every_observed_query; all_physical_layers_and_heads",
        normalization="terminal_FFN_float32_RMS_margin_identity; native_roundoff_separate",
        pre_ffn_readout="final_RMS_and_unembedding_on_native_pre_FFN_states",
        prediction_alignment="query=P+t-1; native_causal_mask",
        history="exact_saved_prefix; roots_do_not_differentiate_through_discrete_sampling",
        key_groups="saved_prefix_groups_plus_within_unit_history",
        future_tokens_used=True, labels_used_for_scoring=False, labels_used_for_training=False,
        parameter_fitting=False, interventions=False, automatic_method_selection=False,
        candidate_generation=False, semantic_filter=False,
        source_types="existing_automatic_blocks; semantic_roles_unassigned",
        observer_scope="teacher_forced_observer; may_differ_from_original_generator",
        detector_evaluation=False, existing_scores_modified=False)


def plan_units(args, settings, reader):
    jobs, selected_responses = [], []
    for index, response in enumerate(settings["responses"]):
        if args.response_ids and response["id"] not in args.response_ids:
            continue
        selected_responses.append(index)
        directory = args.output / "responses" / f"{index:04d}"
        sources = reader.json(f"value_transport/capture/{index:04d}/sources.json")
        groups = sources["group_ids"]
        if (len(groups) != len(response["token_ids"]) or min(groups) < 0
                or max(groups) >= len(sources["blocks"]) + 3):
            raise ValueError(f"{response['id']}: source groups do not match saved inputs")
        if len(response["token_text"]) != len(response["token_ids"]):
            raise ValueError(f"{response['id']}: token ID/text lengths differ")
        start_stage(directory / "sources.json", sources, args.resume)
        intervals = capture_units(response, args.start_target, args.stop_target, args.max_unit_tokens)
        for selected, interval in enumerate(intervals):
            if args.max_units and selected >= args.max_units:
                break
            identity = dict(response_index=index, response_id=response["id"],
                            source_id=response["source_id"], **interval)
            unit = directory / f"unit_{identity['start']:06d}"
            jobs.append((response, sources, unit, identity))
    found = {settings["responses"][index]["id"] for index in selected_responses}
    if args.response_ids and found != set(args.response_ids):
        raise ValueError(f"Unknown response IDs: {set(args.response_ids) - found}")
    start_stage(args.output / "plan.json", dict(response_indices=selected_responses,
                units=[identity for _, _, _, identity in jobs]), args.resume)
    return jobs


def capture_unit(model, response, sources, directory, identity):
    import torch
    from state_audit.functional_capture import capture_observed

    prompt = response["prompt_length"]
    prefix = response["token_ids"][:prompt + identity["start"]]
    targets = response["token_ids"][prompt + identity["start"]:prompt + identity["stop"]]
    if len(prefix) + len(targets) - 1 > model.native.config.max_position_embeddings:
        raise ValueError(f"{directory}: observed input exceeds model context; no truncation")
    started = perf_counter()
    device = model.native.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    arrays = capture_observed(model, prefix, targets, sources["group_ids"][:len(prefix)],
                              len(sources["blocks"]) + 3)
    write_arrays(directory / "observed.npz", **arrays)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    write_json(directory / "complete.json", {**identity,
        "capture_seconds_this_run": perf_counter() - started, "peak_cuda_bytes": peak})


def execute(args, settings, jobs):
    from state_audit.model import load_model

    model, tokenizer = None, None
    checked = set()
    for response, sources, directory, identity in tqdm(jobs, desc="original answer units"):
        if (directory / "complete.json").is_file():
            continue
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        if response["id"] not in checked:
            validate_tokenizer(response, tokenizer)
            checked.add(response["id"])
        capture_unit(model, response, sources, directory, identity)


def pack(destination, archive):
    if not (destination / "protocol.json").is_file():
        raise FileNotFoundError(f"{destination}: no functional audit protocol to package")
    archive = archive or destination.with_name(destination.name + "_review.zip")
    if archive.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Review archive must be outside the result directory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and ".partial." not in path.name:
                output.write(path, path.relative_to(destination))
    temporary.replace(archive)
    return str(archive)


def copy_review_context(source, destination):
    """Copy existing labels/baselines AFTER capture; never parse them for measurements."""
    with closing(CaptureReader(source)) as reader:
        if reader.exists("annotations.json"):
            (destination / "annotations.json").write_bytes(reader.bytes("annotations.json"))
        for index in range(len(reader.json("settings.json")["responses"])):
            name = f"value_transport/responses/{index:04d}/scores.npz"
            if reader.exists(name):
                target = destination / "baselines" / f"{index:04d}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(reader.bytes(name))


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing run directory or compact value-path ZIP")
    parser.add_argument("--output", type=Path, required=True, help="New v3 native audit directory")
    parser.add_argument("--stage", choices=("run", "capture", "report", "pack"), default="run")
    parser.add_argument("--response-ids", nargs="+")
    parser.add_argument("--start-target", type=int, default=0)
    parser.add_argument("--stop-target", type=int)
    parser.add_argument("--max-units", type=int, default=0, help="Explicit per-answer chunk cap; 0 = all")
    parser.add_argument("--max-unit-tokens", type=int, default=96, help="Split longer units; never skip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    if min(args.start_target, args.max_units) < 0 or min(args.max_unit_tokens, args.cpu_threads) < 1:
        parser.error("Invalid token, unit or thread budget")
    if args.stop_target is not None and args.stop_target <= args.start_target:
        parser.error("stop-target must exceed start-target")
    if args.stage in ("run", "capture") and args.input is None:
        parser.error("--input is required for model stages")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.stage == "pack":
        print(json.dumps({"review_archive": pack(args.output, args.archive)}))
        return
    if args.stage == "report":
        settings = read_json(args.output / "settings.json")
    else:
        import torch
        torch.set_num_threads(args.cpu_threads)
        with closing(CaptureReader(args.input)) as reader:
            settings = reader.json("settings.json")
            captured = reader.json("value_transport/capture_settings.json")
            if captured["model"] != settings["model"] or captured["responses"] != settings["responses"]:
                raise ValueError("Input capture/settings model or responses differ")
            start_stage(args.output / "protocol.json", protocol(args, settings), args.resume)
            write_json(args.output / "settings.json", settings)
            jobs = plan_units(args, settings, reader)
        execute(args, settings, jobs)
        copy_review_context(args.input, args.output)
    result = report(args.output, settings)
    write_json(args.output / "summary.json", result)
    result["review_archive"] = pack(args.output, args.archive)
    print(json.dumps(result, ensure_ascii=False))
    if args.stage in ("run", "capture") and result["status"] != "captured":
        raise SystemExit("Original-answer capture incomplete; inspect coverage.json in the review ZIP.")


if __name__ == "__main__":
    main()
