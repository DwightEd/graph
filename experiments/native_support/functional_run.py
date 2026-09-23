"""Prepare phrase banks, measure native functional paths, report and auto-pack."""

import argparse
import json
from collections import Counter
from contextlib import closing
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from state_audit.storage import read_json, start_stage, write_arrays, write_json
from tqdm import tqdm

from .choice_cache import CaptureReader
from .functional_bank import answer_units, prepare_bank
from .functional_report import report
from .inputs import validate_tokenizer


def protocol(args, settings):
    return dict(version=2, model=settings["model"], responses=settings["responses"],
        response_ids=args.response_ids, start_target=args.start_target, stop_target=args.stop_target,
        max_units=args.max_units, max_unit_tokens=args.max_unit_tokens, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens, purpose="native_phrase_function_audit",
        selection="token_aligned_punctuation_units_with_list_markers; not_detected_reanchors",
        candidate_generation="observer_proposal_and_semantic_check; no_truth_judgment",
        candidate_context="answer_prefix_and_unit_only; original_native_prefix_unchanged",
        candidate_control_tokens="JSON_unicode_escaped_in_quoted_data",
        candidate_repair="one_feedback_retry_for_structural_or_semantic_failure",
        gradients="native_QK_RMS_SwiGLU; whole_phrase_objective",
        root_readout="gradient_dot_embedding; local_sensitivity_not_additive_ledger",
        head_readout="receiver_OV_message_sensitivity; direct_residual_plus_current_FFN_mediated",
        head_boundary="fixed_observed_prefix_query; no_alignment_of_different_candidate_suffixes",
        head_query_scope="all_observed_queries; alternative_branches_boundary_only",
        normalization="terminal_FFN_float32_RMS_margin_identity; native_roundoff_separate",
        semantic_probability="softmax_of_sequence_log_scores_in_finite_bank; not_exhaustive_LM_meanings",
        pre_ffn_readout="final_RMS_and_unembedding_on_native_pre_FFN_states; not_intervened_rollout",
        history="evaluate_all_prefix_roots_against_current_phrase; no_cross_token_support_recursion",
        prediction_alignment="query=P+start-1; per_branch_causal_masks",
        semantic_assignment_independently_verified=False, candidate_coverage_known=False,
        labels_used_for_scoring=False, labels_used_for_training=False, parameter_fitting=False,
        future_tokens_used=True, interventions=False, automatic_method_selection=False,
        source_types="existing_automatic_blocks; semantic_roles_unassigned",
        observer_scope="teacher_forced_observer; may_differ_from_original_generator",
        detector_evaluation=False, existing_scores_modified=False)


def plan_units(args, settings, reader):
    jobs = []
    found = set()
    for index, response in enumerate(settings["responses"]):
        if args.response_ids and response["id"] not in args.response_ids:
            continue
        found.add(response["id"])
        directory = args.output / "responses" / f"{index:04d}"
        sources = reader.json(f"value_transport/capture/{index:04d}/sources.json")
        groups = sources["group_ids"]
        if len(groups) != len(response["token_ids"]) or min(groups) < 0 or max(groups) >= len(sources["blocks"]) + 3:
            raise ValueError(f"{response['id']}: source groups do not match saved inputs")
        start_stage(directory / "sources.json", sources, args.resume)
        selected = 0
        for start, stop in answer_units(response):
            if stop <= args.start_target or (args.stop_target is not None and start >= args.stop_target):
                continue
            if args.max_units and selected >= args.max_units:
                break
            selected += 1
            unit = directory / f"unit_{start:06d}"
            identity = dict(response_index=index, response_id=response["id"],
                            source_id=response["source_id"], start=start, stop=stop)
            if stop - start > args.max_unit_tokens:
                write_json(unit / "skipped.json", {**identity, "reason": "unit_token_budget"})
            else:
                jobs.append((response, sources, unit, identity))
    if args.response_ids and found != set(args.response_ids):
        raise ValueError(f"Unknown response IDs: {set(args.response_ids) - found}")
    return jobs


def capture_bank(model, response, sources, bank, directory, identity):
    import torch
    from state_audit.functional_capture import capture_candidate

    prefix = response["token_ids"][:response["prompt_length"] + bank["start"]]
    if prefix != bank["prefix_ids"]:
        raise ValueError(f"{directory}: bank prefix no longer matches observed input")
    start_stage(directory / "captured_bank.json", bank, resume=True)
    groups = sources["group_ids"][:len(prefix)]
    group_count = len(sources["blocks"]) + 3
    started = perf_counter()
    device = model.native.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, candidate in enumerate(tqdm(bank["candidates"], desc="phrase candidates", leave=False)):
        path = directory / f"candidate_{index:02d}.npz"
        if path.is_file():
            continue
        if len(prefix) + len(candidate["token_ids"]) > model.native.config.max_position_embeddings:
            raise ValueError(f"{directory}: candidate exceeds model context; no truncation")
        arrays = capture_candidate(model, prefix, candidate["token_ids"], groups, group_count,
                                   all_head_queries=index == 0)
        write_arrays(path, **arrays)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    write_json(directory / "complete.json", {**identity,
        "capture_seconds_this_run": perf_counter() - started, "peak_cuda_bytes": peak})


def execute(args, settings, jobs):
    from state_audit.model import load_model

    model, tokenizer = None, None
    for response, sources, directory, identity in tqdm(jobs, desc="functional units"):
        if (directory / "complete.json").is_file():
            continue
        bank_path = directory / "bank.json"
        bank = read_json(bank_path) if bank_path.is_file() else None
        if bank is not None and (not bank["valid"] or args.stage == "prepare"):
            continue
        if args.stage == "capture" and bank is None:
            raise ValueError(f"{directory}: prepare the automatic bank first")
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        validate_tokenizer(response, tokenizer)
        if bank is None:
            bank = prepare_bank(model, tokenizer, response, identity["start"], identity["stop"],
                                directory, args.max_new_tokens)
        if not bank["valid"]:
            tqdm.write(f"{response['id']} [{identity['start']}:{identity['stop']}]: "
                       f"bank rejected ({bank['reason']})")
        if bank["valid"] and args.stage != "prepare":
            capture_bank(model, response, sources, bank, directory, identity)


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


def summarize(destination, settings):
    result = report(destination, settings)
    banks = [read_json(path) for path in destination.glob("responses/*/unit_*/bank.json")]
    result.update(proposed_units=len(banks), accepted_units=sum(bank["valid"] for bank in banks),
                  rejected_units=sum(not bank["valid"] for bank in banks),
                  rejection_reasons=dict(Counter(bank["reason"] for bank in banks if not bank["valid"])),
                  budget_skipped_units=len(list(destination.glob("responses/*/unit_*/skipped.json"))))
    if result["accepted_units"] == 0:
        result["status"] = "no_valid_banks"
    elif result["completed_units"] == 0:
        result["status"] = "banks_prepared"
    elif result["completed_units"] < result["accepted_units"]:
        result["status"] = "partial_capture"
    else:
        result["status"] = "captured"
    write_json(destination / "summary.json", result)
    return result


def copy_review_context(source, destination):
    """Copy label/baseline bytes only AFTER measurement; never use them in readouts."""
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
    parser.add_argument("--output", type=Path, required=True, help="Separate new functional audit directory")
    parser.add_argument("--stage", choices=("run", "prepare", "capture", "report", "pack"), default="run")
    parser.add_argument("--response-ids", nargs="+")
    parser.add_argument("--start-target", type=int, default=0)
    parser.add_argument("--stop-target", type=int)
    parser.add_argument("--max-units", type=int, default=0, help="0 means all selected punctuation units")
    parser.add_argument("--max-unit-tokens", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Bank proposal/check budget")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    if min(args.start_target, args.max_units) < 0 or min(args.max_unit_tokens, args.max_new_tokens, args.cpu_threads) < 1:
        parser.error("Invalid token, unit or thread budget")
    if args.stop_target is not None and args.stop_target <= args.start_target:
        parser.error("stop-target must exceed start-target")
    if args.stage in ("run", "prepare", "capture") and args.input is None:
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
    result = summarize(args.output, settings)
    if args.stage in ("run", "capture"):
        copy_review_context(args.input, args.output)
    result["review_archive"] = pack(args.output, args.archive)
    print(json.dumps(result, ensure_ascii=False))
    if args.stage in ("run", "prepare", "capture") and result["accepted_units"] == 0:
        raise SystemExit("No valid candidate banks; review archive saved, but no functional measurement. "
                         "Inspect proposal/validation files before retrying.")


if __name__ == "__main__":
    main()
