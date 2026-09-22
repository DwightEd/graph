"""A frozen, label-independent RAGTruth expansion with external train references."""

from copy import copy
from hashlib import sha256

from state_audit.dataset.jsonl import read_jsonl
from state_audit.storage import read_json, start_stage, write_json

from .ragtruth import encode_response


def choose_sources(rows, sources, task, split, generator, count, excluded, seed):
    candidates = [row for row in rows if row["split"] == split and row["model"] == generator
                  and sources[str(row["source_id"])]["task_type"] == task
                  and str(row["source_id"]) not in excluded]
    candidates.sort(key=lambda row: (sha256(f"{seed}:{row['source_id']}".encode()).hexdigest(), str(row["id"])))
    selected, seen = [], set()
    for row in candidates:
        source = str(row["source_id"])
        if source not in seen:
            selected.append(row)
            seen.add(source)
        if len(selected) == count:
            return selected
    raise ValueError(f"Requested {count} independent {split} sources, found {len(selected)}")


def select_cohorts(args, rows, sources):
    inspected = read_json(args.exclude_output / "settings.json")
    excluded = {str(row["source_id"]) for row in inspected["responses"]}
    reference = choose_sources(rows, sources, args.task, "train", args.generator,
                               args.reference_count, excluded, args.selection_seed)
    reference_sources = {str(row["source_id"]) for row in reference}
    target = choose_sources(rows, sources, args.task, "test", args.generator,
                            args.limit, excluded | reference_sources, args.selection_seed)
    return {"reference": reference, "test": target}, sorted(excluded)


def prepare_cohort(args, name, rows, sources, tokenizer, model_name, excluded):
    from .run import save_settings

    current = copy(args)
    current.output = args.output / name
    responses, annotations = [], {}
    for row in rows:
        response, annotation = encode_response(row, sources[str(row["source_id"])], tokenizer)
        responses.append(response)
        annotations[response["id"]] = annotation
    cohort = {"dataset": str(args.dataset.resolve()), "task": args.task,
              "split": "train" if name == "reference" else "test", "generator": args.generator,
              "observer_model": model_name, "selection": "seeded_source_order_one_answer_per_source",
              "selection_seed": args.selection_seed, "labels_used_for_selection": False,
              "labels_used_for_scoring": False, "response_ids": [r["id"] for r in responses],
              "excluded_source_ids": excluded, "scope": "frozen_small_expansion_not_full_dataset"}
    settings = save_settings(current, model_name, responses, cohort)
    write_json(current.output / "input.json", {"model": model_name, "responses": responses})
    write_json(current.output / "annotations.json", annotations)
    return current, settings


def validation_plan(args, groups, excluded, model_name):
    return {"dataset": str(args.dataset.resolve()), "model": model_name, "task": args.task,
            "generator": args.generator, "selection_seed": args.selection_seed, "window": args.window,
            "labels_used_for_selection": False, "excluded_source_ids": excluded,
            "selection": "sha256(seed:source_id); one answer per source; no class balancing",
            "groups": {name: [{"response_id": str(row["id"]), "source_id": str(row["source_id"])}
                               for row in rows] for name, rows in groups.items()},
            "primary_baseline": "route_mean", "candidate": "joint_observed",
            "automatic_method_selection": False,
            "scope": "excludes_supplied_inspected_sources; does_not_certify_unseen_in_all_historical_work"}


def run_validation(args):
    from transformers import AutoTokenizer

    from .run import run_capture
    from .state_inputs import load_features
    from .state_model import run_state_model
    from .state_readout import run_readout

    sources = {str(row["source_id"]): row for row in read_jsonl(args.dataset / "source_info.jsonl")}
    groups, excluded = select_cohorts(args, read_jsonl(args.dataset / "response.jsonl"), sources)
    model_name = args.model or read_json(args.input)["model"]
    plan = validation_plan(args, groups, excluded, model_name)
    start_stage(args.output / "validation_plan.json", plan, args.resume)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    prepared = {name: prepare_cohort(args, name, rows, sources, tokenizer, model_name, excluded)
                for name, rows in groups.items()}
    if args.prepare_only:
        return {"status": "prepared", "model_run": False, **plan}
    for current, settings in prepared.values():
        run_capture(current, settings)
        load_features(current.output, settings)
    target_args, target_settings = prepared["test"]
    run_state_model(target_args.output, target_settings, window=args.window, reference_output=args.output / "reference")
    result = run_readout(target_args.output, window=args.window)
    result["validation_plan"] = str(args.output / "validation_plan.json")
    write_json(args.output / "summary.json", result)
    return result
