"""A frozen, label-independent RAGTruth expansion with external train references."""

from copy import copy
from hashlib import sha256

from state_audit.storage import read_json, write_json

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
