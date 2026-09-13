"""Read-only progress and failure accounting; never reads RAGTruth annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(run):
    settings = json.loads((run / "settings.json").read_text())
    result = {
        "run": str(run.resolve()),
        "progress": json.loads((run / "progress.json").read_text())
        if (run / "progress.json").exists()
        else None,
        "expected_responses": sum(
            bool(line.strip())
            for line in Path(settings["input_path"]).read_text().splitlines()
        ),
        "stage_completed": {
            stage: len(list((run / stage).glob("*.json")))
            for stage in ("A", "B", "C", "D", "merge")
        },
        "question_status": Counter(),
        "question_failures": Counter(),
        "event_failures": Counter(),
        "source_answerability": Counter(),
        "verification_status": Counter(),
        "source_reader_errors": Counter(),
        "source_citation_errors": Counter(),
        "failed_self_checks": Counter(),
        "extraction_errors": Counter(),
        "atomic_frame_failures": Counter(),
        "cloze_failures": Counter(),
        "reader_A_request_accounting": Counter(),
        "atomic_budget_states": Counter(),
        "atomic_condition_failures": Counter(),
        "atomic_unprocessed_units": 0,
        "source_condition_states": Counter(),
        "contrast_status": Counter(),
        "word_states": Counter(),
        "risk_claims": 0,
        "risk_claims_with_valid_contrast": 0,
        "selected_questions": 0,
        "semantic_scored_words": 0,
        "all_words": 0,
        "label_join_performed": False,
    }
    for path in sorted((run / "A").glob("*.json")):
        data = json.loads(path.read_text())["data"]
        result["extraction_errors"].update(
            [data["extraction"]["reader_error"]]
            if "reader_error" in data["extraction"]
            else []
        )
        extraction = data["extraction"]
        result["reader_A_request_accounting"].update(data.get("reader_outcomes", {}))
        for failure in extraction.get("cloze_failures", []):
            result["cloze_failures"][failure.get("reason", "unknown")] += 1
        for raw in extraction.get("raw_frame_calls", []):
            if raw.get("reader_error"):
                result["extraction_errors"][raw["reader_error"]] += 1
        for failure in extraction.get("atomic_frame_failures", []):
            result["atomic_frame_failures"][failure.get("reason", "unknown")] += 1
        budget = extraction.get("atomic_budget", {})
        if budget:
            result["atomic_budget_states"][budget["status"]] += 1
            result["atomic_unprocessed_units"] += len(budget["unprocessed_unit_ids"])
        for question in data["questions"]:
            result["question_status"][question["status"]] += 1
            if question.get("reason"):
                result["question_failures"][question["reason"]] += 1
            if question.get("event_error"):
                result["event_failures"][question["event_error"]] += 1
            check = question.get("self_check", {})
            if question["status"] == "invalid_question" and check:
                for flag, expected in (
                    ("question_faithful", True),
                    ("all_conditions_preserved", True),
                    ("answer_leaked_in_question", False),
                ):
                    if check.get(flag) is not expected:
                        result["failed_self_checks"][flag] += 1
                if check.get("answer_quote") != question.get("answer_quote"):
                    result["failed_self_checks"]["answer_quote_mismatch"] += 1
                if check.get("reader_error"):
                    result["failed_self_checks"][check["reader_error"]] += 1
            result["contrast_status"][
                question.get("contrast", {}).get("status", "not_built")
            ] += 1
            for key, state in (
                ("source_answer", "answerability"),
                ("source_verification", "status"),
            ):
                prediction = question.get(key, {})
                if prediction:
                    result[
                        "source_answerability"
                        if key == "source_answer"
                        else "verification_status"
                    ][prediction.get(state, "missing")] += 1
                if prediction.get("reader_error"):
                    result["source_reader_errors"][prediction["reader_error"]] += 1
                if prediction.get("citation_error"):
                    result["source_citation_errors"][prediction["citation_error"]] += 1
                binding = prediction.get("atomic_condition_binding", {})
                if binding.get("valid") is False:
                    result["atomic_condition_failures"][binding["reason"]] += 1
                conditions = prediction.get("condition_checks", [])
                for condition in conditions if isinstance(conditions, list) else []:
                    if isinstance(condition, dict):
                        result["source_condition_states"][
                            condition.get("status", "missing")
                        ] += 1
        result["word_states"].update(w["coverage_state"] for w in data["words"])
        result["semantic_scored_words"] += sum(not w["abstain"] for w in data["words"])
        result["all_words"] += len(data["words"])
        result["risk_claims"] += data["risk_claims_total"]
        result["risk_claims_with_valid_contrast"] += data[
            "risk_claims_with_valid_contrast"
        ]
        result["selected_questions"] += len(data["native_selected_question_ids"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
