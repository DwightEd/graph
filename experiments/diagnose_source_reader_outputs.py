"""Read-only source-reader barrier counts; never repair predictions or join labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from route_graph.frozen_reader import write_json_once
from route_graph.json_framing import parse_framed_json


def diagnose(run):
    counts, files = Counter(), {}
    for path in sorted((run / "reader_cache").glob("*.json")):
        raw_file = path.read_bytes()
        saved = json.loads(raw_file)
        request = saved["request"]
        payload = request["payload"]
        if request["labels"] or not {"source", "conditions", "question"} <= payload.keys():
            continue
        files[path.name] = hashlib.sha256(raw_file).hexdigest()
        counts["source_requests"] += 1
        prediction = saved["prediction"]
        diagnostic_prediction = prediction
        if "reader_error" in prediction:
            counts["frozen_" + prediction["reader_error"]] += 1
            raw = saved.get("raw_output") or ""
            # Only count this possible framing change. Never publish it back to
            # the cache, A artifacts or evaluator as a replacement prediction.
            if raw.rstrip().endswith("."):
                try:
                    diagnostic_prediction = parse_framed_json(raw.rstrip()[:-1])["prediction"]
                    counts["intact_root_if_only_final_period_removed"] += 1
                except (ValueError, TypeError):
                    pass
        else:
            counts["frozen_parsed"] += 1
        if "reader_error" in diagnostic_prediction:
            continue
        counts["diagnostic_intact_roots"] += 1
        p = diagnostic_prediction
        if p.get("answerability") == "answerable":
            counts["diagnostic_answerable"] += 1
            answer, quote = p.get("source_answer"), p.get("source_answer_quote")
            counts["diagnostic_answerable_without_nonempty_answer"] += not isinstance(answer, str) or not answer.strip()
            counts["diagnostic_answerable_without_nonempty_quote"] += not isinstance(quote, str) or not quote.strip()
            if isinstance(quote, str) and quote.strip():
                counts["diagnostic_answerable_quote_present_in_raw_source"] += quote in payload["source"]
        quotes = p.get("evidence_quotes", [])
        if isinstance(quotes, list):
            strings = [quote for quote in quotes if isinstance(quote, str) and quote]
            counts["diagnostic_with_nonempty_evidence_quotes"] += bool(strings)
            counts["diagnostic_with_evidence_quote_absent_from_raw_source"] += any(q not in payload["source"] for q in strings)
            counts["diagnostic_with_mask_literal_in_evidence_quote"] += any("<MISSING_SLOT>" in q for q in strings)
    return {
        "schema": "source-reader-barrier-diagnostic@1", "run": str(run.resolve()),
        "settings_sha256": hashlib.sha256((run / "settings.json").read_bytes()).hexdigest(),
        "progress_snapshot": json.loads((run / "progress.json").read_text()),
        "counts": dict(counts), "observed_source_cache_files": files,
        "labels_used": False, "predictions_modified": False,
        "meaning": "request-level syntax/necessary-field barriers only; no factual accuracy or recovered prediction results",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run)
    if args.output:
        write_json_once(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "observed_source_cache_files"}, indent=2))
