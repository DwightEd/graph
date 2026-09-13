"""Prepare all source-pointer contrasts from a completed frozen A/B stage.

CPU only; no labels, reader calls or native forwards. Reuses the exact B pool.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from next_iteration.source_pointer_contrast import prepare_contrast
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once
from route_graph.soft_graph_runner import read_artifact, rows, verify_executed_code


def prepare_run(run, output):
    run = run.resolve()
    frozen = json.loads((run / "settings.json").read_text())
    verify_executed_code(run, frozen)
    inputs = Path(frozen["input_path"])
    if file_sha256(inputs) != frozen["input_sha256"]:
        raise ValueError("frozen input bytes changed")
    population = rows(argparse.Namespace(inputs=inputs))
    if any(not (run / stage / f"{row['id']}.json").is_file()
           for row in population for stage in ("A", "B")):
        raise ValueError("all frozen responses need completed A and B artifacts")
    files = [Path(__file__).with_name(n) for n in
             ("__init__.py", "source_pointer_prepare.py", "source_pointer_contrast.py", "reader_receipt.py")]
    code = {str(p.resolve()): file_sha256(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "executed_code"
    snapshot.mkdir()
    for path in files:
        shutil.copyfile(path, snapshot / path.name)
        if file_sha256(snapshot / path.name) != code[str(path.resolve())]:
            raise ValueError("prospective code changed while freezing snapshot")
    attempts, counts, references = [], Counter(), {}
    for row in population:
        verified = {}
        a = read_artifact(run, "A", row, frozen, verified)
        b = read_artifact(run, "B", row, frozen, verified)
        for stage in ("A", "B"):
            path = run / stage / f"{row['id']}.json"
            references[str(path)] = file_sha256(path)
        counts["responses"] += 1
        counts["original_words"] += len(row["response"].split())
        for claim in b["claims"]:
            counts["claims"] += 1
            counts["claims_without_B_source_terms"] += int(not claim["source_terms"])
            for index in range(len(claim["source_terms"])):
                draft = prepare_contrast(row, a, b, claim["claim_id"], index)
                attempts.append(draft)
                counts["attempts"] += 1
                counts["status:" + draft["status"]] += 1
    result = {"schema": "source-pointer-contrast-preparation@1", "parent_run": str(run),
              "parent_settings_file_sha256": file_sha256(run / "settings.json"),
              "parent_settings_sha256": digest(frozen), "input_sha256": frozen["input_sha256"],
              "input_path": str(inputs), "parent_artifacts": references,
              "code_sha256": code,
              "counts": dict(counts), "native_forward_calls": 0, "reader_calls": 0,
              "labels_read": False, "not_ground_truth": True,
              "candidate_selection": "every_previously_frozen_B_source_term; no_new_search",
              "status": "prepared_candidates_not_semantically_verified"}
    # Publish the manifest last; interrupted output is retained, never repaired.
    verify_executed_code(run, frozen)
    if any(file_sha256(p) != checksum or file_sha256(snapshot / Path(p).name) != checksum
           for p, checksum in code.items()):
        raise ValueError("prospective live/snapshot code changed during preparation")
    drafts = output / "drafts.jsonl"
    with drafts.open("x") as stream:
        for draft in attempts:
            stream.write(json.dumps(draft, ensure_ascii=False, allow_nan=False) + "\n")
    result["drafts_sha256"] = file_sha256(drafts)
    if any(file_sha256(p) != checksum for p, checksum in code.items()):
        raise ValueError("prospective code changed while publishing drafts")
    write_json_once(output / "manifest.json", result)
    return {k: v for k, v in result.items() if k not in {"parent_artifacts", "code_sha256"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_run(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
