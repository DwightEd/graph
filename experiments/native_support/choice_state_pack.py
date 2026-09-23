"""Bundle evaluated states and their signed capture evidence without rescoring."""

import io
import json
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from state_audit.storage import read_json

from .choice_cache import CaptureReader
from .transport_pack import compact_arrays, pack_schema

RESULT_FILES = (
    "settings.json", "scoring_protocol.json", "summary.json", "evaluation.json",
    "evaluation_status.json", "comparisons.json", "comparisons.csv", "tokens.csv",
    "onsets.csv", "high_risk_normals.csv", "recovery.csv", "ranking_audit.json",
    "state_audit.json",
)


def copy_results(packed, output, settings):
    missing = []
    for name in RESULT_FILES:
        path = output / name
        if path.is_file():
            packed.write(path, f"choice_state/{name}")
        else:
            missing.append(name)
    for index, response in enumerate(settings["responses"]):
        directory = Path("responses") / f"{index:04d}"
        scores = output / directory / "scores.npz"
        with np.load(scores, allow_pickle=False) as saved:
            target = np.arange(len(response["token_ids"]) - response["prompt_length"])
            expected = (target, response["prompt_length"] + target - 1,
                        response["token_ids"][response["prompt_length"]:])
            for name, values in zip(("target", "query", "token_id"), expected):
                if not np.array_equal(saved[name], values):
                    raise ValueError(f"{response['id']}: result {name} alignment differs")
        for name in ("scores.npz", "state.npz", "sources.json"):
            path = directory / name
            packed.write(output / path, f"choice_state/{path.as_posix()}")
    return missing


def copy_annotations(packed, output, annotations, evaluation_status):
    if evaluation_status != "evaluated":
        return False
    snapshot = output / "annotations.json"
    if snapshot.is_file():
        if annotations is not None and read_json(annotations) != read_json(snapshot):
            raise ValueError("--annotations differs from the evaluated annotation snapshot")
        annotations = snapshot
    elif annotations is None:
        # Older runs with --annotations kept the evaluated file outside output.
        annotations = Path(read_json(output / "evaluation.json")["annotations_path"])
    packed.write(annotations, "annotations.json")
    packed.write(annotations, "choice_state/annotations.json")
    return True


def copy_capture(packed, reader, index, response):
    prefix = f"value_transport/capture/{index:04d}"
    packed.writestr(f"{prefix}/sources.json", reader.bytes(f"{prefix}/sources.json"))
    baseline = f"value_transport/responses/{index:04d}/scores.npz"
    packed.writestr(baseline, reader.bytes(baseline))
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    common_fields = None
    for target in range(count):
        name = f"{prefix}/token_{target:06d}.npz"
        saved = reader.arrays(name)
        expected = (target, prompt + target - 1, response["token_ids"][prompt + target])
        if tuple(int(saved[key]) for key in ("target", "query", "token_id")) != expected:
            raise ValueError(f"{response['id']}/{target}: capture alignment differs")
        arrays = compact_arrays(saved, prompt)
        common_fields = set(arrays) if common_fields is None else common_fields.intersection(arrays)
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        packed.writestr(name, buffer.getvalue())
        if (target + 1) % 25 == 0 or target + 1 == count:
            print(f"\rpack {response['id']}: {target + 1}/{count}", end="", flush=True)
    print()
    return {"response_id": response["id"], "tokens": count, "capture_fields": sorted(common_fields)}


def write_bundle(packed, reader, output, annotations):
    settings = read_json(output / "settings.json")
    summary = read_json(output / "summary.json")
    protocol = read_json(output / "scoring_protocol.json")
    if settings != reader.json("settings.json"):
        raise ValueError("result/input settings differ; use the capture that produced these scores")
    capture = reader.json("value_transport/capture_settings.json")
    if protocol["capture"] != capture:
        raise ValueError("result/input capture protocols differ")
    packed.writestr("settings.json", reader.bytes("settings.json"))
    packed.writestr("value_transport/capture_settings.json", json.dumps(capture, ensure_ascii=False))
    missing = copy_results(packed, output, settings)
    labelled = copy_annotations(packed, output, annotations, summary["evaluation_status"])
    responses = [copy_capture(packed, reader, index, response)
                 for index, response in enumerate(settings["responses"])]
    schema = pack_schema()
    schema["captured_tokens"] = sum(response["tokens"] for response in responses)
    packed.writestr("audit_pack_schema.json", json.dumps(schema, indent=2))
    manifest = {
        "version": 1, "purpose": "joint_choice_state_and_signed_capture_review",
        "model_run": False, "scores_changed": False, "results_root": "choice_state",
        "annotations_included": labelled, "annotation_snapshot_present": (output / "annotations.json").is_file(),
        "evaluation_status": summary["evaluation_status"],
        "missing_result_files": missing, "responses": responses,
        "native_history_attention": "top_8_only_plus_exact_total_and_retained_mass",
        "root_token_contrasts": "all_input_roots_and_saved_candidates",
        "not_available": ["uncaptured_vocabulary_choices", "full_residual_vectors",
                          "semantic_fact_versus_grammar_labels", "FFN_pre_post_source_vectors"],
        "files": [{"name": item.filename, "bytes": item.file_size} for item in packed.infolist()],
    }
    packed.writestr("review_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def pack_results(source, output, archive=None, annotations=None):
    output = Path(output)
    archive = Path(archive) if archive is not None else output.with_name(output.name + "_review.zip")
    if archive.exists():
        raise FileExistsError(f"{archive} exists; choose a new --archive path")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with closing(CaptureReader(source)) as reader:
        with NamedTemporaryFile(dir=archive.parent, suffix=".partial.zip", delete=False) as staging:
            temporary = Path(staging.name)
        # A failed run is never published under the final archive filename.
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as packed:
                manifest = write_bundle(packed, reader, output, annotations)
            archive.hardlink_to(temporary)
        finally:
            if temporary.exists():
                temporary.unlink()
    result = {"path": str(archive), "bytes": archive.stat().st_size,
              "responses": len(manifest["responses"]),
              "captured_tokens": sum(row["tokens"] for row in manifest["responses"])}
    print(json.dumps({"review_archive": result}), flush=True)
    return result
