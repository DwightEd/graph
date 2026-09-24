"""Package existing observable results without loading captures or changing scores."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


RESPONSE_FILES = {"scores.npz", "sources.json", "neighbors.json", "trajectory.png"}
REVIEW_FILES = ("settings.json", "protocol.json", "score_settings.json", "reference_fit.json",
                "summary.json", "tokens.csv", "operators.csv", "transitions.csv", "coverage.json")


def light_file(relative):
    if len(relative.parts) == 1:
        return relative.suffix in (".json", ".csv")
    return (len(relative.parts) == 3 and relative.parts[0] == "responses"
            and relative.name in RESPONSE_FILES)


def inventory(destination, mode):
    included, excluded = [], []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or ".partial." in path.name:
            continue
        relative = path.relative_to(destination)
        entries = included if mode == "full" or light_file(relative) else excluded
        entries.append(dict(path=relative.as_posix(), bytes=path.stat().st_size))
    return included, excluded


def review_manifest(destination, mode, included, excluded):
    settings = json.loads((destination / "settings.json").read_text(encoding="utf-8"))
    expected = list(REVIEW_FILES)
    for index in range(len(settings["responses"])):
        expected.extend(f"responses/{index:04d}/{name}" for name in ("scores.npz", "neighbors.json"))
    present = {entry["path"] for entry in included}
    missing = [name for name in expected if name not in present]
    return dict(mode=mode, result_files_complete=not missing, missing_review_files=missing,
                model_forward=False, rescoring=False, original_results_modified=False,
                scope="saved_results; raw_response_reconstruction_requires_full_capture",
                included_bytes=sum(entry["bytes"] for entry in included),
                excluded_bytes=sum(entry["bytes"] for entry in excluded),
                included=included, excluded=excluded)


def pack(destination, archive=None, mode="light"):
    """Light mode omits token tensors/state matrices; full mode keeps the capture."""
    destination = Path(destination)
    if not (destination / "protocol.json").is_file():
        raise FileNotFoundError(f"{destination}: no observable protocol to package")
    suffix = "_review_light.zip" if mode == "light" else "_review.zip"
    archive = Path(archive) if archive else destination.with_name(destination.name + suffix)
    if archive.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Review archive must be outside the result directory")

    included, excluded = inventory(destination, mode)
    manifest = review_manifest(destination, mode, included, excluded)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as output:
        for entry in included:
            output.write(destination / entry["path"], entry["path"])
        output.writestr("review_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(archive)
    return dict(review_archive=str(archive), archive_bytes=archive.stat().st_size,
                pack_mode=mode, excluded_bytes=manifest["excluded_bytes"],
                missing_review_files=manifest["missing_review_files"])
