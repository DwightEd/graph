"""Review export must preserve results and work without loading model captures."""

import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from experiments.native_support.observable_pack import pack
from experiments.native_support.observable_run import main


def test_pack_only_is_read_only_and_full_capture_is_opt_in(tmp_path, capsys):
    destination = tmp_path / "observable"
    response = destination / "responses" / "0000"
    response.mkdir(parents=True)
    files = {"settings.json": json.dumps({"responses": [{"id": "11907"}]}).encode(),
             "protocol.json": b"{}", "annotations.json": b'{"11907": [0, 1]}',
             "tokens.csv": b"target,raw_route\n0,0.2\n",
             "responses/0000/scores.npz": b"saved scores",
             "responses/0000/neighbors.json": b"[]",
             "responses/0000/token_000000.npz": b"raw capture" * 10000,
             "responses/0000/state.npz": b"large state" * 10000,
             "responses/0000/timing_000000.json": b"{}"}
    for name, content in files.items():
        (destination / name).write_bytes(content)
    before = {name: (destination / name).stat().st_mtime_ns for name in files}
    full_archive = destination.with_name("observable_review.zip")
    full_archive.write_bytes(b"previous full archive")

    with patch("experiments.native_support.observable_run.capture", side_effect=AssertionError), \
            patch("experiments.native_support.observable_run.score", side_effect=AssertionError), \
            patch("experiments.native_support.observable_run.finish", side_effect=AssertionError), \
            patch("state_audit.model.load_model", side_effect=AssertionError):
        main(["--stage", "pack", "--output", str(destination)])
    result = json.loads(capsys.readouterr().out)
    with ZipFile(result["review_archive"]) as archive:
        assert archive.read("responses/0000/scores.npz") == files["responses/0000/scores.npz"]
        assert archive.read("annotations.json") == files["annotations.json"]
        manifest = json.loads(archive.read("review_manifest.json"))
        excluded = {entry["path"] for entry in manifest["excluded"]}
        assert "responses/0000/token_000000.npz" in excluded
        assert "responses/0000/state.npz" in excluded
        assert "responses/0000/state.npz" not in archive.namelist()
        assert not manifest["result_files_complete"]
        assert "summary.json" in manifest["missing_review_files"]
    assert result["archive_bytes"] == destination.with_name("observable_review_light.zip").stat().st_size
    assert full_archive.read_bytes() == b"previous full archive"
    assert result["excluded_bytes"] > 200000

    pack(destination, mode="full")
    with ZipFile(full_archive) as archive:
        assert archive.read("responses/0000/state.npz") == files["responses/0000/state.npz"]
    for name, content in files.items():
        assert (destination / name).read_bytes() == content
        assert (destination / name).stat().st_mtime_ns == before[name]


def test_archive_inside_result_directory_is_rejected(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        pack(tmp_path, tmp_path / "review.zip")
