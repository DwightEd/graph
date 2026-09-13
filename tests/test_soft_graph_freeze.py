import json
from types import SimpleNamespace

import pytest

from route_graph import soft_graph_runner as runner
from route_graph.audit_artifacts import file_sha256


def test_settings_are_published_only_after_snapshot_and_resumption_verifies_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "code"
    (root / "route_graph").mkdir(parents=True)
    source = root / "route_graph/only.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(runner, "GRAPH", root)
    monkeypatch.setattr(runner, "code_files", lambda: {"route_graph/only.py": file_sha256(source)})
    monkeypatch.setattr(runner, "model_manifest", lambda path: [{"name": "fake", "sha256": "a" * 64}])
    inputs = tmp_path / "input.jsonl"; inputs.write_text("{}\n")
    output = tmp_path / "output"; output.mkdir()
    args = SimpleNamespace(inputs=inputs, output=output, observer_model=tmp_path / "model", reader_model=tmp_path / "model", evaluation_manifest=None)
    original = runner.shutil.copy2
    def interrupt_copy(*args, **kwargs):
        raise OSError("simulated interrupted copy")
    monkeypatch.setattr(runner.shutil, "copy2", interrupt_copy)
    with pytest.raises(OSError, match="interrupted"):
        runner.settings(args)
    assert not (output / "settings.json").exists()
    monkeypatch.setattr(runner.shutil, "copy2", original)
    saved = runner.settings(args)
    assert runner.settings(args) == saved
    snapshot = output / "executed_code/route_graph/only.py"
    snapshot.write_text("changed snapshot\n")
    with pytest.raises(ValueError, match="snapshot"):
        runner.settings(args)


def test_evaluation_precheck_rejects_changed_live_code_and_snapshot_manifest(tmp_path, monkeypatch):
    root = tmp_path / "code"
    root.mkdir()
    original = root / "test.py"; original.write_text("frozen\n")
    monkeypatch.setattr(runner, "GRAPH", root)
    output = tmp_path / "run"
    snapshot = output / "executed_code"; snapshot.mkdir(parents=True)
    (snapshot / "test.py").write_bytes(original.read_bytes())
    frozen = {"code_sha256": {"test.py": file_sha256(original)}}
    (snapshot / "manifest.json").write_text(json.dumps(frozen["code_sha256"]))
    runner.verify_executed_code(output, frozen)
    original.write_text("changed live code\n")
    with pytest.raises(ValueError, match="live code"):
        runner.verify_executed_code(output, frozen)
    (snapshot / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        runner.verify_executed_code(output, frozen)
