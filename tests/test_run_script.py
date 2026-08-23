from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "experiments" / "non_neural_structure_audit" / "run.sh"


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()
    drive = resolved.drive.removesuffix(":").lower()
    return f"/{drive}/{resolved.relative_to(resolved.anchor).as_posix()}"


def _bash_executable() -> str:
    executable = shutil.which("bash")
    if executable:
        return executable
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    pytest.skip("bash is required to test run.sh")


def _recorded_calls(path: Path) -> list[list[str]]:
    calls: list[list[str]] = []
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "__CALL__":
            current = []
        elif line == "__END__":
            assert current is not None
            calls.append(current)
            current = None
        else:
            assert current is not None
            current.append(line)
    assert current is None
    return calls


@pytest.mark.parametrize(
    ("train_limit", "test_limit", "expected_train", "expected_test"),
    [("", "", None, None), ("17", "9", "17", "9")],
)
def test_run_script_passes_only_nonempty_limits(
    tmp_path: Path,
    train_limit: str,
    test_limit: str,
    expected_train: str | None,
    expected_test: str | None,
) -> None:
    call_log = tmp_path / "calls.txt"
    fake_python = tmp_path / "fake-python.sh"
    fake_python.write_bytes(
        b"#!/usr/bin/env bash\n"
        b"printf '__CALL__\\n' >> \"$CALL_LOG\"\n"
        b'printf \'%s\\n\' "$@" >> "$CALL_LOG"\n'
        b"printf '__END__\\n' >> \"$CALL_LOG\"\n"
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "CALL_LOG": _bash_path(call_log),
            "DATA_ROOT": _bash_path(tmp_path / "data"),
            "OUTPUT_DIR": _bash_path(tmp_path / "output"),
            "PYTHON": _bash_path(fake_python),
            "SCOPE": "smoke",
            "TRAIN_LIMIT": train_limit,
            "TEST_LIMIT": test_limit,
        }
    )
    subprocess.run(
        [_bash_executable(), _bash_path(RUN_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = _recorded_calls(call_log)
    fit_call = next(call for call in calls if "fit" in call)
    score_call = next(call for call in calls if "score" in call)

    if expected_train is None:
        assert "--limit" not in fit_call
    else:
        assert fit_call[fit_call.index("--limit") + 1] == expected_train
    if expected_test is None:
        assert "--limit" not in score_call
    else:
        assert score_call[score_call.index("--limit") + 1] == expected_test
