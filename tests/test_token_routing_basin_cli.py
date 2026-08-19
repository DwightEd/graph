import os
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.token_routing_basin.main import parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "experiments" / "token_routing_basin" / "run.sh"


def test_fit_cli_keeps_method_parameters_in_one_entry_point():
    args = parse_args(
        [
            "fit",
            "--train-split",
            "train",
            "--output",
            "reference.npz",
            "--window",
            "5",
            "--smoothing-decay",
            "0.8",
        ]
    )
    assert args.command == "fit"
    assert args.window == 5
    assert args.smoothing_decay == pytest.approx(0.8)


def test_one_click_script_runs_fit_score_evaluate(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        bash = str(git_bash) if git_bash.is_file() else None
    if bash is None:
        pytest.skip("bash is unavailable")

    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir(parents=True)
    (data / "train" / "manifest.json").write_text("{}\n", encoding="utf-8")
    (data / "test" / "manifest.json").write_text("{}\n", encoding="utf-8")
    fake_python = tmp_path / "python.sh"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$CALL_LOG"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--output" ]]; then
    mkdir -p "$(dirname "${args[$((i+1))]}")"
    printf 'x\n' > "${args[$((i+1))]}"
  fi
  if [[ "${args[$i]}" == "--output-dir" ]]; then
    mkdir -p "${args[$((i+1))]}"
    printf '{}\n' > "${args[$((i+1))]}/report.json"
  fi
done
""",
        encoding="utf-8",
        newline="\n",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "output"
    log = tmp_path / "calls.log"
    completed = subprocess.run(
        [bash, RUN_SCRIPT.as_posix()],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "ROOT": data.as_posix(),
            "OUT": output.as_posix(),
            "PYTHON": fake_python.as_posix(),
            "DEVICE": "cpu",
            "LIMIT": "2",
            "CALL_LOG": log.as_posix(),
            "SKIP_TESTS": "1",
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "experiments.token_routing_basin.main fit" in calls[0]
    assert "experiments.token_routing_basin.main score" in calls[1]
    assert "experiments.token_routing_basin.main evaluate" in calls[2]
    assert all("--limit 2" in call for call in calls[:2])
    assert (output / "reference.npz").is_file()
    assert (output / "test_scores.npz").is_file()
    assert (output / "evaluation" / "report.json").is_file()
