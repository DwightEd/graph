from pathlib import Path
import subprocess
import sys

from experiments.attention_mechanism_audit.run import command_line


def test_cli_freezes_three_physical_stages_and_defaults():
    parser = command_line()
    roles = parser.parse_args(
        [
            "roles",
            "--data",
            "data",
            "--source-info",
            "source.jsonl",
            "--tokenizer",
            "model",
            "--output",
            "roles.jsonl",
        ]
    )
    capture = parser.parse_args(
        [
            "capture",
            "--data",
            "data",
            "--roles",
            "roles.jsonl",
            "--source-info",
            "source.jsonl",
            "--model",
            "model",
            "--output",
            "mechanisms.npz",
        ]
    )
    evaluation = parser.parse_args(
        [
            "evaluate",
            "--data",
            "data",
            "--artifact",
            "mechanisms.npz",
            "--output",
            "evaluation.json",
        ]
    )

    assert roles.command == "roles"
    assert capture.command == "capture"
    assert capture.torch_dtype == "auto"
    assert capture.vocab_chunk_size == 4096
    assert capture.gradient_probes == 8
    assert capture.role_null_bin_width == 32
    assert evaluation.bootstrap == 1000


def test_help_and_shell_syntax_work_without_torch_or_transformers():
    repository = Path(__file__).resolve().parents[3]
    help_result = subprocess.run(
        [sys.executable, "-m", "experiments.attention_mechanism_audit.run", "--help"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    shell_result = subprocess.run(
        ["bash", "-n", "experiments/attention_mechanism_audit/run_qa.sh"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert help_result.returncode == 0, help_result.stderr
    assert "roles" in help_result.stdout and "capture" in help_result.stdout
    assert shell_result.returncode == 0, shell_result.stderr
