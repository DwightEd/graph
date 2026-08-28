import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.attention_mechanism_audit.run import command_line


PRIMARY_FEATURES = (
    "drift_functional_history_to_grounding_log_ratio"
    "__layer_mean__late_minus_early",
    "dispersion_functional_entropy_observed__layer_mean__late_minus_early",
    "dispersion_functional_cancellation__layer_mean__late_minus_early",
    "routing_entropy_upper__layer_mean__late_minus_early",
    "routing_total_evidence_ancestry__layer_mean__late_minus_early",
    "counterfactual_evidence_bypass__mean",
)


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


def test_shell_does_not_use_global_set_failure_flags():
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "experiments" / "attention_mechanism_audit" / "run_qa.sh"
    ).read_text(encoding="utf-8")

    assert not any(
        line.strip().startswith("set -") for line in script.splitlines()
    )
    assert "pipefail" not in script


def test_shell_preserves_capture_failure_and_never_opens_evaluation(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    source_info = tmp_path / "source_info.jsonl"
    test_split = tmp_path / "test"
    model_path = tmp_path / "model"
    output = tmp_path / "output"
    role_index = output / "prompt_roles.jsonl"
    artifact = output / "mechanisms.npz"
    evaluation = output / "evaluation.json"
    calls = tmp_path / "calls.txt"
    evaluate_marker = tmp_path / "evaluate_called"
    json_marker = tmp_path / "json_render_called"
    fake_python = tmp_path / "fake_python"

    source_info.write_text("{}\n", encoding="utf-8")
    test_split.mkdir()
    model_path.mkdir()
    output.mkdir()
    stale_evaluation = '{"schema":"old-report","sentinel":"STALE_JSON"}\n'
    evaluation.write_text(stale_evaluation, encoding="utf-8")

    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[:2] == [\n"
        "    '-m',\n"
        "    'experiments.attention_mechanism_audit.run',\n"
        "] and len(arguments) >= 3:\n"
        "    command = arguments[2]\n"
        "elif arguments and arguments[0] == '-':\n"
        "    command = 'json-render'\n"
        "else:\n"
        "    command = 'unexpected'\n"
        "\n"
        "with Path(os.environ['FAKE_CALLS']).open(\n"
        "    'a', encoding='utf-8'\n"
        ") as file:\n"
        "    file.write(command + '\\n')\n"
        "\n"
        "if command == 'roles':\n"
        "    output = Path(arguments[arguments.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text('{}\\n', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if command == 'capture':\n"
        "    print('Traceback (most recent call last):', file=sys.stderr)\n"
        "    print('  File fake_capture.py, line 1, in capture', file=sys.stderr)\n"
        "    print('RuntimeError: sentinel capture failure', file=sys.stderr)\n"
        "    raise SystemExit(37)\n"
        "\n"
        "if command == 'evaluate':\n"
        "    Path(os.environ['EVALUATE_MARKER']).write_text(\n"
        "        'called', encoding='utf-8'\n"
        "    )\n"
        "    raise SystemExit(0)\n"
        "\n"
        "if command == 'json-render':\n"
        "    Path(os.environ['JSON_MARKER']).write_text(\n"
        "        'called', encoding='utf-8'\n"
        "    )\n"
        "    raise SystemExit(0)\n"
        "\n"
        "raise SystemExit(99)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "REPO": str(repository),
            "TEST_SPLIT": str(test_split),
            "SOURCE_INFO": str(source_info),
            "MODEL_PATH": str(model_path),
            "TOKENIZER_PATH": str(model_path),
            "PYTHON": str(fake_python),
            "DEVICE": "cpu",
            "OUT": str(output),
            "ROLE_INDEX": str(role_index),
            "ARTIFACT": str(artifact),
            "EVALUATION": str(evaluation),
            "START_STAGE": "1",
            "FORCE_ROLES": "1",
            "FAKE_CALLS": str(calls),
            "EVALUATE_MARKER": str(evaluate_marker),
            "JSON_MARKER": str(json_marker),
        }
    )

    result = subprocess.run(
        ["bash", "experiments/attention_mechanism_audit/run_qa.sh"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 37
    assert "Traceback (most recent call last):" in result.stderr
    assert "RuntimeError: sentinel capture failure" in result.stderr
    assert "failed with exit code 37" in result.stderr
    assert "[3/3]" not in result.stdout
    assert "STALE_JSON" not in result.stdout
    assert calls.read_text(encoding="utf-8").splitlines() == ["roles", "capture"]
    assert not artifact.exists()
    assert evaluation.read_text(encoding="utf-8") == stale_evaluation
    assert not evaluate_marker.exists()
    assert not json_marker.exists()


def test_embedded_report_renderer_accepts_only_the_mechanism_schema(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "experiments" / "attention_mechanism_audit" / "run_qa.sh"
    ).read_text(encoding="utf-8")
    start = '\"${PYTHON}\" - \"${EVALUATION}\" <<\'PY\'\n'
    renderer = script.split(start, 1)[1].split("\nPY\n}", 1)[0]

    rows = []
    increments = {}
    for index, name in enumerate(PRIMARY_FEATURES):
        direction = "low" if name.startswith("routing_total_evidence") else "high"
        rows.append(
            {
                "feature": name,
                "direction": direction,
                "oriented": {"auroc": 0.6 + index / 100, "auprc": 0.4},
                "source_group_permutation": {
                    "mean_positive_minus_negative": -0.1
                    if direction == "low"
                    else 0.1,
                    "p_value_two_sided": 0.01,
                },
                "source_group_permutation_fdr_q": 0.02,
            }
        )
        increments[name] = {
            "available": True,
            "auroc_delta": 0.03,
            "auprc_delta": 0.02,
        }
    report = {
        "schema": "attention-hallucination-mechanism-answer-evaluation",
        "samples": 149,
        "positive_answers": 52,
        "prevalence": 52 / 149,
        "primary_answer_feature_names": list(PRIMARY_FEATURES),
        "primary_answer_univariate": rows,
        "primary_feature_length_increment": increments,
        "token_onset_diagnostics": {
            "drift": {
                "responses_with_first_onset": 20,
                "source_disjoint_same_position_matches": 18,
                "mean_onset_minus_matched_non_onset_delta": 0.2,
                "source_bootstrap": {"ci_low": 0.1, "ci_high": 0.3},
            }
        },
        "mechanism_observability": {"functional_contribution": True},
        "claim_boundary": "post-hoc mechanism test",
    }
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(report), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-", str(evaluation)],
        input=renderer,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ATTENTION HALLUCINATION MECHANISM AUDIT" in result.stdout
    assert "SIX FROZEN PRIMARY MECHANISM TESTS" in result.stdout
    assert "GROUPED PROBES" not in result.stdout

    report["schema"] = "attention-operator-answer-mechanism-evaluation"
    evaluation.write_text(json.dumps(report), encoding="utf-8")
    old_schema = subprocess.run(
        [sys.executable, "-", str(evaluation)],
        input=renderer,
        text=True,
        capture_output=True,
        check=False,
    )

    assert old_schema.returncode != 0
    assert "refusing to print an old operator-validation report" in old_schema.stderr
