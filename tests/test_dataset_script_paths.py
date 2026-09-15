"""Exercise shell argument forwarding without running attention analysis."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


DEFAULT_LABELS = (
    "/share/home/tm902089733300000/a903202310/lys/"
    "data/RAGTruth/dataset/response.jsonl"
)


@pytest.mark.parametrize("script", ["evaluate.sh", "run_all.sh"])
@pytest.mark.parametrize("override", [None, "", "custom labels/response.jsonl"])
def test_dataset_path_is_forwarded_only_to_evaluation(tmp_path, script, override):
    repo = Path(__file__).resolve().parents[1]
    log = tmp_path / "commands.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    stub.chmod(0o755)
    env = os.environ.copy()
    for key in ("CACHE", "SPLIT", "POPULATION", "INDEX", "METADATA", "ANNOTATIONS",
                "ATTENTION_ROOT", "TRAIN_CACHE", "TEST_CACHE", "COMPLETED_ONLY"):
        env.pop(key, None)
    env.update(PY=str(stub), OUTPUT=str(tmp_path / "results"),
               COMMAND_LOG=str(log), BOOTSTRAP="0")
    if override is not None:
        env["ANNOTATIONS"] = override
    command = ["bash", str(repo / "experiments/unsupervised_token_graph" / script)]
    if script == "evaluate.sh":
        env["SPLIT"] = "train"
        command.append("--completed-only")
    subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    evaluation = [args for args in calls if "evaluate" in args]
    assert len(evaluation) == (1 if script == "evaluate.sh" else 2)
    expected_splits = ["train"] if script == "evaluate.sh" else ["train", "test"]
    for args, split in zip(evaluation, expected_splits):
        assert args[args.index("--annotations") + 1] == (override or DEFAULT_LABELS)
        assert args[args.index("--split") + 1] == split
    for args in calls:
        if "evaluate" not in args:
            assert "--annotations" not in args
            assert DEFAULT_LABELS not in args
    if script == "evaluate.sh":
        assert len(calls) == 1
        args = calls[0]
        assert "--completed-only" in args
        assert args[args.index("--output") + 1].endswith("evaluation_partial.json")
    else:
        assert len(calls) == 5  # One test invocation, then analysis/evaluation per split.
    assert not (tmp_path / "results").exists()  # The mock never writes checkpoints.
