"""A random local Llama is a software witness, never a detection result."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from main import main
from route_graph.capture import CaptureConfig, FrozenGraphCapture, file_digest
from route_graph.data import read_jsonl, text_digest, write_jsonl
from route_graph.detector import RouteDetector


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("route-llama")
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[PAD]",
                "[BOS]",
                "[EOS]",
                "[UNK]",
                "Question",
                "Evidence",
                "Gallery",
                "opens",
                "Monday",
                "Tuesday",
                "Answer",
                "Later",
                "Friday",
                "Sunday",
                ".",
                ":",
            ]
        )
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(17)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
        )
    ).save_pretrained(path)
    return path


def prepared_response(index, text="Gallery opens Monday . Later Friday ."):
    return {
        "schema": "route-graph/response@1",
        "id": str(index),
        "source_id": str(index),
        "task": "QA",
        "generator": "synthetic-fixture",
        "split": "test",
        "prompt": "Question Evidence Gallery opens Monday . Answer :",
        "source_span": [18, 40],
        "response": text,
        "response_sha256": text_digest(text),
    }


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_capture_includes_first_token_and_cannot_see_target_or_suffix(
    tiny_model, tmp_path, dtype
):
    input_path = tmp_path / "input.jsonl"
    write_jsonl(
        input_path,
        [
            prepared_response(0),
            prepared_response(1, "Gallery opens Tuesday . Later Sunday ."),
        ],
    )
    output = tmp_path / "capture"
    config = CaptureConfig(
        input_path, output, tiny_model, end_layers=(1,), max_tokens=128, dtype=dtype
    )
    result = FrozenGraphCapture(config).run()
    rows = read_jsonl(output / "features.jsonl")
    a = [row for row in rows if row["response_id"] == "0"]
    b = [row for row in rows if row["response_id"] == "1"]
    assert len(a) == len(b) == 7
    assert a[0]["token_index"] == 0
    assert a[0]["predictor_index"] == a[0]["prompt_tokens"] - 1
    for i in range(3):  # Includes the predictor of Monday vs Tuesday itself.
        assert a[i]["candidate_ids"] == b[i]["candidate_ids"]
        for key in ("observed", "null", "residual", "signal"):
            np.testing.assert_allclose(a[i][key], b[i][key], atol=1e-6)
    assert result["tokens"] == 14
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["labels_used"] is False
    assert manifest["model_type"] == "llama"
    assert len(manifest["feature_names"]) == len(a[0]["residual"])


def test_capture_rejects_context_overflow_before_producing_features(
    tiny_model, tmp_path
):
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, [prepared_response(0)])
    output = tmp_path / "overflow"
    with pytest.raises(ValueError, match="context"):
        FrozenGraphCapture(
            CaptureConfig(input_path, output, tiny_model, max_tokens=5)
        ).run()
    assert not (output / "manifest.json").exists()


def test_public_commands_run_frozen_model_to_full_stream_evaluation(
    tiny_model, tmp_path, capsys
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sources, responses = [], []
    for i in range(8):
        sources.append(
            {
                "source_id": str(i),
                "task_type": "QA",
                "prompt": "Question Evidence Gallery opens Monday . Answer :",
                "source_info": {"passages": "Gallery opens Monday ."},
            }
        )
        responses.append(
            {
                "id": str(i),
                "source_id": str(i),
                "model": "fixture",
                "response": "Gallery opens Monday . Later Friday .",
                "labels": [{"start": 0, "end": 7}],
            }
        )
    write_jsonl(dataset / "source_info.jsonl", sources)
    write_jsonl(dataset / "response.jsonl", responses)
    prepared, captured, detected = (
        tmp_path / "input.jsonl",
        tmp_path / "capture",
        tmp_path / "detection",
    )
    main(
        [
            "prepare",
            "--dataset",
            str(dataset),
            "--output",
            str(prepared),
            "--generator",
            "fixture",
            "--max-sources",
            "8",
        ]
    )
    main(
        [
            "extract",
            "--input",
            str(prepared),
            "--output",
            str(captured),
            "--model",
            str(tiny_model),
            "--end-layers",
            "1",
            "--max-tokens",
            "128",
        ]
    )
    main(
        [
            "detect",
            "--features",
            str(captured),
            "--output",
            str(detected),
            "--neighbors",
            "2",
        ]
    )
    main(
        [
            "evaluate",
            "--scores",
            str(detected / "scores.jsonl"),
            "--labels",
            str(dataset / "response.jsonl"),
            "--output",
            str(tmp_path / "evaluation.json"),
            "--bootstrap",
            "4",
        ]
    )
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["subsets"]["all_tokens"]["tokens"] == 28
    assert report["subsets"]["first_error"]["positives"] == 4
    assert set(report["subsets"]["all_tokens"]["metrics"]) >= {
        "residual",
        "null",
        "signal",
        "residual_knn",
    }
    assert (detected / "reference.json").exists()
    # A self-consistent digest cannot substitute for complete reference streams.
    feature_path = captured / "features.jsonl"
    features = read_jsonl(feature_path)
    removed = next(
        row for row in features if row["split"] == "train" and row["token_index"] == 0
    )
    features.remove(removed)
    feature_path.write_text("\n".join(map(json.dumps, features)), encoding="utf-8")
    manifest_path = captured / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(tokens=len(features), features_sha256=file_digest(feature_path))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete capture"):
        RouteDetector(captured, tmp_path / "incomplete-detection", neighbors=2).run()


def test_remote_script_runs_all_stages_from_another_directory(tiny_model, tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = (
        str(git_bash) if os.name == "nt" and git_bash.exists() else shutil.which("bash")
    )
    if not bash:
        pytest.skip("bash is required for the remote runner test")
    script = Path(__file__).resolve().parents[1] / "scripts/run_route_evaluation.sh"
    dataset = tmp_path / "ragtruth"
    source_rows, response_rows = [], []
    for i in range(8):
        row = prepared_response(i)
        start, end = row["source_span"]
        source_rows.append(
            {
                "source_id": str(i),
                "task_type": "QA",
                "prompt": row["prompt"],
                "source_info": {"passages": row["prompt"][start:end]},
            }
        )
        response_rows.append(
            {
                "id": str(i),
                "source_id": str(i),
                "model": "fixture",
                "response": row["response"],
                "labels": [{"start": 0, "end": 7}],
            }
        )
    write_jsonl(dataset / "source_info.jsonl", source_rows)
    write_jsonl(dataset / "response.jsonl", response_rows)
    output = tmp_path / "remote result"
    env = dict(
        os.environ,
        PYTHON_BIN=Path(sys.executable).as_posix(),
        MODEL_PATH=tiny_model.as_posix(),
        RAGTRUTH_DIR=dataset.as_posix(),
        OUTPUT_DIR=output.as_posix(),
        GENERATOR="fixture",
        MAX_SOURCES="8",
        DEVICE="cpu",
        DTYPE="float32",
        END_LAYERS="1",
        MAX_TOKENS="128",
        MAX_ATTENTION_MB="64",
        NEIGHBORS="2",
        BOOTSTRAP="2",
    )
    command = [bash, script.as_posix()]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = output.with_suffix(".log").read_text()
    for stage in (
        "capture responses",
        "route tokens",
        "score tokens",
        "evaluate subsets",
    ):
        assert stage in log
    report = json.loads((output / "evaluation.json").read_text())
    assert report["subsets"]["all_tokens"]["tokens"] == 28
    assert (output / "COMPLETE").is_file()
    assert output.with_suffix(".log").is_file()
    before = (output / "detection/scores.jsonl").read_bytes()
    repeated = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert repeated.returncode != 0
    assert (output / "detection/scores.jsonl").read_bytes() == before
    failed_output = tmp_path / "failed result"
    env.update(OUTPUT_DIR=failed_output.as_posix(), MAX_TOKENS="1")
    failed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert failed.returncode != 0
    assert not (failed_output / "COMPLETE").exists()
    assert not (failed_output / "evaluation.json").exists()
    assert "context limit exceeded" in failed_output.with_suffix(".log").read_text()
