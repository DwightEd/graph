from pathlib import Path

import pytest
import torch

from state_audit.capture import CaptureSpec, capture_run
from state_audit.demo import build_demo
from state_audit.generation import GenerationOptions, generate_run
from state_audit.model import load_model


@pytest.fixture(params=["llama", "mistral", "qwen2"])
def tiny_run(tmp_path: Path, request):
    torch.set_num_threads(1)
    data, checkpoint = build_demo(tmp_path / "fixture", request.param)
    model, tokenizer = load_model(str(checkpoint), "main", "cpu", "float32")
    options = GenerationOptions(
        mode="replay",
        template="chat",
        seed=7,
        temperature=0.0,
        top_p=0.9,
        max_new_tokens=8,
        max_length=256,
    )
    settings = dict(name=str(checkpoint), revision="main", device="cpu", dtype="float32")
    root = tmp_path / "run"
    generate_run(model, tokenizer, data, root, options, settings)
    capture_run(model, root, CaptureSpec())
    return model, tokenizer, root, data, options, settings
