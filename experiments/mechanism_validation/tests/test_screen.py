import json

import torch

from cache import AttentionSample
from experiments.mechanism_validation.screen import MechanismScreen


class OneSampleDataset:
    def __iter__(self):
        diagonal = torch.zeros((1, 1, 3), dtype=torch.float16)
        yield type("ResearchSample", (), {
            "sample_id": "sample", "source_id": "source",
            "task_type": "QA", "data_source": "MARCO",
            "attention": lambda self: AttentionSample(
                "sample", "source", 1, torch.arange(3, dtype=torch.int32), diagonal,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([.4, .3], dtype=torch.float16), .01,
            ),
            "release_attention": lambda self: None,
        })()


def test_screen_streams_one_artifact_per_response_without_labels(tmp_path):
    result = MechanismScreen(OneSampleDataset(), tmp_path).run()

    assert result == {"responses": 1, "tokens": 2}
    with torch.no_grad():
        artifact = torch.load(tmp_path / "sample.pt", weights_only=True)
    assert set(artifact) == {
        "sample_id", "source_id", "prompt_length", "task_type", "data_source",
        "values", "valid",
    }
    assert artifact["prompt_length"] == 1
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["labels_included"] is False
