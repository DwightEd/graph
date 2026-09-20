import json

import pytest

from state_audit.datasets import convert_ragtruth, load_examples
from state_audit.tokenization import generated_offsets


@pytest.mark.parametrize(
    "task,info,prompt,count",
    [
        (
            "QA",
            dict(question="Color?", passages="passage 1:Blue.\n\n passage 2:Red."),
            "Color? passage 1:Blue.\npassage 2:Red.",
            2,
        ),
        ("Summary", "Mira wears blue.", "Summarize: Mira wears blue.", 1),
        (
            "Data2txt",
            dict(name="Mira", color="blue"),
            "Describe: {'name': 'Mira', 'color': 'blue'}",
            1,
        ),
    ],
)
def test_ragtruth_adapter_preserves_prompt_labels_and_metadata(tmp_path, task, info, prompt, count):
    source = dict(source_id="s", task_type=task, source_info=info, prompt=prompt)
    response = dict(
        id="1",
        source_id="s",
        response="Red.",
        split="test",
        model="original",
        labels=[dict(start=0, end=3, text="Red", label_type="conflict")],
    )
    (tmp_path / "source_info.jsonl").write_text(json.dumps(source) + "\n")
    (tmp_path / "response.jsonl").write_text(json.dumps(response) + "\n")
    output = tmp_path / "normalized.jsonl"
    convert_ragtruth(tmp_path, output, "test", None)
    sample = load_examples(output)[0]
    assert sample.prompt == prompt and len(sample.evidence) == count
    assert sample.labels == response["labels"]
    assert sample.metadata["model"] == "original"


def test_generated_alignment_failure_does_not_guess_offsets():
    class Tokenizer:
        all_special_ids = [0]

        def __call__(self, text, **kwargs):
            return dict(input_ids=[3], offset_mapping=[(0, 1)])

    assert generated_offsets(Tokenizer(), [7], "x") is None
    assert generated_offsets(Tokenizer(), [3, 0], "x") == [[0, 1], [0, 0]]
