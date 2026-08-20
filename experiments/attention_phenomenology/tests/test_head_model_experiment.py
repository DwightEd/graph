from types import SimpleNamespace

import torch

from experiments.attention_phenomenology.causal_head_model import HeadSequence
from experiments.attention_phenomenology.head_model_experiment import (
    mask_response_reuse,
    source_disjoint_train_validation_split,
    stratified_sample_ids,
)


def _sequence(sample: str, source: str, task: str, positive: bool) -> HeadSequence:
    labels = torch.tensor([0.0, float(positive)])
    return HeadSequence(
        sample_id=sample,
        source_id=source,
        task_type=task,
        values=torch.zeros(2, 1, 1, 1),
        labels=labels,
    )


def test_train_validation_split_never_separates_a_source_group():
    sequences = [
        _sequence("a1", "a", "QA", False),
        _sequence("a2", "a", "QA", True),
        _sequence("b", "b", "QA", False),
        _sequence("c", "c", "QA", True),
        _sequence("d", "d", "Summary", False),
        _sequence("e", "e", "Summary", True),
        _sequence("f", "f", "Summary", False),
        _sequence("g", "g", "Summary", True),
    ]

    train, validation = source_disjoint_train_validation_split(
        sequences,
        validation_fraction=0.25,
        seed=7,
    )

    train_sources = {sequence.source_id for sequence in train}
    validation_sources = {sequence.source_id for sequence in validation}
    assert train_sources.isdisjoint(validation_sources)
    assert {sequence.source_id for sequence in sequences} == (
        train_sources | validation_sources
    )
    assert {sequence.sample_id for sequence in train if sequence.source_id == "a"} in (
        {"a1", "a2"},
        set(),
    )
    assert {int(sequence.labels.sum().item() > 0) for sequence in validation} == {0, 1}


def test_limited_experiment_samples_each_task_instead_of_index_prefix():
    tasks = {
        **{f"d{index}": "Data2txt" for index in range(8)},
        **{f"q{index}": "QA" for index in range(8)},
        **{f"s{index}": "Summary" for index in range(8)},
    }

    class Dataset:
        def __init__(self):
            self.sample_ids = list(tasks)

        def __getitem__(self, sample_id):
            return SimpleNamespace(task_type=tasks[sample_id])

    selected = stratified_sample_ids(Dataset(), limit=9, seed=5)
    selected_tasks = [tasks[sample_id] for sample_id in selected]

    assert selected == stratified_sample_ids(Dataset(), limit=9, seed=5)
    assert selected_tasks.count("Data2txt") == 3
    assert selected_tasks.count("QA") == 3
    assert selected_tasks.count("Summary") == 3


def test_reuse_control_keeps_geometry_and_zeros_only_reuse_coordinates():
    values = torch.arange(2 * 1 * 2 * 6, dtype=torch.float32).reshape(2, 1, 2, 6)

    masked = mask_response_reuse(values, reuse_top_k=2)

    torch.testing.assert_close(masked[..., :4], values[..., :4])
    torch.testing.assert_close(masked[..., 4:], torch.zeros_like(values[..., 4:]))
    assert masked.shape == values.shape
    assert not torch.equal(masked, values)
