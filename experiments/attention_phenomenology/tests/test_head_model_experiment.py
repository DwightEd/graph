import torch

from experiments.attention_phenomenology.causal_head_model import HeadSequence
from experiments.attention_phenomenology.head_model_experiment import (
    source_disjoint_train_validation_split,
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
