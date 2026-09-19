import torch

from experiments.unsupervised_token_graph.structured_compatibility.model import (
    CLASS_NAMES,
    StructuredCompatibility,
    corruption_batch,
)


def test_corruptions_preserve_shapes_and_balanced_classes():
    current = torch.randn(3, 2, 4, 5)
    previous = torch.randn(3, 2, 4, 5)
    source_donor = torch.randn(3, 2, 4, 5)
    previous_donor = torch.randn(3, 2, 4, 5)

    center = torch.zeros(2, 4, 5)
    scale = torch.ones(2, 4, 5)
    corrupted_current, corrupted_previous, labels = corruption_batch(
        current,
        previous,
        source_donor,
        previous_donor,
        center,
        scale,
    )
    assert corrupted_current.shape == (15, 2, 4, 5)
    assert corrupted_previous.shape == (15, 2, 4, 5)
    assert labels.tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3


def test_model_outputs_one_logit_per_corruption_class():
    model = StructuredCompatibility(2, 4, 5, hidden=3)
    logits = model(
        torch.randn(7, 2, 4, 5),
        torch.randn(7, 2, 4, 5),
    )
    assert logits.shape == (7, len(CLASS_NAMES))


def test_source_corruptions_preserve_each_head_prompt_mass():
    current = torch.rand(2, 3, 4, 5)
    previous = torch.rand(2, 3, 4, 5)
    source_donor = torch.rand(2, 3, 4, 5)
    previous_donor = torch.rand(2, 3, 4, 5)

    center = torch.zeros(3, 4, 5)
    scale = torch.ones(3, 4, 5)
    corrupted, _, _ = corruption_batch(
        current,
        previous,
        source_donor,
        previous_donor,
        center,
        scale,
    )
    batch = len(current)
    original_prompt = current[..., 0] + current[..., 1]
    source_head = corrupted[2 * batch:3 * batch]
    source_time = corrupted[3 * batch:4 * batch]

    torch.testing.assert_close(
        source_head[..., 0] + source_head[..., 1],
        original_prompt,
    )
    torch.testing.assert_close(
        source_time[..., 0] + source_time[..., 1],
        original_prompt,
    )
