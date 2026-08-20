import torch

from experiments.attention_phenomenology.causal_head_model import (
    CausalLayerTemporalModel,
)


def test_model_emits_current_and_next_token_logits():
    model = CausalLayerTemporalModel(
        num_layers=3,
        num_heads=2,
        num_features=4,
        hidden_dim=8,
    )
    values = torch.randn(2, 5, 3, 2, 4)

    output = model(values)

    assert output.current_logits.shape == (2, 5)
    assert output.next_logits.shape == (2, 5)


def test_temporal_model_is_prefix_causal():
    torch.manual_seed(7)
    model = CausalLayerTemporalModel(
        num_layers=2,
        num_heads=2,
        num_features=3,
        hidden_dim=6,
    ).eval()
    values = torch.randn(1, 6, 2, 2, 3)

    full = model(values)
    prefix = model(values[:, :4])

    torch.testing.assert_close(full.current_logits[:, :4], prefix.current_logits)
    torch.testing.assert_close(full.next_logits[:, :4], prefix.next_logits)


def test_permuting_heads_changes_the_model_input_semantics():
    torch.manual_seed(11)
    model = CausalLayerTemporalModel(
        num_layers=2,
        num_heads=2,
        num_features=2,
        hidden_dim=5,
    ).eval()
    values = torch.zeros(1, 3, 2, 2, 2)
    values[..., 0, 0] = 1.0
    swapped = values.flip(dims=(3,))

    original = model(values).current_logits
    permuted = model(swapped).current_logits

    assert not torch.allclose(original, permuted)
