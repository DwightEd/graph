import torch

from experiments.attention_phenomenology.causal_head_model import (
    CausalLayerTemporalDetector,
    CausalLayerTemporalModel,
    HeadSequence,
    TrainingConfig,
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


def _learnable_sequence(sample: int) -> HeadSequence:
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1], dtype=torch.float32)
    values = torch.zeros(8, 1, 2, 1)
    values[:, 0, 0, 0] = labels * 2.0 - 1.0 + sample * 0.01
    values[:, 0, 1, 0] = -values[:, 0, 0, 0]
    return HeadSequence(
        sample_id=str(sample),
        source_id=f"source-{sample}",
        task_type="QA",
        values=values,
        labels=labels,
    )


def test_detector_fits_on_train_and_selects_on_validation():
    train = [_learnable_sequence(index) for index in range(6)]
    validation = [_learnable_sequence(index) for index in range(6, 8)]
    detector = CausalLayerTemporalDetector(
        TrainingConfig(
            hidden_dim=8,
            epochs=30,
            batch_size=2,
            learning_rate=0.02,
            patience=10,
            seed=3,
        )
    )

    history = detector.fit(train, validation)
    evaluation = detector.evaluate(validation)

    assert history
    assert detector.best_epoch >= 1
    assert evaluation["current"]["auroc"] > 0.9
    assert evaluation["forecast_1"]["tokens"] == 14
