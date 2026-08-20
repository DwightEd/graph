from pathlib import Path

import numpy as np

from experiments.attention_phenomenology.head_effects import HeadLayerEffectMap


def test_effect_map_keeps_layer_and_head_cells_separate(tmp_path: Path):
    values = np.zeros((6, 2, 2, 1), dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
    values[:3, 1, 0, 0] = np.asarray([0.0, 0.5, 1.0])
    values[3:, 1, 0, 0] = np.asarray([2.0, 2.5, 3.0])
    values[:3, 0, 1, 0] = np.asarray([2.0, 2.5, 3.0])
    values[3:, 0, 1, 0] = np.asarray([0.0, 0.5, 1.0])

    effects = HeadLayerEffectMap().compute(
        [(values, labels)],
        feature_names=("route",),
    )

    assert effects.standardized_mean_difference.shape == (2, 2, 1)
    assert effects.standardized_mean_difference[1, 0, 0] > 0
    assert effects.standardized_mean_difference[0, 1, 0] < 0
    assert effects.standardized_mean_difference[0, 0, 0] == 0

    paths = effects.save(tmp_path, prefix="validation")

    assert paths["csv"].is_file()
    assert paths["figure"].is_file()
    assert "layer,head,feature" in paths["csv"].read_text(encoding="utf-8")
