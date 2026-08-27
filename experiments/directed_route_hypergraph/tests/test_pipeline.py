from dataclasses import asdict

import pytest
import torch

from experiments.directed_route_hypergraph.config import ModelConfig
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.directed_route_hypergraph.pipeline import (
    ARCHITECTURE_VERSION,
    METHOD,
    restore_model,
)


def test_restore_rejects_old_method_before_loading_incompatible_weights(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"method": "directed_route_hypergraph_ordered_layout"}, path)

    with pytest.raises(ValueError, match="different method"):
        restore_model(path, "cpu")


@pytest.mark.parametrize("version", [None, 1, ARCHITECTURE_VERSION + 1])
def test_restore_rejects_incompatible_version_before_loading_weights(
    tmp_path,
    version,
):
    path = tmp_path / f"version_{version}.pt"
    payload = {"method": METHOD, "state_dict": {"not": torch.ones(1)}}
    if version is not None:
        payload["architecture_version"] = version
    torch.save(payload, path)

    with pytest.raises(ValueError, match="incompatible architecture version"):
        restore_model(path, "cpu")


def test_current_endpoint_recovery_checkpoint_round_trips(tmp_path):
    config = ModelConfig(dropout=0.0)
    expected = DirectedRouteHypergraphEncoder(3, 2, config)
    path = tmp_path / "current.pt"
    torch.save(
        {
            "method": METHOD,
            "architecture_version": ARCHITECTURE_VERSION,
            "layer_count": 3,
            "head_count": 2,
            "model_config": asdict(config),
            "state_dict": expected.state_dict(),
        },
        path,
    )

    checkpoint, restored = restore_model(path, "cpu")

    assert checkpoint["method"] == METHOD
    assert checkpoint["architecture_version"] == ARCHITECTURE_VERSION
    assert restored.layer_count == 3
    assert restored.head_count == 2
    assert restored.config == config
    for name, value in expected.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
