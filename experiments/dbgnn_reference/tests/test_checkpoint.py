import numpy as np
import torch

from experiments.dbgnn_reference.checkpoint import make_portable


def test_make_portable_converts_numpy_scalar_metadata(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(2)},
            "best_validation_loss": np.float64(0.5),
            "history": [{"positive_pairs": np.int64(7)}],
        },
        path,
    )

    make_portable(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert checkpoint["best_validation_loss"] == 0.5
    assert checkpoint["history"][0]["positive_pairs"] == 7
    assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(2))
