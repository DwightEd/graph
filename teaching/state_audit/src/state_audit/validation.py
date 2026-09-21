"""Optional scientific reconstruction checks, separate from capture and analysis."""

import numpy as np

from .analysis.roles import attention_from_keys
from .state import ModelState
from .storage import read_arrays


def check_trace(root, sample):
    state = ModelState.open(root / "samples" / f"{sample:06d}" / "trace")
    checks = {}
    for layer in state.iter_layers():
        trace = dict(queries=layer.positions, **layer.tensors)
        rebuilt = attention_from_keys(trace, trace["key"])
        np.testing.assert_allclose(
            rebuilt, trace["attention"], atol=2e-3, rtol=2e-2, equal_nan=False
        )
        heads = len(trace["attention"])
        values = np.repeat(trace["value"], heads // len(trace["value"]), axis=0)
        head_output = (trace["attention"] @ values).transpose(1, 0, 2)
        np.testing.assert_allclose(
            head_output, trace["head_readout"], atol=2e-3, rtol=2e-2, equal_nan=False
        )
        weights = read_arrays(root / "weights" / f"layer_{layer.layer:03d}.npz")
        flattened = head_output.reshape(len(layer.positions), -1)
        projected = flattened @ weights["output_projection"].T + weights["bias"]
        np.testing.assert_allclose(
            projected, trace["attention_write"], atol=2e-3, rtol=2e-2, equal_nan=False
        )
        checks[layer.layer] = dict(
            attention_max_error=float(np.max(abs(rebuilt - trace["attention"]))),
            projection_max_error=float(np.max(abs(projected - trace["attention_write"]))),
        )
    return checks
