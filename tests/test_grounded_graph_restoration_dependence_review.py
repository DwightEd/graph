"""CPU integrity checks for fixed-H_empty source-payload dependence."""

import hashlib
import json

import numpy as np
import pytest

from next_iteration.grounded_graph_restoration_dependence import (
    aggregate,
    erased_arrays,
    validate_branch,
)


def _array_receipt(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _packet():
    return {
        "sha256": "packet-1",
        "input_ids": [1, 10, 11, 12, 20],
        "weak_targets": [{"owner_node_index": 0}],
        "nodes": [
            {"prompt_token_indices": [1, 2]},
            {"prompt_token_indices": [3]},
        ],
    }


def _write_erasure(tmp_path, *, owners=(0,)):
    path = tmp_path / "erasure"
    (path / "sources").mkdir(parents=True)
    packet = _packet()
    source = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)
    cached_query = np.array([[5.0, 4.0]], dtype=np.float32)
    receipt = {
        "packet_sha256": packet["sha256"],
        "replacement_token_id": 0,
        "source_owner_indices": list(owners),
        "erased_prompt_token_indices": [1, 2],
        "executed_input_ids": [1, 0, 0, 12, 20],
        "actual_observer_forwards": 1,
        "arrays": {"source": _array_receipt(source), "query": _array_receipt(cached_query)},
    }
    record = {
        "status": "complete",
        "source_id": "s1",
        "task": "Data2txt",
        "original_receipt_sha256": "original-receipt",
        "erasure_receipt": receipt,
    }
    (path / "sources" / "s1.json").write_text(json.dumps(record))
    np.savez(path / "sources" / "s1.npz", source=source, query=cached_query)
    entry = {"source_id": "s1", "task": "Data2txt"}
    arrays = {"source": np.zeros_like(source), "query": np.array([[1.0, 2.0]], dtype=np.float32)}
    restoration_receipt = {"original_receipt_sha256": "original-receipt", "replacement_token_id": 0}
    return path, entry, packet, arrays, restoration_receipt


def test_cached_payload_erasure_reuses_its_source_but_keeps_restoration_h_empty_fixed(tmp_path):
    path, entry, packet, arrays, restoration_receipt = _write_erasure(tmp_path)

    changed = erased_arrays(path, entry, packet, arrays, restoration_receipt)

    np.testing.assert_array_equal(changed["source"], [[9.0, 8.0], [7.0, 6.0]])
    np.testing.assert_array_equal(changed["query"], arrays["query"])
    assert changed["query"] is arrays["query"]

    # A plausible-looking cached result for a different selected owner must not
    # be reused as a graph-payload intervention.
    path, entry, packet, arrays, restoration_receipt = _write_erasure(tmp_path / "wrong-owner", owners=(1,))
    with pytest.raises(ValueError, match="packet/node/anchor order"):
        erased_arrays(path, entry, packet, arrays, restoration_receipt)


def _branch(gain, pointer):
    return {
        "adapter_logp_gain": gain,
        "coordinate_pointer_probability": pointer,
    }


def test_source_mean_gate_requires_both_positive_empty_query_gain_and_payload_drop():
    records = [{
        "branches": {
            "original_source_empty_query": {
                "graph": _branch([0.4, 0.2], [0.8, 0.6]),
                "no_edges": _branch([-0.1, -0.1], [0.2, 0.2]),
            },
            "payload_erased_source_empty_query": {
                "graph": _branch([0.1, 0.1], [0.3, 0.2]),
                "no_edges": _branch([-0.3, -0.3], [0.1, 0.1]),
            },
        },
    }]

    result = aggregate(records)

    assert result["graph"]["source_mean_anchor_gain"] == pytest.approx(0.3)
    assert result["graph"]["source_mean_anchor_gain_drop"] == pytest.approx(0.2)
    assert result["graph"]["source_mean_coordinate_probability_drop"] == pytest.approx(0.45)
    assert result["graph"]["mechanism_gate_passed"] is True
    # A payload-dependent degradation below the empty baseline is not a gain.
    assert result["no_edges"]["source_mean_anchor_gain_drop"] == pytest.approx(0.2)
    assert result["no_edges"]["mechanism_gate_passed"] is False


def test_branch_validation_rejects_nonfinite_measurement_and_invalid_pointer_probability():
    branch = {
        "query_indices": [4],
        "base_logp": [-1.0],
        "graph": {"adapter_logp": [-0.8], "adapter_logp_gain": [0.2], "coordinate_pointer_probability": [0.7]},
        "no_edges": {"adapter_logp": [-0.9], "adapter_logp_gain": [0.1], "coordinate_pointer_probability": [0.2]},
    }
    validate_branch(branch)

    nonfinite = {**branch, "graph": {**branch["graph"], "adapter_logp_gain": [float("nan")]}}
    with pytest.raises(ValueError, match="nonfinite"):
        validate_branch(nonfinite)
    invalid_probability = {**branch, "no_edges": {**branch["no_edges"], "coordinate_pointer_probability": [1.01]}}
    with pytest.raises(ValueError, match="zero to one"):
        validate_branch(invalid_probability)
