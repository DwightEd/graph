from __future__ import annotations

import numpy as np
import pytest

from experiments.reanchor_flow.artifact_payload import save_native_audit
from experiments.reanchor_flow.artifact_validation import validate_native_audit
from experiments.reanchor_flow.flow import FlowSignal
from experiments.reanchor_flow.tests.test_subset_artifacts import _fixture, _metadata


def _saved_artifact(tmp_path):
    world, audit, target = _fixture()
    metadata = _metadata()
    path = tmp_path / "audit.npz"
    save_native_audit(path, world, audit, metadata)
    return path, world, target, metadata


def _arrays(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _validate(path, world, target, metadata) -> None:
    validate_native_audit(path, world, target, FlowSignal.MESSAGE, metadata)


@pytest.mark.parametrize(
    ("missing", "message"),
    (
        ("route_head_action", "route_head_action"),
        ("route_origin_name", "route_origin_name"),
        ("root_selection_score", "root_selection_score"),
        ("carrier_rescue", "carrier_rescue"),
        ("frozen_corridor_target", "frozen_corridor_target"),
    ),
)
def test_resume_rejects_truncated_schema_three_artifact(
    tmp_path, missing: str, message: str
) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays.pop(missing)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match=message):
        _validate(path, world, target, metadata)


def test_resume_rejects_misaligned_edge_columns(tmp_path) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays["edge_head"] = arrays["edge_head"][:-1]
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="edge_layer group"):
        _validate(path, world, target, metadata)


def test_resume_rejects_truncated_route_rows(tmp_path) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays["route_row_position"] = arrays["route_row_position"][1:]
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="route row coordinates"):
        _validate(path, world, target, metadata)


def test_resume_rejects_changed_world_unit_coordinates(tmp_path) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays["token_unit_id"] = arrays["token_unit_id"].copy()
    arrays["token_unit_id"][0] = 1
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="token_unit_id does not match"):
        _validate(path, world, target, metadata)


def test_resume_rejects_corridor_count_without_complete_plan(tmp_path) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays["corridor_edge_count"] = arrays["corridor_edge_count"] + 1
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="frozen_corridor_edge_index count"):
        _validate(path, world, target, metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selected_root_evaluated", False, "root evaluation state"),
        ("corridor_necessity", np.nan, "corridor exact values"),
        ("carrier_evaluated_count", 0, "carrier evaluation count"),
        ("full_chain_confirmed", False, "full-chain confirmation"),
    ),
)
def test_resume_rejects_inconsistent_confirmation_state(
    tmp_path, field: str, value: object, message: str
) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays[field] = np.asarray(value)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match=message):
        _validate(path, world, target, metadata)


def test_resume_rejects_selected_root_summary_drift(tmp_path) -> None:
    path, world, target, metadata = _saved_artifact(tmp_path)
    arrays = _arrays(path)
    arrays["selected_root_causal_score"] = np.asarray(123.0)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="selected-root summaries"):
        _validate(path, world, target, metadata)
