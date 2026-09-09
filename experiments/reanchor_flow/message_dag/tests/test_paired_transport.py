import numpy as np
import pytest

from experiments.reanchor_flow.message_dag.paired_transport import PairedRouteScreen


def save_cut(path, chunks, signed):
    np.savez_compressed(
        path,
        cut_schema=np.array(1),
        branch_names=np.array(["V_content", "K_routing"]),
        row_position=np.array([4, 5, 6]),
        event_row=np.array(0),
        cut_signed=signed,
        cut_closure_error=np.zeros(2),
        **chunks,
    )


def test_paired_screen_closes_matched_and_unmatched_edges_without_imputation(tmp_path):
    plus_signed = np.zeros((2, 1, 2, 2), np.float32)
    minus_signed = np.zeros_like(plus_signed)
    plus_signed[0, 0, 1] = [6, 2]
    minus_signed[0, 0, 1] = [2, 4]
    plus_signed[1, 0, 1] = [3, 0]
    save_cut(
        tmp_path / "plus.npz",
        {
            "L0Q1": np.array([[[[6.0, 2.0]]]]),
            "L1Q1": np.array([[[[3.0, 0.0]]]]),
        },
        plus_signed,
    )
    save_cut(
        tmp_path / "minus.npz",
        {"L0Q1": np.array([[[[2.0, 4.0]]]])},
        minus_signed,
    )

    screen = PairedRouteScreen().compare(tmp_path / "plus.npz", tmp_path / "minus.npz")

    np.testing.assert_allclose(screen["selective_signed"][0, 0, 1], [2, -1])
    np.testing.assert_allclose(screen["common_signed"][0, 0, 1], [4, 3])
    np.testing.assert_allclose(screen["unmatched_plus_signed"][1, 0, 1], [3, 0])
    assert screen["matched_edges"] == 2
    assert screen["unmatched_plus_edges"] == 2
    assert screen["unmatched_minus_edges"] == 0
    assert screen["matched_fraction"] == pytest.approx(7 / 8.5)
    np.testing.assert_array_equal(screen["carrier_index"], [0, 0, 1])
    np.testing.assert_array_equal(screen["top_edge_index"], [0, 0, 1, 0, 0])
    assert screen["status"] == "eligible"
    assert screen["labels_used"] is False
    assert screen["world_partition_error"] == pytest.approx(0)
    assert screen["paired_closure_error"] == pytest.approx(0)


def test_paired_screen_distinguishes_no_response_from_invalid_closure(tmp_path):
    signed = np.zeros((1, 1, 2, 2), np.float32)
    chunks = {"L0Q1": np.zeros((1, 1, 1, 2), np.float32)}
    save_cut(tmp_path / "plus.npz", chunks, signed)
    save_cut(tmp_path / "minus.npz", chunks, signed)

    screen = PairedRouteScreen().compare(tmp_path / "plus.npz", tmp_path / "minus.npz")
    assert screen["status"] == "no_response"
    assert screen["carrier_index"].size == 0

    broken = signed.copy()
    broken[0, 0, 1, 0] = 1
    save_cut(tmp_path / "broken.npz", chunks, broken)
    with pytest.raises(ValueError, match="partition"):
        PairedRouteScreen().compare(tmp_path / "broken.npz", tmp_path / "minus.npz")


def test_paired_screen_rejects_unclosed_or_position_misaligned_worlds(tmp_path):
    signed = np.zeros((1, 1, 2, 2), np.float32)
    chunks = {"L0Q1": np.zeros((1, 1, 1, 2), np.float32)}
    save_cut(tmp_path / "plus.npz", chunks, signed)
    save_cut(tmp_path / "minus.npz", chunks, signed)

    with np.load(tmp_path / "minus.npz", allow_pickle=False) as stored:
        values = dict(stored)
    values["row_position"] = np.array([4, 6, 7])
    np.savez_compressed(tmp_path / "minus.npz", **values)
    with pytest.raises(ValueError, match="positions"):
        PairedRouteScreen().compare(tmp_path / "plus.npz", tmp_path / "minus.npz")

    values["row_position"] = np.array([4, 5, 6])
    values["cut_closure_error"] = np.array([0.0, 1.0])
    np.savez_compressed(tmp_path / "minus.npz", **values)
    with pytest.raises(ValueError, match="closed"):
        PairedRouteScreen().compare(tmp_path / "plus.npz", tmp_path / "minus.npz")
