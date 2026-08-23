import numpy as np

from experiments.non_neural_structure_audit.artifacts import (
    npz_shapes,
    save_npz,
    write_csv,
)


def test_empty_csv_result_clears_a_previous_result(tmp_path):
    output = tmp_path / "result.csv"
    output.write_text("stale,result\n", encoding="utf-8")

    write_csv(output, [])

    assert output.read_text(encoding="utf-8-sig") == ""


def test_npz_shapes_reads_headers_without_loading_array_values(tmp_path):
    output = tmp_path / "arrays.npz"
    save_npz(output, scores=np.zeros((2, 3, 4), dtype=np.float32))

    assert npz_shapes(output, ["scores"]) == {"scores": (2, 3, 4)}
