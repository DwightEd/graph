"""Scientific invariants: no label fitting, source isolation and causal prefix stability."""

import copy
import json
from zipfile import ZipFile

import numpy as np
import pytest

from experiments.native_support.dual_state.data import head_routes, validate_reference
from experiments.native_support.dual_state.report import intervals
from experiments.native_support.dual_state.run import main
from experiments.native_support.dual_state.scoring import (
    calibrate, fit_reference, head_scores, reference_records, window_mean,
)
from experiments.native_support.readout.data import BASELINES
from state_audit.storage import write_arrays, write_json


def observation(identity, source, values):
    values = np.asarray(values, dtype=float)
    return dict(id=identity, source_id=source, head_route=values,
                response_energy=np.ones_like(values),
                target=np.arange(len(values)), response_length=len(values))


def test_current_survives_and_causal_prefix_does_not_read_future():
    reference = [observation("a", "a", [[0, 1]] * 25), observation("b", "b", [[1, 0]] * 25)]
    fitted = fit_reference(reference)
    query = observation("q", "q", [[.2, .3]] * 25)
    query["head_route"][12] = [2, 2]
    scores, state = head_scores(query, fitted, 6)
    changed = copy.deepcopy(query)
    changed["head_route"][13:] = -3
    later, _ = head_scores(changed, fitted, 6)
    prefix = observation("q", "q", query["head_route"][:13])
    truncated, _ = head_scores(prefix, fitted, 6)
    for name in ("head_current", "head_causal_persistent", "head_causal_dual"):
        np.testing.assert_allclose(scores[name][:13], later[name][:13])
        np.testing.assert_allclose(scores[name][:13], truncated[name])
    assert not np.allclose(scores["head_offline_persistent"][:13], later["head_offline_persistent"][:13])
    assert scores["head_causal_dual"][12] == scores["head_current"][12] == 1
    assert np.all(scores["head_causal_dual"] >= scores["head_current"])
    np.testing.assert_array_equal(state["current"], calibrate(query, fitted))


def test_temporal_accumulation_precedes_positive_head_pooling():
    reference = [observation("a", "a", [[0, 0]] * 4), observation("b", "b", [[1, 1]] * 4)]
    query = observation("q", "q", [[2, -1], [-1, 2], [2, -1], [-1, 2]])
    scores, _ = head_scores(query, fit_reference(reference), 2)
    np.testing.assert_allclose(scores["head_current"], np.sqrt(.5))
    # Alternating heads have no persistent per-head positive drift, despite constant pooled current.
    np.testing.assert_allclose(scores["head_causal_persistent"][1:], 0)
    assert window_mean(scores["head_current"], 2)[1] > 0


def test_zero_response_has_no_instantaneous_directional_risk():
    reference = [observation("a", "a", [[-1]] * 4), observation("b", "b", [[-.5]] * 4)]
    query = observation("q", "q", [[0]] * 4)
    query["response_energy"][:] = 0
    np.testing.assert_array_equal(calibrate(query, fit_reference(reference)), 0)


def test_reference_source_weights_ties_and_whole_source_exclusion():
    records = [observation("a1", "a", [[0]] * 4), observation("a2", "a", [[0]] * 4),
               observation("b", "b", [[1]] * 4), observation("c", "c", [[2]] * 4)]
    selected = reference_records(records, "a")
    assert {r["id"] for r in selected} == {"b", "c"}
    query = observation("q", "q", [[.5], [.5], [.5], [.5]])
    first = calibrate(query, fit_reference(records[:3]))
    second = calibrate(query, fit_reference([records[0], records[2]]))
    np.testing.assert_allclose(first, 0)
    np.testing.assert_allclose(first, second)
    tied = observation("q", "q", [[0]] * 4)
    np.testing.assert_allclose(calibrate(tied, fit_reference(records[:3])), -.5)
    with pytest.raises(ValueError, match="two source-disjoint"):
        reference_records(records[:3], "a")


def test_role_energy_and_source_block_permutation():
    response = np.zeros((4, 2, 6, 3))
    response[..., 0, 0] = 1
    response[..., 1, 0] = 2
    response[..., 2, 1] = 4  # history after two source blocks
    response[..., 5, 2] = 5  # self
    row = dict(response_total=response, read_mass=np.zeros((4, 2, 6)))
    route, energy, names = head_routes(iter([row]), 2, "middle")
    np.testing.assert_allclose(route, .5)
    np.testing.assert_allclose(energy, 12)
    assert names == ["L1/H0", "L1/H1", "L2/H0", "L2/H1"]
    swapped = {key: value[:, :, [1, 0, 2, 3, 4, 5]] for key, value in row.items()}
    changed, _, _ = head_routes(iter([swapped]), 2, "middle")
    np.testing.assert_array_equal(route, changed)


def create_capture(path):
    rng = np.random.default_rng(11)
    responses, annotations = [], {}
    for index in range(3):
        count, prompt = 20, 3
        response = dict(id=str(index), source_id=f"s{index}", prompt_length=prompt,
                        token_ids=list(range(count+prompt)), token_text=[" x"]*(count+prompt))
        responses.append(response)
        directory = path / "responses" / f"{index:04d}"
        scores = {name: rng.random(count) for name in BASELINES}
        tokens = np.asarray(response["token_ids"][prompt:])
        write_arrays(directory / "scores.npz", target=np.arange(count), token_id=tokens, **scores)
        write_json(directory / "sources.json", dict(blocks=[{}, {}]))
        annotations[str(index)] = dict(source_id=f"s{index}", token_ids=tokens.tolist(), labels=[0]*6+[1]*5+[0]*9)
        for target in range(count):
            residual = rng.normal(size=(4, 2, 6, 3))
            ffn = rng.normal(size=residual.shape)
            read = rng.random((4, 2, 6))
            read /= read.sum(-1, keepdims=True)
            write_arrays(directory / f"token_{target:06d}.npz", target=target, query=prompt+target-1,
                         token_id=tokens[target], response_residual=residual, response_ffn=ffn,
                         response_total=residual+ffn, read_mass=read)
    write_json(path / "settings.json", dict(model="synthetic", responses=responses))
    write_json(path / "protocol.json", dict(rank=3, seed=11, dtype="float64", roles=[], channels=[]))
    return annotations


def test_pipeline_scores_before_labels_and_labels_cannot_change_scores(tmp_path, monkeypatch):
    from experiments.native_support.choice_cache import CaptureReader
    source = tmp_path / "capture"
    annotations = create_capture(source)
    first, second = tmp_path / "first", tmp_path / "second"
    # Scoring is usable without annotation files at all.
    main(["--input", str(source), "--output", str(first)])
    write_json(source / "annotations.json", annotations)
    original = CaptureReader.json

    def check_boundary(reader, name):
        if name == "annotations.json":
            assert all((second / "responses" / f"{index:04d}" / "scores.npz").is_file() for index in range(3))
        return original(reader, name)

    monkeypatch.setattr(CaptureReader, "json", check_boundary)
    main(["--input", str(source), "--output", str(second)])
    for index in range(3):
        relative = f"responses/{index:04d}/scores.npz"
        with np.load(first / relative) as before, np.load(second / relative) as after:
            assert before.files == after.files
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key])
    evaluation = json.loads((second / "evaluation.json").read_text())
    assert evaluation["methods"]["head_causal_dual"]["all_error"]["tokens"] == 60
    with ZipFile(second.with_name("second_review_light.zip")) as archive:
        assert "span_budget.csv" in archive.namelist()
        assert "normal_run_budget.csv" in archive.namelist()
        assert not any(name.endswith("head_state.npz") or name.endswith("reference.npz") for name in archive.namelist())


def test_annotation_runs_preserve_adjacent_onsets_and_missingness():
    record = dict(labels=np.array([0, 1, 1, 1, 0, 0]), valid=np.array([1, 1, 1, 1, 0, 1], dtype=bool),
                  onsets=np.array([0, 1, 0, 1, 0, 0], dtype=bool))
    assert intervals(record, True) == [(1, 3), (3, 4)]
    assert intervals(record, False) == [(0, 1), (5, 6)]


def test_external_reference_rejects_overlap_and_measurement_mismatch():
    dataset = dict(records=[dict(source_id="a")], settings=dict(model="model"), schema=["L0/H0"],
                   protocol=dict(rank=8, seed=37, dtype="bfloat16", roles=[], channels=[]))
    with pytest.raises(ValueError, match="disjoint"):
        validate_reference(dataset, dataset)
    reference = copy.deepcopy(dataset)
    reference["records"] = [dict(source_id="b")]
    reference["protocol"]["rank"] = 16
    with pytest.raises(ValueError, match="rank"):
        validate_reference(dataset, reference)
