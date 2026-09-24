"""Holdout integrity, feature identities and native-cache-only readout."""

import copy
import json
import numpy as np
import pytest

from experiments.native_support.readout.features import head_features, temporal_features
from experiments.native_support.readout.models import fit_predict, source_folds, validate_reference
from experiments.native_support.readout.run import main
from state_audit.storage import write_arrays, write_csv, write_json


def record(identity, source, values, labels):
    labels = np.asarray(labels)
    return dict(id=identity, source_id=source, labels=labels,
                valid=np.ones(len(labels), dtype=bool), views={"instant": np.asarray(values, dtype=float)})


def test_temporal_context_respects_endpoints_and_preserves_missingness():
    values = np.array([[1., np.nan], [3., 4.], [8., 6.]])
    result, names = temporal_features(values, ["a", "b"], 2)
    assert np.isnan(result[0, names.index("past/a")])
    assert result[0, names.index("future/a")] == 5.5
    assert result[2, names.index("past/b")] == 4
    assert result[2, names.index("innovation/a")] == 6
    assert np.isnan(result[-1, names.index("future/a")])


def test_source_holdout_includes_every_answer_of_the_source():
    records = [record("a1", "a", [[1], [2]], [0, 1]), record("a2", "a", [[3], [4]], [1, 0]),
               record("b1", "b", [[5], [6]], [0, 1])]
    for _, training, targets in source_folds(records):
        assert not ({r["source_id"] for r in training} & {records[i]["source_id"] for i in targets})
    assert source_folds(records)[0][2] == [0, 1]


def test_heldout_labels_cannot_change_predictions_or_scaler(tmp_path):
    training = [record("a", "a", [[-2], [-1], [1], [2]], [0, 0, 1, 1])]
    heldout = [record("b", "b", [[100], [200]], [0, 1])]
    first, _ = fit_predict(training, heldout, [0], "instant", "logistic", tmp_path / "first")
    heldout[0]["labels"] = 1 - heldout[0]["labels"]
    second, _ = fit_predict(training, heldout, [0], "instant", "logistic", tmp_path / "second")
    np.testing.assert_array_equal(first[0], second[0])
    import joblib
    model = joblib.load(tmp_path / "first" / "instant_logistic.joblib")
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, [0])


def test_one_class_training_is_unavailable_not_fabricated(tmp_path):
    training = [record("a", "a", [[0], [1]], [0, 0])]
    predictions, detail = fit_predict(training, training, [0], "instant", "logistic", tmp_path)
    assert predictions is None
    assert detail["status"] == "unavailable_training_requires_both_classes"


def test_external_reference_cannot_overlap_sources():
    dataset = dict(records=[dict(source_id="a")])
    with pytest.raises(ValueError, match="disjoint"):
        validate_reference(dataset, dataset)


def test_source_block_permutation_preserves_features_but_head_identity_remains():
    generator = np.random.default_rng(37)
    rows = []
    for _ in range(4):
        residual = generator.normal(size=(2, 2, 7, 3))
        ffn = generator.normal(size=residual.shape)
        attention = generator.random((2, 2, 7))
        attention /= attention.sum(-1, keepdims=True)
        rows.append(dict(response_residual=residual, response_ffn=ffn,
                         response_total=residual + ffn, read_mass=attention))
    original, columns = head_features(rows, 3, [0, 1])
    permuted = copy.deepcopy(rows)
    for row in permuted:
        for name, value in row.items():
            row[name] = value[:, :, [2, 0, 1, 3, 4, 5, 6]]
    reordered, names = head_features(permuted, 3, [0, 1])
    assert columns == names
    np.testing.assert_allclose(original, reordered, atol=1e-6)
    swapped = [{name: value[:, ::-1] for name, value in row.items()} for row in rows]
    changed, _ = head_features(swapped, 3, [0, 1])
    assert not np.allclose(original, changed)
    assert np.all(original[0, [i for i, name in enumerate(columns) if name.startswith("memory_mass/")]] == 0)


def test_full_cache_to_heldout_metrics_and_pack_without_model(tmp_path):
    from experiments.native_support.readout.features import SCALARS
    from experiments.native_support.readout.data import CHOICE_FIELDS, attach_choice_features, load_dataset
    from zipfile import ZipFile
    generator = np.random.default_rng(11)
    source, destination, choice = tmp_path / "input", tmp_path / "output", tmp_path / "choice"
    responses, annotations, operators = [], {}, []
    for index in range(3):
        count, prompt = 12, 3
        response = dict(id=str(index), source_id=f"s{index}", prompt_length=prompt,
                        token_ids=list(range(count + prompt)), token_text=[" x"] * (count + prompt))
        responses.append(response)
        directory = source / "responses" / f"{index:04d}"
        scores = {name: generator.random(count) for name in SCALARS}
        scores["route_offline_mean"] = scores["raw_route"]
        tokens = np.asarray(response["token_ids"][prompt:])
        write_arrays(directory / "scores.npz", target=np.arange(count), token_id=tokens, **scores)
        write_json(directory / "sources.json", dict(blocks=[{}, {}]))
        annotations[str(index)] = dict(source_id=f"s{index}", token_ids=tokens.tolist(), labels=[0, 1] * 6)
        for target in range(count):
            residual = generator.normal(size=(2, 2, 6, 3))
            ffn = generator.normal(size=residual.shape)
            read = generator.random((2, 2, 6))
            read /= read.sum(-1, keepdims=True)
            write_arrays(directory / f"token_{target:06d}.npz", target=target, query=prompt + target - 1,
                         token_id=tokens[target], response_residual=residual, response_ffn=ffn,
                         response_total=residual + ffn, read_mass=read)
            for layer in range(2):
                operators.append(dict(response_id=str(index), target=target, layer=layer,
                    sketch_effective_rank=2, pullback_energy=1, skip_energy=1,
                    ffn_pullback_energy=.5, pullback_alignment=.75))
        write_arrays(directory / "state.npz", matrix=generator.normal(size=(count, 2, 16, 3)),
                     reading=generator.random((count, 2, 8)), response_scale=np.ones((count, 2)))
        write_arrays(choice / "responses" / f"{index:04d}" / "scores.npz",
                     target=np.arange(count), token_id=tokens, **{name: generator.random(count) for name in CHOICE_FIELDS})
    settings = dict(model="synthetic", responses=responses)
    write_json(source / "settings.json", settings)
    write_json(choice / "settings.json", settings)
    write_json(source / "annotations.json", annotations)
    write_json(source / "protocol.json", dict(model="synthetic", rank=3, seed=37, dtype="float32", roles=[], channels=[]))
    write_csv(source / "operators.csv", operators, list(operators[0]))
    main(["--input", str(source), "--output", str(destination), "--features", "heads",
          "--models", "logistic", "--cpu-threads", "1"])
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["labels_used_for_training"] and not summary["model_forward"]
    assert summary["evaluation"]["methods"]["native_state_logistic"]["all_error"]["tokens"] == 36
    assert summary["feature_dimensions"]["native_state"] == 134
    with ZipFile(destination.with_name("output_review_light.zip")) as archive:
        assert "folds.json" in archive.namelist()
        assert not any(name.endswith(".joblib") for name in archive.namelist())
    dataset = load_dataset(source)
    attach_choice_features(dataset, choice, 16)
    assert dataset["records"][0]["views"]["choice"].shape[1] == len(dataset["schema"]["instant"]) + 4
    settings["responses"][0]["token_ids"][0] = 999
    write_json(choice / "settings.json", settings)
    with pytest.raises(ValueError, match="identity differs"):
        attach_choice_features(dataset, choice, 16)
