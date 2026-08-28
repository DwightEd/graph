import numpy as np
import pytest

from experiments.attention_mechanism_audit import evaluate


def test_fold_preprocessing_fits_median_and_scaler_on_train_only(monkeypatch):
    observed = {}

    class RecordingScaler:
        def fit(self, value):
            observed["fit"] = np.asarray(value).copy()
            return self

        def transform(self, value):
            return np.asarray(value)

    monkeypatch.setattr(evaluate, "StandardScaler", RecordingScaler)
    train = np.asarray(
        [[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0], [5.0, 30.0]]
    )
    test = np.asarray([[1_000_000.0, np.nan]])

    prediction = evaluate._fit_fold_predict(
        train,
        np.asarray([0, 1, 0, 1], dtype=np.int8),
        test,
        seed=7,
    )

    np.testing.assert_allclose(
        observed["fit"],
        [[1.0, 20.0], [3.0, 10.0], [3.0, 20.0], [5.0, 30.0]],
    )
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_primary_fdr_excludes_exploratory_features(monkeypatch):
    p_values = iter((0.01, 1.0e-6, 0.04))

    def fake_source_permutation(label, score, source_id, *, replicates, seed):
        return {
            "available": True,
            "reason": None,
            "source_effects": 4,
            "mean_positive_minus_negative": 1.0,
            "p_value_two_sided": next(p_values),
            "replicates": replicates,
            "algorithm": "within_source_mean_difference_sign_permutation",
        }

    monkeypatch.setattr(
        evaluate,
        "_source_effect_sign_permutation",
        fake_source_permutation,
    )
    rows = evaluate.univariate_answer_report(
        np.asarray(
            [
                [0.0, 3.0, 6.0],
                [1.0, 2.0, 5.0],
                [2.0, 1.0, 4.0],
                [3.0, 0.0, 3.0],
            ]
        ),
        ("primary_a", "exploratory", "primary_b"),
        {"primary_a": "high", "exploratory": "exploratory", "primary_b": "low"},
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray(["a", "b", "c", "d"]),
        primary_names=("primary_a", "primary_b"),
        bootstrap_replicates=31,
        seed=11,
    )
    by_name = {row["feature"]: row for row in rows}

    assert by_name["primary_a"]["source_group_permutation_fdr_q"] == 0.02
    assert by_name["primary_b"]["source_group_permutation_fdr_q"] == 0.04
    assert (
        by_name["exploratory"]["source_group_permutation"]["p_value_two_sided"]
        == 1.0e-6
    )
    assert by_name["exploratory"]["included_in_primary_fdr"] is False
    assert by_name["exploratory"]["source_group_permutation_fdr_q"] is None
    assert by_name["primary_b"]["direction"] == "low"


def test_source_effect_permutation_uses_sources_as_independent_units():
    source = np.repeat(np.asarray(["a", "b", "c", "d"]), 2)
    label = np.tile(np.asarray([0, 1], dtype=np.int8), 4)
    score = np.asarray([0.0, 1.0, 2.0, 4.0, -1.0, 2.0, 5.0, 9.0])

    result = evaluate._source_effect_sign_permutation(
        label,
        score,
        source,
        replicates=127,
        seed=13,
    )

    assert result["available"] is True
    assert result["source_effects"] == 4
    assert result["mean_positive_minus_negative"] == 2.5
    assert 0.0 < result["p_value_two_sided"] <= 1.0
    assert result["algorithm"] == "within_source_mean_difference_sign_permutation"


def test_source_effect_permutation_is_unavailable_without_mixed_label_sources():
    result = evaluate._source_effect_sign_permutation(
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray(["negative", "negative", "positive", "positive"]),
        replicates=31,
        seed=15,
    )

    assert result["available"] is False
    assert result["source_effects"] == 0
    assert result["p_value_two_sided"] is None
    assert result["reason"] == "fewer_than_two_sources_with_both_answer_classes"


def test_length_confound_oof_scores_every_answer_and_reports_primary_deltas():
    source = np.repeat(np.asarray([f"source-{i}" for i in range(6)]), 2)
    label = np.tile(np.asarray([0, 1], dtype=np.int8), 6)
    prompt = np.asarray([10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60])
    response = np.asarray([2, 3, 2, 4, 3, 4, 2, 5, 3, 5, 4, 6])
    feature = np.column_stack(
        (
            label.astype(np.float64),
            np.asarray([np.nan, 0.0, 1.0, 0.0, 1.0, 0.0] * 2),
        )
    )

    report = evaluate.length_confound_report(
        prompt,
        response,
        feature,
        ("mechanism_a", "mechanism_b"),
        label,
        source,
        folds=3,
        seed=17,
    )

    joint = report["joint_length"]
    assert report["supervised_readability_not_detector"] is True
    assert joint["available"] is True
    assert joint["answers_scored"] == len(label)
    assert joint["samples"] == len(label)
    assert joint["folds_used"] == 3
    for name in ("mechanism_a", "mechanism_b"):
        current = report["primary_feature_increment_over_length"][name]
        assert current["available"] is True
        assert current["length_only"] == joint
        assert current["length_plus_feature"]["answers_scored"] == len(label)
        assert current["auroc_delta"] is not None
        assert current["auprc_delta"] is not None
        assert current["supervised_readability_not_detector"] is True


def test_length_confound_single_class_is_explicitly_unavailable_without_partial_oof():
    label = np.zeros(6, dtype=np.int8)
    report = evaluate.length_confound_report(
        np.arange(6),
        np.arange(6) + 1,
        np.arange(6, dtype=np.float64)[:, None],
        ("mechanism",),
        label,
        np.asarray([f"source-{i}" for i in range(6)]),
        folds=5,
        seed=19,
    )

    joint = report["joint_length"]
    assert joint["available"] is False
    assert joint["reason"] == "single_class_labels"
    assert joint["samples"] == len(label)
    assert joint["answers_scored"] == 0
    increment = report["primary_feature_increment_over_length"]["mechanism"]
    assert increment["available"] is False
    assert increment["auroc_delta"] is None
    assert increment["auprc_delta"] is None


def test_length_confound_requires_at_least_two_source_groups():
    report = evaluate.length_confound_report(
        np.asarray([10, 20]),
        np.asarray([2, 3]),
        np.asarray([[0.0], [1.0]]),
        ("mechanism",),
        np.asarray([0, 1], dtype=np.int8),
        np.asarray(["same-source", "same-source"]),
        folds=5,
        seed=23,
    )

    joint = report["joint_length"]
    assert joint["available"] is False
    assert joint["reason"] == "fewer_than_two_source_groups"
    assert joint["samples"] == 2
    assert joint["answers_scored"] == 0


def test_resampling_and_cv_counts_are_validated_before_work_starts(tmp_path):
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        evaluate.evaluate_artifact(
            tmp_path / "missing-data",
            tmp_path / "missing-artifact.npz",
            tmp_path / "report.json",
            bootstrap_replicates=0,
        )
    with pytest.raises(ValueError, match="cv_folds"):
        evaluate.evaluate_artifact(
            tmp_path / "missing-data",
            tmp_path / "missing-artifact.npz",
            tmp_path / "report.json",
            cv_folds=1,
        )
    with pytest.raises(ValueError, match="bootstrap replicates"):
        evaluate.source_bootstrap(
            np.asarray([0, 1]),
            np.asarray([0.0, 1.0]),
            np.asarray(["a", "b"]),
            replicates=0,
            seed=1,
        )
    with pytest.raises(ValueError, match="OOF folds"):
        evaluate.length_confound_report(
            np.asarray([1, 2]),
            np.asarray([1, 2]),
            np.asarray([[0.0], [1.0]]),
            ("feature",),
            np.asarray([0, 1]),
            np.asarray(["a", "b"]),
            folds=1,
            seed=1,
        )
