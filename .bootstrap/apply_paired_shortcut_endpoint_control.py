from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


evaluate_path = "experiments/attention_mechanism_audit/evaluate.py"
replace_once(
    evaluate_path,
    '''    return result


def _position_match_design(
''',
    '''    return result


def _paired_source_bootstrap_difference(
    label: np.ndarray,
    observed: np.ndarray,
    rewired: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap observed-minus-rewired metrics on identical source clusters."""

    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size != 2:
            continue
        values.append(
            (
                roc_auc_score(label[index], observed[index])
                - roc_auc_score(label[index], rewired[index]),
                average_precision_score(label[index], observed[index])
                - average_precision_score(label[index], rewired[index]),
            )
        )
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "replicates": 0,
            "auroc_difference_ci95": [None, None],
            "average_precision_difference_ci95": [None, None],
        }
    return {
        "replicates": int(len(values)),
        "auroc_difference_ci95": [
            float(value) for value in np.quantile(values[:, 0], (0.025, 0.975))
        ],
        "average_precision_difference_ci95": [
            float(value) for value in np.quantile(values[:, 1], (0.025, 0.975))
        ],
    }


def shortcut_endpoint_control_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Compare observed and rewired shortcut candidates on one paired token set."""

    observed_name = "shortcut_route_candidate_mean"
    rewired_name = "shortcut_route_rewired_control_mean"
    required = (
        "label",
        "source_id",
        observed_name,
        rewired_name,
        f"{observed_name}__valid",
        f"{rewired_name}__valid",
    )
    summary: dict[str, object] = {
        "estimand": "observed_candidate_minus_adjacent_endpoint_rewire",
        "token_scope": "intersection_of_observed_and_rewired_fixed_validity_masks",
        "bootstrap_unit": "source_id_cluster",
        "valid_tokens": 0,
        "valid_sources": 0,
        "observed_auroc": None,
        "rewired_auroc": None,
        "auroc_difference": None,
        "auroc_difference_ci95": [None, None],
        "observed_average_precision": None,
        "rewired_average_precision": None,
        "average_precision_difference": None,
        "average_precision_difference_ci95": [None, None],
        "bootstrap_replicates": 0,
    }
    missing = [name for name in required if name not in arrays]
    if missing:
        summary["unavailable_reason"] = (
            "paired endpoint-control arrays are missing: " + ", ".join(missing)
        )
        return summary

    observed = np.asarray(arrays[observed_name], dtype=np.float64)
    rewired = np.asarray(arrays[rewired_name], dtype=np.float64)
    valid = np.asarray(arrays[f"{observed_name}__valid"], dtype=bool)
    valid &= np.asarray(arrays[f"{rewired_name}__valid"], dtype=bool)
    valid &= np.isfinite(observed) & np.isfinite(rewired)
    label = np.asarray(arrays["label"], dtype=bool)[valid]
    source = np.asarray(arrays["source_id"])[valid]
    observed = observed[valid]
    rewired = rewired[valid]
    summary.update(
        {
            "valid_tokens": int(valid.sum()),
            "valid_sources": int(np.unique(source).size),
        }
    )
    if not len(label):
        summary["unavailable_reason"] = "no token is valid for both endpoint views"
        return summary
    if np.unique(label).size != 2:
        summary["unavailable_reason"] = (
            "paired endpoint-control subset does not contain both labels"
        )
        return summary

    observed_metrics = _binary_metrics(label, observed)
    rewired_metrics = _binary_metrics(label, rewired)
    summary.update(
        {
            "observed_auroc": observed_metrics["auroc"],
            "rewired_auroc": rewired_metrics["auroc"],
            "auroc_difference": (
                observed_metrics["auroc"] - rewired_metrics["auroc"]
            ),
            "observed_average_precision": observed_metrics["average_precision"],
            "rewired_average_precision": rewired_metrics["average_precision"],
            "average_precision_difference": (
                observed_metrics["average_precision"]
                - rewired_metrics["average_precision"]
            ),
        }
    )
    if bootstrap:
        interval = _paired_source_bootstrap_difference(
            label,
            observed,
            rewired,
            source,
            replicates=bootstrap,
            seed=seed,
        )
        summary.update(
            {
                "auroc_difference_ci95": interval["auroc_difference_ci95"],
                "average_precision_difference_ci95": interval[
                    "average_precision_difference_ci95"
                ],
                "bootstrap_replicates": interval["replicates"],
            }
        )
    return summary


def _position_match_design(
''',
)
replace_once(
    evaluate_path,
    '''        "shortcut_route_detection": shortcut_detection_summary(
            arrays, bootstrap=bootstrap, seed=seed + len(SCORE_ORDER)
        ),
        "shortcut_score_definitions": SHORTCUT_SCORE_DEFINITIONS,
''',
    '''        "shortcut_route_detection": shortcut_detection_summary(
            arrays, bootstrap=bootstrap, seed=seed + len(SCORE_ORDER)
        ),
        "shortcut_endpoint_control": shortcut_endpoint_control_summary(
            arrays,
            bootstrap=bootstrap,
            seed=seed + len(SCORE_ORDER) + len(SHORTCUT_SCORE_NAMES),
        ),
        "shortcut_score_definitions": SHORTCUT_SCORE_DEFINITIONS,
''',
)

run_path = "experiments/attention_mechanism_audit/run.py"
replace_once(
    run_path,
    '''    print("POST-HOC matched hallucinated - correct token differences")
''',
    '''    endpoint = report.get("shortcut_endpoint_control", {})
    if endpoint:
        print("PAIRED observed shortcut candidate - adjacent endpoint rewire")
        if endpoint.get("auroc_difference") is None:
            print(
                "paired    shortcut endpoint control          "
                f"dAUROC=n/a dAP=n/a reason={endpoint.get('unavailable_reason', 'n/a')}"
            )
        else:
            print(
                "paired    shortcut endpoint control          "
                f"tokens={endpoint['valid_tokens']} sources={endpoint['valid_sources']} "
                f"observed_AUROC={endpoint['observed_auroc']:.6f} "
                f"rewired_AUROC={endpoint['rewired_auroc']:.6f} "
                f"dAUROC={endpoint['auroc_difference']:.6f} "
                f"CI={ci(endpoint['auroc_difference_ci95'])} "
                f"dAP={endpoint['average_precision_difference']:.6f} "
                f"CI={ci(endpoint['average_precision_difference_ci95'])}"
            )
    print("POST-HOC matched hallucinated - correct token differences")
''',
)

agents_path = "experiments/attention_mechanism_audit/AGENTS.md"
replace_once(
    agents_path,
    '''- The route-completion hypothesis is specific: a supported response relay has
  a full history write explained by direct evidence plus evidence-conditioned
  carrier/gate writes. A shortcut is the residual history write that remains
  aligned with autonomous history after this evidence-support subspace is
  removed. Compare observed endpoints with the fixed adjacent-endpoint rewire.
''',
    '''- The route-completion hypothesis is specific: a supported response relay has
  a full history write explained by direct evidence plus evidence-conditioned
  carrier/gate writes. The fixed shortcut candidate is route incompleteness
  multiplied by the positive signed contribution of autonomous history to the
  full-history direction; do not residualize two algebraically identical
  remainder vectors. Compare observed and adjacent-rewired candidates on their
  common fixed token set and source-bootstrap their AUROC/AP difference.
''',
)

readme_path = "experiments/attention_mechanism_audit/README.md"
replace_once(
    readme_path,
    '''The adjacent swap preserves target rows, heads,
coefficient values, and the response-value multiset while breaking the exact
carrier endpoint. All scalar measurements are post-capture views of the saved
Gram; they do not replace the raw geometry.
''',
    '''The adjacent swap preserves target rows, heads,
coefficient values, and the response-value multiset while breaking the exact
carrier endpoint. All scalar measurements are post-capture views of the saved
Gram; they do not replace the raw geometry.

The endpoint claim is evaluated as a paired control. The observed shortcut
candidate and its adjacent-rewired counterpart are restricted to the
intersection of their preregistered validity masks, then compared by
observed-minus-rewired AUROC and average precision. Confidence intervals
resample the same `source_id` clusters for both views. Separate AUCs on
separate valid subsets are retained for diagnosis but cannot establish that
exact endpoint identity carries information.
''',
)

test_path = Path("experiments/attention_mechanism_audit/tests/test_evaluate.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_paired_shortcut_endpoint_control_uses_common_validity" in test_text:
    raise RuntimeError("paired shortcut endpoint tests already exist")
test_text = test_text.rstrip() + '''


def test_paired_shortcut_endpoint_control_uses_common_validity():
    arrays = {
        "label": np.asarray([False, True, False, True, False, True]),
        "source_id": np.asarray(["a", "a", "b", "b", "c", "c"]),
        "shortcut_route_candidate_mean": np.asarray(
            [0.0, 1.0, 0.1, 0.9, 100.0, -100.0]
        ),
        "shortcut_route_rewired_control_mean": np.asarray(
            [1.0, 0.0, 0.1, 0.9, -100.0, 100.0]
        ),
        "shortcut_route_candidate_mean__valid": np.asarray(
            [True, True, True, True, True, False]
        ),
        "shortcut_route_rewired_control_mean__valid": np.asarray(
            [True, True, True, True, False, True]
        ),
    }

    result = evaluate.shortcut_endpoint_control_summary(
        arrays, bootstrap=0, seed=7
    )

    assert result["valid_tokens"] == 4
    assert result["valid_sources"] == 2
    np.testing.assert_allclose(result["observed_auroc"], 1.0)
    np.testing.assert_allclose(result["rewired_auroc"], 0.25)
    np.testing.assert_allclose(result["auroc_difference"], 0.75)
    assert result["bootstrap_replicates"] == 0


def test_paired_shortcut_endpoint_bootstrap_resamples_identical_sources():
    source = np.repeat(np.asarray(["a", "b", "c", "d"]), 2)
    label = np.tile(np.asarray([False, True]), 4)
    arrays = {
        "label": label,
        "source_id": source,
        "shortcut_route_candidate_mean": np.tile([0.0, 1.0], 4),
        "shortcut_route_rewired_control_mean": np.tile([1.0, 0.0], 4),
        "shortcut_route_candidate_mean__valid": np.ones(8, dtype=bool),
        "shortcut_route_rewired_control_mean__valid": np.ones(8, dtype=bool),
    }

    result = evaluate.shortcut_endpoint_control_summary(
        arrays, bootstrap=50, seed=11
    )

    assert result["bootstrap_replicates"] == 50
    np.testing.assert_allclose(result["auroc_difference"], 1.0)
    np.testing.assert_allclose(result["auroc_difference_ci95"], [1.0, 1.0])
    assert result["average_precision_difference_ci95"][0] is not None
    assert result["average_precision_difference_ci95"][1] is not None


def test_paired_shortcut_endpoint_control_reports_missing_arrays():
    result = evaluate.shortcut_endpoint_control_summary(
        {"label": np.asarray([False, True]), "source_id": np.asarray(["a", "a"])},
        bootstrap=0,
        seed=1,
    )

    assert result["auroc_difference"] is None
    assert "missing" in result["unavailable_reason"]
''' + "\n"
test_path.write_text(test_text, encoding="utf-8")
