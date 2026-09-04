from types import SimpleNamespace

import pytest

from experiments.reanchor_flow import run
from experiments.reanchor_flow.run import parser, validate_args


def test_cli_rejects_empty_capture_and_invalid_curve():
    args = parser().parse_args(["all", "--max-events", "0"])
    with pytest.raises(ValueError, match="max-events"):
        validate_args(args)

    args = parser().parse_args(["evaluate", "--curve-low", "0"])
    with pytest.raises(ValueError, match="straddle"):
        validate_args(args)


def test_cli_accepts_smoke_defaults():
    args = parser().parse_args(["all", "--smoke"])
    validate_args(args)


def test_evaluate_stdout_uses_matched_boundary_contract(monkeypatch, capsys, tmp_path):
    report = {
        "hypothesis_status": {
            "H3_exact_boundary_missed_entry_association": "inconclusive"
        },
        "observer_hypothesis_status": {
            "H1_exposure_adjusted_preference_drift": "inconclusive",
            "H2_natural_boundary_evidence_specificity": "supported",
            "H3_exact_boundary_missed_entry_association": "inconclusive",
        },
        "model_scope": {"generation_claims_allowed": True},
        "correct_boundary_vs_within_claim": {
            "evidence_specificity": {
                "source_mean": 0.2,
                "ci95": [-0.1, 0.4],
                "events": 8,
            }
        },
        "missed_reanchor_at_claim_boundary": {
            "exact_boundary_primary": {
                "source_mean": -0.3,
                "ci95": [-0.8, 0.1],
                "matched_pairs": 3,
                "candidate_hallucinations": 4,
                "sources": 2,
            }
        },
        "recommended_next_step": "collect more data",
    }
    monkeypatch.setattr(run, "evaluate_results", lambda *args, **kwargs: {"QA": report})
    args = SimpleNamespace(
        output=tmp_path,
        model=tmp_path / "model",
        smoke=False,
        cache=tmp_path,
        bootstrap=10,
        seed=1,
        pre_window=5,
        post_window=3,
        curve_low=-5,
        curve_high=10,
    )
    run.evaluate(args)
    assert "pairs=3/4" in capsys.readouterr().out
