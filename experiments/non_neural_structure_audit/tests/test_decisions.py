from experiments.non_neural_structure_audit.config import EvaluationConfig
from experiments.non_neural_structure_audit.decisions import gate_decisions


def test_invalid_endpoint_null_is_inconclusive_and_cannot_authorize_a_graph():
    direct = {
        "relation": "direct_role",
        "tokens": 100,
        "positive_tokens": 20,
        "association_shift_q": 0.01,
        "auprc": 0.2,
        "prevalence": 0.1,
        "auprc_ci_low": 0.15,
    }
    lineage = {
        "relation": "lineage_margin",
        "endpoint_null_valid": False,
        "endpoint_auprc_delta": 0.2,
        "endpoint_auprc_delta_ci_low": 0.1,
        "endpoint_null_q": 0.01,
        "layer_auprc_delta": 0.0,
        "layer_auprc_delta_ci_low": -0.1,
        "layer_shuffle_q": 1.0,
    }

    decisions = gate_decisions(
        [direct, lineage],
        scope="confirmation",
        samples=100,
        source_groups=100,
        positive_responses=50,
        artifact_binding_verified=True,
        full_a0_verified=True,
        config=EvaluationConfig(scope="confirmation"),
    )
    by_audit = {row["audit"]: row for row in decisions}

    assert by_audit["A2"]["status"] == "INCONCLUSIVE_NULL_INVALID"
    assert by_audit["A2"]["authorized_model_component"] == "none"
    assert by_audit["A9"]["authorized_model_component"] == "none"


def test_single_class_confirmation_is_inconclusive_instead_of_raising():
    direct = {
        "relation": "direct_role",
        "tokens": 100,
        "positive_tokens": 0,
        "auprc": float("nan"),
        "prevalence": 0.0,
        "auprc_ci_low": float("nan"),
    }
    lineage = {
        "relation": "lineage_margin",
        "endpoint_null_valid": True,
    }

    decisions = gate_decisions(
        [direct, lineage],
        scope="confirmation",
        samples=100,
        source_groups=100,
        positive_responses=50,
        artifact_binding_verified=True,
        full_a0_verified=True,
        config=EvaluationConfig(scope="confirmation"),
    )
    by_audit = {row["audit"]: row for row in decisions}

    assert by_audit["A1"]["status"] == "INCONCLUSIVE_LOW_POWER"
    assert by_audit["A2"]["status"] == "INCONCLUSIVE_REQUIRED_NULL_MISSING"
    assert by_audit["A4"]["status"] == "INCONCLUSIVE_LOW_POWER"
    assert by_audit["A10"]["status"] == "NOT_AVAILABLE_FROM_ATTENTION_CACHE"


def test_discovery_exposes_an_invalid_null_without_authorizing_a_graph():
    rows = [
        {"relation": "direct_role"},
        {"relation": "lineage_margin", "endpoint_null_valid": False},
    ]

    decisions = gate_decisions(
        rows,
        scope="discovery",
        samples=30,
        source_groups=30,
        positive_responses=10,
        artifact_binding_verified=True,
        full_a0_verified=True,
        config=EvaluationConfig(scope="discovery"),
    )
    by_audit = {row["audit"]: row for row in decisions}

    assert by_audit["A2"]["status"] == "INCONCLUSIVE_NULL_INVALID"
    assert all(row["authorized_model_component"] == "none" for row in decisions[1:])


def test_artifact_binding_alone_does_not_pass_full_a0_or_open_formal_gates():
    decisions = gate_decisions(
        [],
        scope="confirmation",
        samples=100,
        source_groups=100,
        positive_responses=50,
        artifact_binding_verified=True,
        full_a0_verified=False,
        config=EvaluationConfig(scope="confirmation"),
    )

    assert decisions[0]["status"] == "INCONCLUSIVE_A0_CONTROLS_MISSING"
    assert all(row["status"] == "BLOCKED_BY_A0" for row in decisions[1:])
    assert all(row["authorized_model_component"] == "none" for row in decisions)
