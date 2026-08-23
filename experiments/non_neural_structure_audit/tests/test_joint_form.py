import numpy as np

from experiments.non_neural_structure_audit.joint_form import compare_joint_forms


def test_interaction_model_detects_xor_that_additive_logistic_misses():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(240, 2))
    labels = ((x[:, 0] * x[:, 1]) > 0).astype(np.int8)
    groups = np.repeat(np.arange(60), 4)

    result = compare_joint_forms(
        labels,
        x,
        groups,
        direct_columns=(0,),
        folds=5,
        seed=9,
    )

    by_name = {row["model"]: row for row in result}
    assert by_name["interaction"]["mean_auprc"] > 0.9
    assert (
        by_name["interaction"]["mean_auprc"] > by_name["additive"]["mean_auprc"] + 0.2
    )
