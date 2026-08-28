import torch

from experiments.attention_mechanism_audit.audit import mechanism_effects


def test_mechanism_effects_are_the_registered_factorial_contrasts():
    scores = {
        "full": {"target_logprob": torch.tensor([-1.0, -2.0])},
        "evidence_removed": {"target_logprob": torch.tensor([-3.0, -4.0])},
        "response_removed": {"target_logprob": torch.tensor([-2.0, -5.0])},
        "evidence_response_removed": {
            "target_logprob": torch.tensor([-4.0, -7.0]),
            "target_margin": torch.tensor([0.5, -0.5]),
        },
    }
    scores["full"]["target_margin"] = torch.tensor([0.25, -0.25])

    effects = mechanism_effects(scores)

    torch.testing.assert_close(
        effects["evidence_message_effect"], torch.tensor([2.0, 2.0])
    )
    torch.testing.assert_close(
        effects["response_message_effect"], torch.tensor([1.0, 3.0])
    )
    torch.testing.assert_close(
        effects["evidence_message_effect_without_response"],
        torch.tensor([2.0, 2.0]),
    )
    torch.testing.assert_close(
        effects["response_message_effect_without_evidence"],
        torch.tensor([1.0, 3.0]),
    )
    torch.testing.assert_close(
        effects["evidence_response_message_interaction"], torch.zeros(2)
    )
    assert effects["evidence_response_removed_logprob"] is scores[
        "evidence_response_removed"
    ]["target_logprob"]
    assert effects["evidence_response_removed_margin"] is scores[
        "evidence_response_removed"
    ]["target_margin"]
    assert effects["full_margin"] is scores["full"]["target_margin"]
