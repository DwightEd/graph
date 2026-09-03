import torch

from experiments.grounded_anchor_flow.flow import (
    EVIDENCE,
    RESPONSE,
    analyze_flow,
    path_closure,
    token_transition,
)


def graph_from_support(support: torch.Tensor, response_start: int, evidence_mask):
    response, tokens = support.shape
    token_flow = torch.zeros(response, tokens, 4)
    token_flow[..., 0] = support
    token_flow[..., 2] = support
    token_flow[..., 3] = support
    return {
        "token_flow": token_flow,
        "response_start": response_start,
        "evidence_mask": torch.tensor(evidence_mask, dtype=torch.bool),
    }


def test_transition_normalizes_targets_and_closure_sums_all_paths():
    support = torch.tensor(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0],
        ]
    )
    transition = token_transition(support[..., None].expand(-1, -1, 4), 2)
    response = transition[2:]
    closure = path_closure(response)

    torch.testing.assert_close(transition[:2, 0].sum(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(transition[:3, 1].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.count_nonzero(torch.tril(response)) == 0
    expected = torch.tensor([[1.0, 0.75], [0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(closure, expected)

    response = torch.tensor(
        [[0.0, 0.5, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    closure = path_closure(response)
    torch.testing.assert_close(closure[0, 2], torch.tensor(0.125, dtype=torch.float64))


def test_prompt_relay_reaches_anchor_without_becoming_response_seeded_transit():
    # evidence token 0 -> response anchor 0 -> response target 1
    support = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = analyze_flow(graph_from_support(support, 1, [True]), future_window=4)

    assert result.valid.tolist() == [True, True]
    assert result.anchor_valid.tolist() == [False, True]
    torch.testing.assert_close(result.source_path_posterior[1, EVIDENCE], torch.tensor(0.5))
    torch.testing.assert_close(result.source_path_posterior[1, RESPONSE], torch.tensor(0.5))
    torch.testing.assert_close(result.response_seeded_anchor_flow[1], torch.tensor(0.0))
    assert result.dominant_anchor[1].item() == 0


def test_response_seeded_multihop_path_is_detected_at_the_anchor():
    # response0 -> response1 -> response2, with no prompt route to the target.
    support = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    result = analyze_flow(graph_from_support(support, 1, [True]), future_window=4)

    assert result.valid[2]
    assert result.anchor_valid[2]
    torch.testing.assert_close(result.response_seeded_path_share[2], torch.tensor(1.0))
    torch.testing.assert_close(result.response_seeded_anchor_flow[2], torch.tensor(1.0))
    assert result.dominant_anchor[2].item() == 1
    assert result.future_anchor_influence[1] > 0


def test_direct_response_dependency_is_not_anchor_mediation():
    support = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = analyze_flow(graph_from_support(support, 1, [True]))

    assert result.valid[1]
    assert not result.anchor_valid[1]
    torch.testing.assert_close(result.response_seeded_path_share[1], torch.tensor(1.0))
    assert torch.isnan(result.response_seeded_anchor_flow[1])


def test_balanced_prior_does_not_reward_a_group_for_having_more_tokens():
    # Two prompt tokens and one response token have identical direct paths.
    support = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]
    )
    result = analyze_flow(graph_from_support(support, 2, [True, False]))

    torch.testing.assert_close(result.response_seeded_path_share[1], torch.tensor(0.5))
    torch.testing.assert_close(result.source_path_posterior[1, :2].sum(), torch.tensor(0.5))
