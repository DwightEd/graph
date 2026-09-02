import torch

from experiments.evidence_route_state.capture import prediction_events


def test_response_tokens_keep_predictor_q_separate_from_target_q_plus_one():
    token_ids = torch.tensor([10, 11, 12, 20, 21, 22])

    events = prediction_events(token_ids, response_start=3)

    assert events.query_position.tolist() == [2, 3, 4]
    assert events.prediction_position.tolist() == [3, 4, 5]
    assert events.target_id.tolist() == [20, 21, 22]
    torch.testing.assert_close(
        events.prediction_position,
        events.query_position + 1,
        rtol=0,
        atol=0,
    )

    # The first response token is predicted at the last prompt position. It
    # therefore has no response-history source in its physical computation.
    assert events.query_position[0] < 3
