import torch

from experiments.grounded_route.aggregation import route_moments


def test_route_moments_separate_prompt_and_response_neighbours():
    message = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
        ]
    )
    moments = route_moments(
        message=message,
        weight=torch.tensor([1.0, 1.0, 2.0]),
        target=torch.tensor([0, 0, 0]),
        head=torch.tensor([0, 0, 0]),
        role=torch.tensor([0, 0, 1]),
        response_count=1,
        head_count=1,
    )

    assert torch.allclose(moments.mean[0, 0, 0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(moments.mean[0, 0, 1], torch.tensor([0.0, 2.0]))
    assert torch.allclose(moments.mass[0, 0], torch.tensor([2.0, 2.0]))
    assert moments.spread[0, 0, 0, 0] > 0.9
