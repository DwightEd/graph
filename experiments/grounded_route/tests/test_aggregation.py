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


def test_route_moments_preserve_small_nonzero_spread():
    moments = route_moments(
        message=torch.tensor([[1.0], [1.0002]]),
        weight=torch.tensor([0.5, 0.5]),
        target=torch.tensor([0, 0]),
        head=torch.tensor([0, 0]),
        role=torch.tensor([0, 0]),
        response_count=1,
        head_count=1,
    )

    # Population variance is (1e-4)^2. route_moments adds 1e-8 before sqrt.
    expected = torch.tensor((2.0e-8) ** 0.5)
    torch.testing.assert_close(
        moments.spread[0, 0, 0, 0],
        expected,
        atol=1e-7,
        rtol=1e-4,
    )
