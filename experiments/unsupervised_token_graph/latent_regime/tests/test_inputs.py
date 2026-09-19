import numpy as np

from experiments.unsupervised_token_graph.latent_regime.inputs import (
    channel_observations,
)


class Channel:
    queries = np.array([4])

    def row(self, index):
        return (
            np.array([0, 2, 4]),
            np.array([.2, .3, .4]),
        )


class Sample:
    prompt_length = 4
    response_length = 2


def test_channel_observation_is_self_and_total_prompt_mass():
    values, present = channel_observations(Channel(), Sample())
    np.testing.assert_allclose(values[1], [.4, .5])
    assert bool(present[1])
