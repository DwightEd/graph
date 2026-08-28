import torch

from ..schema import HISTORY, PROMPT, SELF, source_roles


def test_source_roles_keep_future_positions_unavailable():
    role = source_roles(torch.arange(7), target=4, response_start=3)
    assert role.tolist() == [PROMPT, PROMPT, PROMPT, HISTORY, SELF, -1, -1]
