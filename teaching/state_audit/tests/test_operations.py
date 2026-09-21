import numpy as np
import pytest
import torch

from state_audit.capture import CaptureSpec, capture_sample
from state_audit.experiments.interventions import score_targets
from state_audit.intervention import intervene
from state_audit.operations import Delete, Inject, Replace, Target, apply_operations
from state_audit.state import ModelState
from state_audit.storage import read_json


def test_cartesian_selection_and_order_keep_other_coordinates_unchanged():
    original = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
    target = Target("value", (0,), positions=(1, 3), heads=(0, 2, 3), features=(0, 2))
    changed = apply_operations(original, [Delete(target), Inject(target, 2.0)])
    expected = original.clone()
    for head in (0, 2, 3):
        for position in (1, 3):
            expected[head, position, [0, 2]] = 2.0
    torch.testing.assert_close(changed, expected)
    torch.testing.assert_close(original, torch.arange(60).reshape(4, 5, 3).float())


@pytest.mark.parametrize(
    "site", ["query", "key", "value", "mlp_activation", "embedding", "final_hidden"]
)
def test_replacement_changes_native_site_and_identity_replacement_is_exact(
    tiny_run, tmp_path, site
):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    state = ModelState.open(directory / "trace")
    query = answer["prompt_length"] + 2
    global_site = site in ("embedding", "final_hidden")
    target = Target(site, () if global_site else (0,), positions=(query,))
    layer = None if global_site else 0
    donor = state.at(site, [query], layer)
    baseline = score_targets(model, answer, [3])
    replaced = score_targets(model, answer, [3], [Replace(target, donor)])
    assert replaced == baseline
    spec = CaptureSpec(layers=(0,), representations=(site,))
    with intervene(model, [Delete(target)]):
        capture_sample(model, answer, tmp_path / "changed", spec)
    changed_state = ModelState.open(tmp_path / "changed")
    changed = changed_state.at(site, [query], layer)
    assert np.all(changed == 0)
    assert score_targets(model, answer, [3]) == baseline
