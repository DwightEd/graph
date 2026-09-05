from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.reanchor_flow.worlds import load_world, save_world

from .etcc_helpers import paired_world


def test_paired_world_round_trip_and_causal_prefix(tmp_path) -> None:
    world = paired_world()
    path = tmp_path / "pair.npz"
    save_world(path, world)
    loaded = load_world(path)
    assert loaded.sample_id == world.sample_id
    assert loaded.candidate_unit_id == (1, 2)
    assert loaded.targets == world.targets
    torch.testing.assert_close(loaded.clean_token_ids, world.clean_token_ids)

    prefix = loaded.prefix(loaded.targets[0])
    assert len(prefix.clean_token_ids) == loaded.targets[0].query_position + 2
    assert len(prefix.units.token_unit_id) == len(prefix.clean_token_ids) - 1


def test_paired_world_rejects_response_changes_and_unnamed_changes() -> None:
    world = paired_world()
    response_changed = world.corrupt_token_ids.clone()
    response_changed[-1] = 11
    with pytest.raises(ValueError, match="teacher-forced response"):
        replace(world, corrupt_token_ids=response_changed).check()

    outside_changed = world.corrupt_token_ids.clone()
    outside_changed[0] = 12
    with pytest.raises(ValueError, match="outside candidate"):
        replace(world, corrupt_token_ids=outside_changed).check()


def test_every_named_candidate_must_change() -> None:
    world = paired_world()
    only_first = world.corrupt_token_ids.clone()
    only_first[2] = world.clean_token_ids[2]
    with pytest.raises(ValueError, match="every candidate"):
        replace(world, corrupt_token_ids=only_first).check()


def test_isolated_world_reverts_every_other_candidate() -> None:
    world = paired_world()
    isolated = world.isolate(1)
    assert isolated.candidate_unit_id == (1,)
    assert int(isolated.corrupt_token_ids[1]) == int(world.corrupt_token_ids[1])
    assert int(isolated.corrupt_token_ids[2]) == int(world.clean_token_ids[2])
    torch.testing.assert_close(
        isolated.corrupt_token_ids[world.response_start :],
        world.clean_token_ids[world.response_start :],
    )
