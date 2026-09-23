"""Source/choice identity and native budget alignment for cached transport inputs."""

import numpy as np
import pytest
from state_audit.storage import write_arrays, write_json

from experiments.native_support.transport_observations import (
    load_observations,
    observe_token,
)


def native_fixture(target):
    response = {"id": "native-example", "prompt_length": 5,
                "token_ids": [9, 3, 4, 5, 6, 7, 9, 8, 10],
                "token_text": ["<s>", "Evidence ", "one.", "Second.", " Answer:",
                               "A", "<s>", "B", "C"]}
    sources = {"blocks": [[1, 2], [3]], "prompt_groups": [5, 0, 0, 1, 4]}
    query = response["prompt_length"] + target - 1
    groups = np.full(query + 1, 2)
    groups[:5] = sources["prompt_groups"]
    groups[query] = 3
    groups[np.asarray(response["token_ids"][:query + 1]) == 9] = 5
    rng = np.random.default_rng(13 + target)
    attention = rng.uniform(.1, 1, (2, 3, query + 1)).astype(np.float32)
    attention /= attention.sum(-1, keepdims=True)
    magnitude = attention * rng.uniform(.2, 2, attention.shape)
    current = {"target": np.asarray(target), "query": np.asarray(query),
               "token_id": np.asarray(response["token_ids"][5 + target]),
               "attention": attention, "group_ids": groups,
               "group_attention": attention @ np.eye(6, dtype=np.float32)[groups],
               "edge_value_energy": magnitude ** 2,
               "edge_response_energy": magnitude ** 2 * 4,
               "group_effect": rng.normal(size=(2, 3, 6, 5)).astype(np.float32),
               "ffn_effect": rng.normal(size=(2, 5)).astype(np.float32),
               "state": np.asarray([target + 1., target + 2.]),
               "candidate_ids": np.asarray([response["token_ids"][5 + target], 20, 21]),
               "candidate_logits": np.asarray([2., 1., 0.]),
               "entropy": np.asarray(.5), "surprisal": np.asarray(1.),
               "candidate_tail_mass": np.asarray(.2)}
    return response, sources, current


def test_budget_uses_edge_message_norms_and_keeps_every_group():
    _response, _sources, current = native_fixture(3)
    measured = observe_token(current, 5, 2, 2)
    expected = np.stack([
        np.sqrt(current["edge_value_energy"])[..., current["group_ids"] == group].sum(-1)
        for group in range(6)
    ], axis=-1)
    np.testing.assert_allclose(measured["message_budget"], expected)
    np.testing.assert_allclose(expected.sum(-1), np.sqrt(current["edge_value_energy"]).sum(-1))
    assert measured["group_effect"].shape == (2, 3, 6, 2)
    assert not np.allclose(expected, np.linalg.norm(current["group_effect"], axis=-1))


def test_choice_contrast_uses_actual_and_competitor_of_same_query():
    _response, _sources, current = native_fixture(2)
    measured = observe_token(current, 5, 2, 2)
    expected = current["group_effect"][..., 2:3] - current["group_effect"][..., 3:]
    np.testing.assert_array_equal(measured["choice_contrast"], expected)
    np.testing.assert_array_equal(measured["candidate_ids"], current["candidate_ids"])
    np.testing.assert_array_equal(measured["ffn_effect"], current["ffn_effect"])
    changed = {**current, "group_effect": current["group_effect"].copy()}
    changed["group_effect"][..., 2:] += .25
    np.testing.assert_allclose(observe_token(changed, 5, 2, 2)["choice_contrast"], expected,
                               atol=3e-7)


def test_head_and_source_permutations_preserve_identity():
    _response, _sources, current = native_fixture(3)
    heads = [2, 0, 1]
    groups = [1, 0, 2, 3, 4, 5]
    changed = dict(current)
    for name in ("attention", "edge_value_energy", "edge_response_energy"):
        changed[name] = current[name][:, heads]
    changed["group_effect"] = current["group_effect"][:, heads][:, :, groups]
    changed["group_attention"] = current["group_attention"][:, heads][:, :, groups]
    changed["group_ids"] = np.asarray(groups)[current["group_ids"]]
    original = observe_token(current, 5, 2, 2)
    permuted = observe_token(changed, 5, 2, 2)
    for name in ("group_effect", "choice_contrast", "group_attention", "message_budget"):
        np.testing.assert_array_equal(permuted[name], original[name][:, heads][:, :, groups])
    np.testing.assert_allclose(permuted["raw_routing"], original["raw_routing"])


@pytest.mark.parametrize("target", [0, 3])
@pytest.mark.parametrize("use_source_mask", [False, True])
def test_routing_buckets_reconstruct_exact_historical_baselines(target, use_source_mask):
    response, _sources, current = native_fixture(target)
    source_mask = np.asarray([True, True, False, True, True]) if use_source_mask else None
    measured = observe_token(current, 5, 2, 2, source_mask)
    positions = np.arange(5 + target)
    ordinary = np.asarray(response["token_ids"][:5 + target]) != 9
    source = (positions < 5) & ordinary
    history = (positions >= 5) & ordinary
    if use_source_mask:
        source[:5] = source_mask
        source[-1] = False
        history = positions >= 5
    difference = history.astype(float) - source
    magnitude = np.sqrt(current["edge_value_energy"])
    expected_route = ((magnitude * difference).sum((-1, -2)) / magnitude.sum((-1, -2))).mean()
    expected_attention = (current["attention"] * difference).sum(-1).mean()
    np.testing.assert_allclose(measured["raw_routing"], expected_route)
    np.testing.assert_allclose(measured["raw_attention"], expected_attention)

    budget = measured["routing_budget"]
    reconstructed = ((budget[..., 3].sum(-1) - budget[..., :3].sum((-1, -2)))
                     / budget.sum((-1, -2))).mean()
    np.testing.assert_allclose(reconstructed, expected_route)
    np.testing.assert_allclose(budget.sum(-1), magnitude.sum(-1))
    if target == 0:
        # In prompt mode ordinary predictor-self is source; in official mode source special is.
        key = 0 if use_source_mask else 4
        np.testing.assert_allclose(budget[..., 2], magnitude[..., key])


def save_fixture(directory):
    response, sources, _current = native_fixture(0)
    write_json(directory / "sources.json", sources)
    for target in range(4):
        _response, _sources, current = native_fixture(target)
        write_arrays(directory / f"token_{target:06d}.npz", **current)
    return response


def test_loading_preserves_history_key_alignment_and_missing_windows(tmp_path):
    response = save_fixture(tmp_path)
    measured, sources = load_observations(response, tmp_path, 2, [9])
    _response, _sources, row = native_fixture(1)
    # Row 1 predicts answer token 1 and reads answer key 0 as its own query/self.
    np.testing.assert_array_equal(measured["history_attention"][1, ..., 0], row["attention"][..., 5])
    assert np.all(measured["history_attention"][0] == 0)
    assert np.all(measured["history_attention"][1, ..., 1:] == 0)
    strict_reads = [native_fixture(target)[2]["attention"][..., 5] for target in (2, 3)]
    np.testing.assert_allclose(measured["future_attention_mean"][0], np.mean(strict_reads, axis=0))
    np.testing.assert_array_equal(measured["future_query_count"], [2, 1, 0, 0])
    np.testing.assert_array_equal(measured["future_observed"], [True, True, False, False])
    assert np.isnan(measured["future_attention_mean"][-2:]).all()
    assert np.isnan(measured["future_response_mean"][-2:]).all()
    assert np.isnan(measured["read_change"][0]).all()
    np.testing.assert_array_equal(measured["read_change_observed"], [False, True, True, True])
    np.testing.assert_array_equal(measured["read_change"][1:], np.diff(measured["group_attention"], axis=0))
    assert measured["group_effect"].shape == (4, 2, 3, 6, 2)
    assert sources[0] == {"id": 0, "positions": [1, 2], "text": "Evidence one.",
                          "semantic_type": "unassigned"}
    assert not (tmp_path / "observations.npz").exists()


def test_loading_rejects_token_alignment_mismatch(tmp_path):
    response = save_fixture(tmp_path)
    _response, _sources, current = native_fixture(2)
    current["token_id"] = np.asarray(999)
    write_arrays(tmp_path / "token_000002.npz", **current)
    with pytest.raises(ValueError, match="token alignment mismatch at 2"):
        load_observations(response, tmp_path, 2, [9])
