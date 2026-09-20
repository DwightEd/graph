from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.unsupervised_token_graph.head_roles.blocks import ordinary_blocks, permutation, sample_swaps
from experiments.unsupervised_token_graph.head_roles.metrics import choose_masks, swap_scores
from experiments.unsupervised_token_graph.head_roles.native import capture, probe_query, rotate
from experiments.unsupervised_token_graph.head_roles.pipeline import masked_contrast, representations
from experiments.unsupervised_token_graph.head_roles.profile import query_plan, save_capture
from experiments.unsupervised_token_graph.head_roles.inputs import channel_features


def test_paper_scores_distinguish_following_position_from_following_content():
    before = np.array([[[.8, .1]], [[.2, .9]]])
    positional = swap_scores(before, before)
    symbolic = swap_scores(before, before[..., ::-1])
    assert positional["positional"][0] == pytest.approx(1.)
    assert positional["gap"][0] < -.5
    assert symbolic["symbolic"][0] == pytest.approx(1.)
    assert symbolic["gap"][0] > .5


def test_uniform_and_zero_mass_are_not_symbolic_evidence():
    uniform = swap_scores(np.full((3, 1, 2), .2), np.full((3, 1, 2), .2))
    assert uniform["positional"][0] == pytest.approx(1.)
    assert uniform["symbolic"][0] == pytest.approx(1.)
    assert uniform["contrast"][0] == 0
    empty = swap_scores(np.zeros((2, 1, 2)), np.zeros((2, 1, 2)))
    assert empty["valid_swaps"][0] == 0
    assert np.isnan(empty["gap"][0])


def test_swaps_exclude_special_keys_query_and_future():
    ids = np.array([1, 4, 5, 6, 7, 1, 8, 9, 10, 11, 12, 13])
    blocks = ordinary_blocks(ids, 10, [1], 2)
    for pair in sample_swaps(blocks, 8, 17):
        order = permutation(len(ids), pair)
        np.testing.assert_array_equal(order[[0, 5, 10, 11]], [0, 5, 10, 11])
        np.testing.assert_array_equal(np.sort(order), np.arange(len(ids)))


def test_rope_uses_half_pairs_and_native_scaling():
    vector = torch.tensor([[1., 2., 3., 4.]])
    actual = rotate(vector, torch.zeros(4), torch.ones(4))
    torch.testing.assert_close(actual, torch.tensor([[-3., -4., 1., 2.]]))


def test_constant_keys_are_positional_and_no_rope_is_symbolic():
    length, position = 8, 7
    pairs = np.array([[[0, 2], [4, 6]], [[0, 2], [2, 4]]])
    query = torch.tensor([[2., 1., .1, .2]])
    phases = torch.arange(length)[:, None] * torch.tensor([.7, 1.1])
    cosine = torch.cat((phases.cos(), phases.cos()), -1)
    sine = torch.cat((phases.sin(), phases.sin()), -1)
    ordinary = torch.ones(length, dtype=torch.bool)
    keys = torch.ones((1, length, 4))
    pos = probe_query(query, keys, cosine, sine, position, pairs, ordinary, .5, .05)[0]
    assert pos["positional"][0] == pytest.approx(1., abs=1e-6)
    assert pos["gap"][0] < -.05
    keys[0, :, 0] = torch.arange(length)
    sym = probe_query(query, keys, torch.ones_like(cosine), torch.zeros_like(sine),
                      position, pairs, ordinary, .5, .05)[0]
    assert sym["symbolic"][0] == pytest.approx(1., abs=1e-6)
    assert sym["gap"][0] > .05


def tiny_llama(heads=4):
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=2,
        max_position_embeddings=64)
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def test_native_gqa_reconstruction_future_invariance_and_no_model_mutation(tmp_path):
    model = tiny_llama()
    ids = np.arange(10, 24)
    positions = [7]
    swaps = [sample_swaps(ordinary_blocks(ids, 7, [10], 2), 3, 17)]
    before = model(torch.tensor(ids[None])).logits.detach()
    first = capture(model, ids, positions, swaps, [10], .05)
    after = model(torch.tensor(ids[None])).logits.detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    changed = ids.copy()
    changed[8:] += 15
    second = capture(model, changed, positions, swaps, [10], .05)
    for layer in first:
        assert first[layer][0]["scores"]["reconstruction_error"].max() < 1e-6
        np.testing.assert_allclose(first[layer][0]["before"], second[layer][0]["before"], atol=1e-7)
        np.testing.assert_allclose(first[layer][0]["after"], second[layer][0]["after"], atol=1e-7)
    save_capture(tmp_path / "capture.npz", first, positions, swaps)
    with np.load(tmp_path / "capture.npz", allow_pickle=False) as saved:
        assert saved["L0__gap"].shape == (1, 4)


def test_random_removal_matches_layer_counts_and_keeps_ambiguous_heads():
    gap = np.array([-.8, -.7, .8, 0., -.6, .6, .7, 0.])
    reliable = np.ones(8, bool)
    reliable[1] = False
    masks = choose_masks(np.arange(2), np.arange(4), gap, reliable, .5, 17, 3)
    assert (~masks["drop_positional"]).sum() == 2
    assert masks["drop_positional"][1]
    for name, keep in masks.items():
        if name.startswith("random"):
            np.testing.assert_array_equal((~keep).reshape(2, 4).sum(1), [1, 1])


def test_removed_heads_cannot_leak_back_through_layer_centering():
    values = np.random.default_rng(17).normal(size=(10, 2, 4, 3))
    keep = np.array([True, False, True, True, True, False, True, True])
    first = representations(values, {"selected": keep})
    changed = values.copy()
    changed[:, :, 1] += 1000
    second = representations(changed, {"selected": keep})
    for method in first:
        np.testing.assert_array_equal(first[method], second[method])


def test_masking_occurs_before_reference_fitting_not_cancelled_by_rescaling():
    values = np.ones((4, 1, 4, 2))
    keep = np.array([True, False, True, False])
    embedded = representations(values, {"all": np.ones(4, bool), "drop": keep})
    assert embedded["raw__all"].shape[1] == 8
    assert embedded["raw__drop"].shape[1] == 4


def test_query_plan_uses_observed_prediction_positions():
    saved = dict(coverage=np.array([False, True, True, True]), prompt_length=8,
                 token_ids=np.arange(12))
    args = SimpleNamespace(probe_queries=2, block_width=2, swaps=4, seed=17)
    positions, swaps = query_plan(saved, args, [])
    assert positions == [8, 10]
    assert all(pair.max() <= query for query, pairs in zip(positions, swaps) for pair in pairs)


def test_route_features_exclude_special_future_and_preserve_nonself_information():
    sample = SimpleNamespace(response_length=2, prompt_length=4,
                             token_ids=np.array([99, 1, 2, 3, 4, 5]))
    channel = SimpleNamespace(queries=[4], row=lambda _: (
        np.array([0, 1, 2, 4, 5]), np.array([.9, .025, .025, .05, .99])))
    values, masses = channel_features(channel, sample, [99], 10)
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1, :4], [.5, .5, 0., 0.])
    assert masses[1] == pytest.approx(.1)
    concentrated = SimpleNamespace(queries=[4], row=lambda _: (
        np.array([1, 4]), np.array([.05, .05])))
    other, _ = channel_features(concentrated, sample, [99], 10)
    assert values[1, 0] == other[1, 0]
    assert values[1, 4] > other[1, 4]
