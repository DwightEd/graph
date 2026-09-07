"""Numerical counterexamples first; optional real tiny-HF-Llama integration."""
import numpy as np
import pytest
import torch

from experiments.reanchor_flow.attention_rhythm import (
    AttentionRhythmObserver, RawMapObserver, RhythmConfig, head_groups,
)
from experiments.reanchor_flow.attention_rhythm_report import (
    analyze_rhythm, bucket_switch, label_observations, pair_alignment, peak_mask,
    source_bootstrap,
)


def collect(a, p, config=RhythmConfig(), chunk=3, norms=None):
    l, h, n, _ = a.shape
    observer = AttentionRhythmObserver(l, h, n, p, np.arange(n) < p // 2, config)
    norms = torch.ones(h, n) if norms is None else norms
    for layer in range(l):
        for begin in range(0, n, chunk):
            observer.observe_rows(layer, begin, a[layer, :, begin:begin + chunk], norms)
    return observer.finish()


def random_attention(l=2, h=4, n=37):
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(l, h, n, n, generator=generator)
    scores.masked_fill_(torch.ones(n, n, dtype=torch.bool).triu(1), -torch.inf)
    return scores.softmax(-1)


def test_streaming_matches_dense_distance_waad_and_inclusive_fai():
    a, p = random_attention(), 5
    c = RhythmConfig(window=7, future_lo=2, future_hi=8)
    result = collect(a, p, c)
    positions = torch.arange(a.shape[-1])
    lag = (positions[:, None] - positions[None]).clamp_min(0)
    np.testing.assert_allclose(result["distance"], (a * lag).sum(-1)[..., p-1:], atol=2e-6)
    np.testing.assert_allclose(result["waad"], (a * lag.clamp_max(7)).sum(-1)[..., p-1:], atol=1e-6)
    for slot, source in enumerate(range(p-1, len(positions))):
        rows = list(range(max(p, source+2), min(len(positions)-1, source+8)+1))
        assert result["fai_count"][slot] == len(rows)
        if rows:
            np.testing.assert_allclose(result["fai"][..., slot], a[..., rows, source].mean(-1), atol=1e-7)
        else:
            assert np.isnan(result["fai"][..., slot]).all()
    np.testing.assert_allclose(result["attention_buckets"].sum(-1), 1, atol=2e-6)


@pytest.mark.parametrize("chunk", [1, 8, 37])
def test_chunk_invariance(chunk):
    a = random_attention()
    left, right = collect(a, 5, chunk=3), collect(a, 5, chunk=chunk)
    for name in ("distance", "waad", "fai", "message_waad", "attention_buckets", "message_buckets"):
        np.testing.assert_allclose(left[name], right[name], atol=2e-6, equal_nan=True)


def test_same_four_buckets_can_hide_large_within_local_sawtooth():
    n, p = 40, 10
    a = torch.zeros(1, 2, n, n)
    for q in range(n):
        a[0, 0, q, max(0, q - 1)] = 1
        a[0, 1, q, 0] = 1
    a[0, 0, 25].zero_()
    a[0, 0, 25, 17] = 1  # distance 8, still the SAME recent-response bucket
    trace = collect(a, p)
    slots = np.array([24, 25, 26]) - (p-1)
    np.testing.assert_equal(trace["waad"][0, 0, slots], [1, 8, 1])
    np.testing.assert_equal(trace["attention_buckets"][0, 0, slots], [[0,0,0,1]] * 3)
    assert not bucket_switch(trace["message_buckets"])[0, 0, slots].any()
    audit = analyze_rhythm(trace)
    assert audit["waad_peaks"][0, 0, 25-p]
    assert audit["bucket_missed_fraction"][0, 0] == 1
    assert trace["local_heads"].tolist() == [0]
    assert trace["global_heads"].tolist() == [1]


def test_message_weighting_is_a_separate_view_not_raw_attention():
    n, p = 24, 6
    a = torch.eye(n)[None, None].repeat(1, 2, 1, 1)
    a[0, :, 20].zero_()
    a[0, :, 20, 19] = .9
    a[0, :, 20, 0] = .1
    norms = torch.ones(2, n)
    norms[:, 0] = 100
    result = collect(a, p, norms=norms)
    slot = 20-(p-1)
    assert result["waad"][0, 0, slot] == pytest.approx(1.9)
    assert result["message_waad"][0, 0, slot] > 9
    assert result["message_buckets"][0, 0, slot].sum() == pytest.approx(10.9)


def test_no_future_is_missing_and_zero_horizon_is_inclusive():
    a = torch.eye(8)[None, None].repeat(1, 2, 1, 1)
    result = collect(a, 3, RhythmConfig(future_lo=0, future_hi=2))
    assert result["fai"][0, 0, -1] == 1  # paper Hlo=0 includes self
    assert result["fai_count"][-1] == 1
    result = collect(a, 3, RhythmConfig(future_lo=1, future_hi=2))
    assert result["fai_count"][-1] == 0
    assert np.isnan(result["fai"][..., -1]).all()


def test_flat_curves_do_not_force_events_and_alignment_empty_is_nan():
    assert not peak_mask(np.ones((2, 20))).any()
    observed, expected, n = pair_alignment(np.zeros((2,20),bool), np.zeros((3,20),bool), np.ones(20,bool))
    assert np.isnan(observed).all() and np.isnan(expected).all() and not n.any()


def test_alignment_uses_fai_denominator_and_count_preserving_null():
    w, f = np.zeros((1,10),bool), np.zeros((1,10),bool)
    w[0,3] = True
    f[0,[4,8]] = True
    observed, expected, n = pair_alignment(w, f, np.ones(10,bool))
    assert observed.item() == .5
    assert expected.item() == .2
    assert n.item() == 2


def test_raw_map_is_exact_crop_not_sparse_or_renormalized():
    a = random_attention()
    rows = np.arange(5,14)
    obs = RawMapObserver(np.array([0,7]), 4, rows, 37)
    for l in range(2):
        for begin in range(0,37,3):
            obs.observe_chunk(l, begin, a[l:l+1,:,begin:begin+3], None, None)
    np.testing.assert_array_equal(obs.maps[0], a[0,0,rows])
    np.testing.assert_array_equal(obs.maps[1], a[1,3,rows])


def test_q_to_q_plus_one_label_alignment_includes_first_predictor():
    trace = collect(random_attention(n=9), 5)
    # Four predicted response tokens; fifth row is a paper response row only.
    trace["waad"][:] = np.array([100,2,8,4,999])
    labels = np.array([0,0,1,0])
    gap = label_observations(trace, labels)["waad_matched_h_minus_n"]
    # Only mixed log2 bin is response indices 1,2: 8-2, not 999 or 100.
    np.testing.assert_array_equal(gap, np.full((2,4),6))


def test_source_balance_does_not_treat_repeated_answers_as_independent():
    rows = [{"source_id":"a", "x":np.array([1.])}] * 10
    rows += [{"source_id":"b", "x":np.array([3.])}]
    result = source_bootstrap(rows, "x")
    assert result["sources"] == 2
    assert result["mean"].item() == 2
    assert result["ci95"] is None


def test_duplicate_rows_and_incomplete_capture_are_rejected():
    obs = AttentionRhythmObserver(1,2,8,3,np.zeros(8,bool),RhythmConfig())
    a = torch.eye(8)[None].repeat(2,1,1)
    obs.observe_rows(0,2,a[:,2:4],torch.ones(2,8))
    with pytest.raises(ValueError, match="duplicate"):
        obs.observe_rows(0,2,a[:,2:4],torch.ones(2,8))
    with pytest.raises(ValueError, match="incomplete"):
        obs.finish()


def test_native_chunk_adapter_wo_norm_without_full_message_materialization():
    a = random_attention(l=1,h=4,n=15)
    torch.manual_seed(4)
    value = torch.randn(1,4,15,3)
    output = torch.randn(12,12)
    obs = AttentionRhythmObserver(1,4,15,5,np.zeros(15,bool),RhythmConfig())
    for begin in range(0,15,4):
        obs.observe_chunk(0,begin,a[:,:,begin:begin+4],value,output)
    blocks = output.reshape(12,4,3).permute(1,0,2)
    norm = torch.einsum("hDd,hsd->hsD", blocks, value[0]).norm(dim=-1)
    direct = collect(a,5,norms=norm)
    np.testing.assert_allclose(obs.finish()["message_waad"],direct["message_waad"],atol=3e-6)


def test_real_tiny_llama_matches_hf_attention_and_chunked_capture():
    transformers = pytest.importorskip("transformers")
    from experiments.reanchor_flow.attention_rhythm import capture_rhythm
    config = transformers.LlamaConfig(vocab_size=61, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, attention_dropout=0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = transformers.LlamaForCausalLM(config).eval()
    ids = torch.arange(30) % 61
    trace = capture_rhythm(model, ids, 5, np.zeros(30,bool), query_chunk=3,
                           map_tokens=10, explicit_heads=((0,0),(1,3)))
    with torch.no_grad():
        native = model(ids[None],output_attentions=True,use_cache=False)
    rows = trace["map_query_position"]
    for i,(layer,head) in enumerate(((0,0),(1,3))):
        np.testing.assert_allclose(trace["attention_maps"][i],native.attentions[layer][0,head,rows],atol=2e-6)


def test_zero_message_budget_is_unknown_not_a_zero_distance():
    result = collect(random_attention(n=15), 5, norms=torch.zeros(4,15))
    assert np.isnan(result["message_waad"]).all()
    assert np.isnan(result["message_fai"]).all()
    assert not result["message_fai_count"].any()
