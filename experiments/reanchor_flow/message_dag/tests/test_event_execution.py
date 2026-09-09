"""Execution changes must preserve physical events, derivatives and cache bounds."""
from collections import Counter

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.message_dag import native_layer
from experiments.reanchor_flow.message_dag.cache import NativeCache
from experiments.reanchor_flow.message_dag.differential import DifferentialLayer, final_directions
from experiments.reanchor_flow.message_dag.event_run import event_group_size
from experiments.reanchor_flow.message_dag.event_trace import trace_events
from experiments.reanchor_flow.tests.test_message_lineage import capture_fixture


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:0',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_layer_grouping_preserves_events_and_reuses_native_inputs(tmp_path,monkeypatch,device):
    torch.set_num_threads(2)
    path,_,weights = capture_fixture(tmp_path,ids=np.arange(20)%29,qk_scale=8)
    weights.device = device
    sites = np.array([[layer,head,row] for row in (2,4,6,8) for layer in range(3) for head in range(4)])
    counts = Counter()
    get = weights.get
    def counted_get(name):
        counts[name] += 1
        return get(name)
    monkeypatch.setattr(weights,'get',counted_get)
    chunks = native_layer.attention_chunks
    def counted_attention(*args,**kwargs):
        counts['attention_reconstruction'] += 1
        yield from chunks(*args,**kwargs)
    monkeypatch.setattr(native_layer,'attention_chunks',counted_attention)
    profile = {}
    with NativeCache(path,weights) as cache:
        grouped = trace_events(cache,sites,event_batch=2,query_chunk=3,window=2,profile=profile)
        assert counts['lm_head.weight']==1
        assert counts['attention_reconstruction']==cache.layers
        assert profile['layer_builds']==cache.layers
        assert all(counts[f'model.layers.{l}.self_attn.q_proj.weight']==1 for l in range(cache.layers))
        single = [trace_events(cache,sites[sites[:,2]==row],query_chunk=3,window=2)[0] for row in (2,4,6,8)]
        assert counts['lm_head.weight']==1  # reused even across separate groups
    assert not cache.event_readouts
    assert len(grouped)==len(single)==4
    for a,b in zip(grouped,single):
        for name in a:
            if np.asarray(a[name]).dtype.kind in 'fc':
                np.testing.assert_allclose(a[name],b[name],atol=3e-6,rtol=3e-4,equal_nan=True,err_msg=name)
            else: np.testing.assert_array_equal(a[name],b[name])


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_attention_cache_and_batched_seeds_preserve_native_values(tmp_path,dtype):
    path,_,weights = capture_fixture(tmp_path,dtype=dtype,qk_scale=8)
    with NativeCache(path,weights) as cache:
        op = DifferentialLayer(cache,1,chunk=3)
        original = torch.cat([a for _,_,a in op.rows_attention()],dim=1)
        op.cache_attention(max_bytes=0)
        assert op.cached_attention is None
        op.cache_attention()
        cached = torch.cat([a for _,_,a in op.rows_attention()],dim=1)
        torch.testing.assert_close(cached,original,atol=0,rtol=0)
        sites = [(1,2),(3,4),(0,6)]
        seeds = op.remote_seeds(sites,2)
        source = torch.arange(len(cache.trace['token_ids']))
        ordinary = ~torch.as_tensor(cache.trace['special_mask'])
        for head,row in sites:
            a = original[head,row]*(((op.rows[row]-source)>2)&ordinary)
            expected = op.output_blocks[head]@(a@op.v[head])
            torch.testing.assert_close(seeds[head,row][0],expected,atol=2e-8,rtol=2e-5)
            np.testing.assert_array_equal(seeds[head,row][1],a.numpy())


def test_group_budget_changes_execution_size_without_dropping_remainder():
    for rows,hidden in ((244,4096),(2048,4096),(17,16)):
        per_event = 9*rows*hidden*4
        for batch in (2,4):
            assert event_group_size(rows,hidden,batch,0)==batch
            group = event_group_size(rows,hidden,batch,1)
            assert group>=batch and group%batch==0
            if group>batch: assert group*per_event<=2**30
            events = list(range(243))
            selected = [e for i in range(0,len(events),group) for e in events[i:i+group]]
            assert selected==events


def test_readout_cache_distinguishes_explicit_contrasts(tmp_path):
    path,_,weights = capture_fixture(tmp_path)
    with NativeCache(path,weights) as cache:
        original = final_directions(cache)
        contrast = [dict(target=int(cache.trace['row_position'][2]+1),
                         positive_id=int(original[2][2]),negative_id=int(original[1][2]))]
        changed = final_directions(cache,contrast)
        torch.testing.assert_close(changed[0][2],-original[0][2])
        assert len(cache.event_readouts)==2
        assert final_directions(cache) is original
