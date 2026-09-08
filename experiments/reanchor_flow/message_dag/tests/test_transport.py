"""Independent derivatives, path conservation and graph-specific counterexamples."""
from contextlib import ExitStack

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.message_dag.cache import NativeCache
from experiments.reanchor_flow.message_dag.differential import DifferentialLayer, rms_jvp, rms_vjp
from experiments.reanchor_flow.message_dag.event_trace import trace_events
from experiments.reanchor_flow.message_dag.transport import CutRecorder, prepare_local_readout
from experiments.reanchor_flow.message_dag.transport import last_crossing_edges
from experiments.reanchor_flow.message_dag.differential import final_directions
from experiments.reanchor_flow.message_dag.transport_report import sample_scores
from experiments.reanchor_flow.tests.test_message_lineage import capture_fixture


def trace_with_cut(cache, sites, folder, chunk=3, contrasts=None):
    folder.mkdir(exist_ok=True)
    suffix = folder/'readout.npz'
    prepare_local_readout(cache,suffix,query_chunk=chunk,contrasts=contrasts)
    with np.load(suffix,allow_pickle=False) as reader, ExitStack() as stack:
        recorders = {}
        for row in np.unique(sites[:,2]):
            rec = CutRecorder(folder/f'edges_{int(row)}.npz',cache,row)
            stack.callback(rec.close);recorders[int(row)] = rec
        return trace_events(cache,sites,window=2,query_chunk=chunk,contrasts=contrasts,
                            cut_readout=reader,cut_recorders=recorders)


def test_native_adjoint_includes_rms_rope_gqa_and_softmax_competition(tmp_path):
    torch.set_num_threads(2)
    path,_,weights = capture_fixture(tmp_path,qk_scale=8)
    with NativeCache(path,weights) as cache:
        op = DifferentialLayer(cache,1,chunk=3)
        torch.manual_seed(43)
        delta,reader = torch.randn_like(op.x),torch.randn_like(op.x)
        weight = torch.rand(op.d)
        pairs = [
            (rms_jvp(delta,op.x,weight,op.eps),rms_vjp(reader,op.x,weight,op.eps)),
            (op.mlp_jvp(delta),op.mlp_vjp(reader)),
            (op.attention_jvp(delta[None])[0][0],op.attention_same_vjp(reader)),
        ]
        for forward,backward in pairs:
            torch.testing.assert_close((forward*reader).sum(),(delta*backward).sum(),atol=3e-5,rtol=2e-5)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_last_crossing_recovers_all_paths_and_keeps_every_physical_edge(tmp_path,dtype):
    path,trace,weights = capture_fixture(tmp_path,dtype=dtype,qk_scale=8)
    sites = np.array([[0,1,2],[1,2,2],[0,3,4]])
    with NativeCache(path,weights) as cache:
        events = trace_with_cut(cache,sites,tmp_path/'cuts')
        plain = trace_events(cache,sites,window=2,query_chunk=3)
        changed_chunk = trace_with_cut(cache,sites,tmp_path/'chunk',chunk=8)
    for event,old,chunk in zip(events,plain,changed_chunk):
        np.testing.assert_allclose(event['margin_response'],old['margin_response'],atol=0,rtol=0)
        reconstructed = (event['cut_hop_positive']-event['cut_hop_negative']).sum(-1)
        np.testing.assert_allclose(reconstructed,event['margin_response'][0,1:],atol=2e-8,rtol=3e-4)
        np.testing.assert_allclose(event['cut_carrier_effect'],chunk['cut_carrier_effect'],atol=2e-8,rtol=3e-4)
        b = int(event['event_row']);total = np.zeros_like(event['cut_signed'])
        with np.load(tmp_path/'cuts'/f'edges_{b}.npz',allow_pickle=False) as archive:
            for key in archive.files:
                if not key.startswith('L'): continue
                layer,begin = map(int,key[1:].split('Q'))
                value = archive[key];end = begin+value.shape[1]
                assert value.shape == (weights.config['num_attention_heads'],end-begin,end-1-b,2)
                a = archive['A'+key]
                assert a.shape==value.shape[:-1]
                for j,q in enumerate(range(begin,end)):
                    assert not value[:,j,max(0,q-b):].any()
                total[layer,:,begin:end] += value.sum(2)
        np.testing.assert_allclose(total,event['cut_signed'],atol=0,rtol=0)
        np.testing.assert_allclose(total.sum((0,1,3)),event['margin_response'][0,1:].sum(0),atol=2e-8,rtol=3e-4)
        assert np.max(np.abs(event['cut_signed'][...,1]))>0  # K routing is genuinely nonzero


def test_candidate_reversal_reverses_signed_edges_without_changing_graph(tmp_path):
    path,trace,weights = capture_fixture(tmp_path,qk_scale=8)
    sites = np.array([[0,1,2]])
    with NativeCache(path,weights) as cache:
        original = trace_with_cut(cache,sites,tmp_path/'original')[0]
        i = 6
        contrast = [dict(target=int(original['target_position'][i]),
                         positive_id=int(original['negative_id'][i]),negative_id=int(original['positive_id'][i]))]
        reversed_event = trace_with_cut(cache,sites,tmp_path/'reverse',contrasts=contrast)[0]
    np.testing.assert_allclose(reversed_event['cut_signed'][:,:,i],-original['cut_signed'][:,:,i],atol=2e-9,rtol=1e-4)
    np.testing.assert_allclose(reversed_event['cut_hop_positive'][:,i],original['cut_hop_negative'][:,i],atol=2e-9,rtol=1e-4)
    with np.load(tmp_path/'original/edges_2.npz') as a, np.load(tmp_path/'reverse/edges_2.npz') as b:
        assert a.files==b.files
        for key in a.files:
            if key.startswith('AL'): np.testing.assert_array_equal(a[key],b[key])


def test_full_backward_reader_would_count_a_two_hop_path_twice(tmp_path):
    """An independent autograd oracle demonstrates why the graph partition matters."""
    path,_,weights = capture_fixture(tmp_path,qk_scale=8)
    sites = np.array([[0,1,2]])
    with NativeCache(path,weights) as cache:
        event = trace_with_cut(cache,sites,tmp_path/'cut')[0]
        target = int(np.abs(event['margin_response'][0,2]).argmax())
        ops = [DifferentialLayer(cache,l,chunk=3) for l in range(cache.layers)]
        delta = torch.zeros(1,cache.rows,weights.config['hidden_size'])
        prefix = []
        for l,op in enumerate(ops):
            prefix.append(delta.clone())
            same,cross,_ = op.attention_jvp(delta)
            post = delta+same+cross
            if l==0: post[0,2] += op.remote_seed(1,2,2)[0]
            delta = post+op.mlp_jvp(post)
        reader = torch.zeros_like(ops[0].x)
        reader[target] = final_directions(cache)[0][target]
        full_readers = {}
        for l in reversed(range(cache.layers)):
            op = ops[l]
            post_reader = reader+op.mlp_vjp(reader)
            full_readers[l] = post_reader
            probe = torch.zeros_like(op.x,requires_grad=True)
            same,cross,_ = op.attention_jvp(probe[None])
            derivative = torch.autograd.grad(((same[0]+cross[0])*post_reader).sum(),probe)[0]
            reader = post_reader+derivative
        repeated = 0.
        for l,op in enumerate(ops):
            for _,_,_,edges in last_crossing_edges(op,prefix[l],full_readers[l]):
                repeated += float(edges.sum())
        one,two = event['margin_response'][0,1:,target]
        assert abs(two)>1e-9
        assert repeated == pytest.approx(float(one+2*two),abs=2e-8,rel=1e-4)
        assert abs(repeated-float(one+two))>abs(float(two))*.9
        correct = event['cut_signed'][:,:,target].sum()
        assert correct == pytest.approx(float(one+two),abs=2e-8,rel=1e-4)


def test_scoring_has_no_labels_no_future_event_and_no_missing_event_fallback(tmp_path):
    scan = dict(row_position=np.arange(4,10),event_index=np.array([[0,0,1],[1,0,3]]),
                special_mask=np.zeros(10,bool),predictor_logprob=np.r_[-np.ones(5),np.nan])
    p,n = np.ones((2,5,2)),np.full((2,5,2),3.)
    np.savez_compressed(tmp_path/'event_5.npz',cut_hop_positive=p,cut_hop_negative=n,
                        explicit_contrast=np.array([False,False,False,True,False]))
    first = sample_scores(tmp_path,scan)
    np.testing.assert_array_equal(first['event_row'],[-1,-1,1,1,3])
    assert np.isnan(first['risk'][0,[0,1,3,4]]).all()
    assert first['risk'][0,2]==pytest.approx(.75)
    assert first['reason'][3]=='explicit_contrast'
    assert first['reason'][4]=='event_not_traced'
    np.savez_compressed(tmp_path/'labels.npz',labels=np.ones(5))
    second = sample_scores(tmp_path,scan)
    for key in first: np.testing.assert_array_equal(first[key],second[key])


@pytest.mark.parametrize('edge_kind',[0,1])
def test_one_physical_cut_edge_matches_native_internal_message_perturbation(tmp_path,edge_kind):
    from transformers import LlamaForCausalLM
    from experiments.reanchor_flow.message_dag.differential import rotate
    import torch.nn.functional as F
    path,trace,weights = capture_fixture(tmp_path,qk_scale=8)
    with NativeCache(path,weights) as cache:
        event = trace_with_cut(cache,np.array([[0,1,2]]),tmp_path/'cuts')[0]
        strength = np.where(event['cut_preview_index'][...,3]==edge_kind,np.abs(event['cut_preview_effect']),0)
        assert strength.max()>0
        target,slot = np.unravel_index(strength.argmax(),strength.shape)
        layer,head,source,kind = event['cut_preview_index'][target,slot]
        delta = torch.zeros(1,cache.rows,weights.config['hidden_size'])
        for l in range(int(layer)):
            op = DifferentialLayer(cache,l,3)
            same,cross,_ = op.attention_jvp(delta)
            post = delta+same+cross
            if l==0: post[0,2] += op.remote_seed(1,2,2)[0]
            delta = post+op.mlp_jvp(post)
        op = DifferentialLayer(cache,int(layer),3)
        z = rms_jvp(delta[0],op.x,op.w['input_norm'],op.eps)
        kv = head//(op.h//op.kv)
        a = next(a[:,target-b] for b,e,a in op.rows_attention() if b<=target<e)[head]
        if kind==0:
            code = a[op.rows[source]]*F.linear(z[source],op.w['value']).reshape(op.kv,op.hd)[kv]
        else:
            dk = F.linear(z[source],op.wk).reshape(op.kv,op.hd)[kv]
            dk = rotate(dk,op.cos[source],op.sin[source])
            ds = (op.q[head,target]*dk).sum()*op.scale
            code = a[op.rows[source]]*ds*(op.v[head,op.rows[source]]-a@op.v[head])
        message = op.w['output'][:,head*op.hd:(head+1)*op.hd]@code
    model = LlamaForCausalLM.from_pretrained(weights.directory).eval()
    model.config._attn_implementation = 'eager'
    position = int(trace['row_position'][target])
    ids = torch.tensor(trace['token_ids'])[None]
    def forward(strength):
        def perturb(module,args,output):
            out = output[0].clone();out[:,position] += strength*message
            return out,*output[1:]
        handle = model.model.layers[int(layer)].self_attn.register_forward_hook(perturb)
        try:
            with torch.no_grad(): logits = model(ids,use_cache=False).logits[0,position]
            return float(logits[event['positive_id'][target]]-logits[event['negative_id'][target]])
        finally: handle.remove()
    finite = (forward(.1)-forward(-.1))/.2
    predicted = float(event['cut_preview_effect'][target,slot])
    assert finite==pytest.approx(predicted,abs=2e-7,rel=.02)
    assert finite*predicted>0


def test_nonfinite_cut_cannot_be_published_as_a_closed_graph(tmp_path):
    path,_,weights = capture_fixture(tmp_path)
    with NativeCache(path,weights) as cache:
        recorder = CutRecorder(tmp_path/'edges.npz',cache,2)
        recorder.hop_positive[0,3,0] = np.nan
        try:
            event = dict(margin_response=np.zeros((3,3,cache.rows-1)))
            with pytest.raises(ValueError,match='nonfinite'):
                recorder.finish(event)
        finally: recorder.close()
    assert not (tmp_path/'edges.npz').exists()
    assert not (tmp_path/'edges.tmp.npz').exists()
