"""Core scientific invariants, explicit module switches, and all four CLI modes."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.charm_structure_audit.model import CHARM, degree, load_checkpoint
from experiments.charm_structure_audit.ablations import change_graph, topology_change, MODEL_ABLATIONS, GRAPH_ABLATIONS
from experiments.charm_structure_audit.data import save_scores, write_json, original_parts
from experiments.charm_structure_audit.positions import annotate, merge_spans, summarize
from experiments.charm_structure_audit.matching import match_answer
from experiments.charm_structure_audit.evaluate import metrics, analyze, paired_scores
from experiments.charm_structure_audit.main import main
from experiments.charm_structure_audit.train import calibrate


def graph_sample(identity='a', split='test'):
    prompt, count = 2, 32
    words = ['word']*count
    text = ' '.join(words)
    offsets = np.array([[5*i,5*i+4] for i in range(count)])
    edges, values = [], []
    for i in range(count):
        edges.append([0, prompt+i])
        values.append([.2]*4)
        if i%4:
            edges.append([prompt+i-1,prompt+i])
            values.append([.2]*4)
    edge = np.array(edges).T
    graph = dict(x=np.full((prompt+count,4),.1,np.float32), edge_index=edge,
                 edge_attr=np.array(values,np.float32), edge_mark=np.stack((edge[0]<prompt,edge[0]>=prompt),axis=1).astype(np.float32),
                 prompt_length=np.array(prompt), layers=np.array(2), heads=np.array(2))
    gold = (np.arange(count)>=12)&(np.arange(count)<16)
    sample = dict(id=identity,source_id=identity+'source',split=split,task='QA',generator='fixture',
        response=np.array(text), gold=gold, onset=np.arange(count)==12,
        spans=np.array([[12,16]]), offsets=offsets, token_ids=np.r_[99,99,np.tile(np.arange(4),8)],
        prompt_length=np.array(prompt), response_tokens=count, positives=int(gold.sum()))
    return graph,sample


def model(layers=2):
    torch.manual_seed(1)
    return CHARM(4,4,dict(hidden_dim=8,gnn_layers=layers,residual_mp=True),edge_chunk=3).eval()


def parent_formula(net, graph):
    """Direct algebra of parent model.py @96cc93f, independently written."""
    state = net.in_proj(torch.as_tensor(graph['x'])).relu()
    for layer in net.mp_layers:
        source,target = graph['edge_index']
        edge = torch.as_tensor(graph['edge_attr'])
        mark = torch.as_tensor(graph['edge_mark'])
        messages = layer.msg_mlp(torch.cat((state[source],edge,mark),dim=-1))
        total = torch.zeros_like(state).index_add(0,torch.as_tensor(target),messages)
        total = total / torch.as_tensor(degree(graph,net.normalization))[:,None]
        update = layer.up_mlp(torch.cat((state,total),dim=-1))
        state = (state+update).relu()
    return net.pred(state).view(-1)


@pytest.mark.parametrize('normalization',['in','out'])
def test_parent_forward_and_gradient(normalization):
    graph,_ = graph_sample()
    left=model()
    left.normalization=normalization
    right=copy.deepcopy(left)
    got=left(graph)
    expected=parent_formula(right,graph)
    torch.testing.assert_close(got,expected,atol=1e-7,rtol=1e-5)
    got.sum().backward()
    expected.sum().backward()
    for a,b in zip(left.parameters(),right.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-4)


@pytest.mark.parametrize('name', MODEL_ABLATIONS+GRAPH_ABLATIONS)
def test_each_ablation_runs_without_mutating_inputs(name):
    graph,_=graph_sample()
    before=copy.deepcopy(graph)
    view=change_graph(graph,name,3)
    score=model()(view,ablation=name,divisor=degree(graph))
    assert score.shape==(34,)
    assert torch.isfinite(score).all()
    for key in graph:
        np.testing.assert_array_equal(graph[key],before[key])


def test_no_graph_ignores_all_edges():
    graph,_=graph_sample()
    net=model()
    empty=change_graph(change_graph(graph,'no_history',0),'no_prompt',0)
    torch.testing.assert_close(net(graph,'no_graph'),net(empty,'no_graph'))


def test_no_graph_has_no_message_gradient():
    graph,_=graph_sample()
    net=model().train()
    net(graph,'no_graph').sum().backward()
    assert all(p.grad is None for layer in net.mp_layers for p in layer.msg_mlp.parameters())
    assert net.in_proj.weight.grad is not None


def test_no_edge_ignores_edge_values():
    graph,_=graph_sample()
    changed=dict(graph,edge_attr=graph['edge_attr']*7)
    net=model()
    torch.testing.assert_close(net(graph,'no_edge'),net(changed,'no_edge'))


def test_no_source_is_edge_only_message_not_no_messages():
    graph,_=graph_sample()
    layer=model().mp_layers[0]
    left=torch.randn(34,8)
    right=torch.randn(34,8)
    a=layer.aggregate(left,graph,5,'no_source')
    b=layer.aggregate(right,graph,5,'no_source')
    torch.testing.assert_close(a,b)
    assert a.abs().sum()>0


def test_no_relay_one_layer_matches_full():
    graph,_=graph_sample()
    net=model(1)
    torch.testing.assert_close(net(graph),net(graph,'no_relay'))


def test_no_relay_no_edges_matches_full():
    graph,_=graph_sample()
    empty=change_graph(change_graph(graph,'no_history',0),'no_prompt',0)
    net=model()
    torch.testing.assert_close(net(empty),net(empty,'no_relay'))


def test_prefix_invariance_for_in_normalization():
    graph,_=graph_sample()
    end=19
    keep=graph['edge_index'][1]<end
    cropped=dict(graph,x=graph['x'][:end],edge_index=graph['edge_index'][:,keep],
                 edge_attr=graph['edge_attr'][keep],edge_mark=graph['edge_mark'][keep])
    net=model()
    torch.testing.assert_close(net(graph)[:end],net(cropped))


def test_head_permutations_preserve_marginals():
    graph,_=graph_sample()
    for name in ('coupled_heads','independent_heads'):
        view=change_graph(graph,name,0)
        for query in np.unique(graph['edge_index'][1]):
            chosen=graph['edge_index'][1]==query
            np.testing.assert_allclose(view['edge_attr'][chosen].sum(axis=0),graph['edge_attr'][chosen].sum(axis=0))


def test_slot_change_not_topology_change():
    graph,_=graph_sample()
    view=dict(graph,edge_index=graph['edge_index'][:,::-1])
    change=topology_change(graph,view)
    assert change['changed_edge_slots']>0
    assert change['removed_fraction']==0


def test_overlap_merge_and_adjacent_keep():
    assert merge_spans([[1,3],[2,4],[4,6]])==[[1,4],[4,6]]


def test_rank_and_threshold_not_equivalent():
    row=metrics([1,0],[.7,.2],.8)
    assert row['auroc']==1 and row['recall']==0
    assert metrics([1],[.8],.8)['tp']==0


def test_calibration_ignores_nontext():
    sample=dict(score=np.array([.1,.2,.3,.99]),gold=np.array([0,0,0,0],bool),offsets=np.array([[0,1],[1,2],[2,3],[0,0]]))
    assert calibrate([sample],.05)['value']==.3


def fixture_files(tmp_path):
    root=tmp_path/'seed_0'
    prepared=tmp_path/'prepared'
    directory=root/'charm_in'
    test=directory/'test'
    net=model()
    hp=dict(hidden_dim=8,gnn_layers=2,residual_mp=True)
    directory.mkdir(parents=True)
    torch.save(dict(model_state=net.state_dict(),hp=hp),directory/'checkpoint.pt')
    records=[]
    parts={}
    test_samples=[]
    for name in ('fit','select','calibration','test'):
        split='test' if name=='test' else 'train'
        graph,sample=graph_sample(name,split)
        record={k:sample[k] for k in ('id','source_id','split','task','generator','response_tokens','positives')}
        record['graph']=str(prepared/'graphs'/split/(name+'.npz'))
        metadata={k:v for k,v in sample.items() if k not in graph}
        save_scores(record['graph'],**graph,**metadata,record_json=np.array(json.dumps(record)))
        records.append(record)
        parts[name]=[name]
        if name=='test':
            score=torch.sigmoid(net(graph)[2:]).detach().numpy()
            save_scores(test/'samples/test.npz',**metadata,score=score,record_json=np.array(json.dumps(record)))
            from experiments.charm_structure_audit.data import token_frame
            token_frame(sample,score,.5).to_csv(test/'tokens.csv',index=False)
            pd.DataFrame([dict(id=name,source_id=sample['source_id'],start=12,end=16)]).to_csv(test/'spans.csv',index=False)
            test_samples=[str(test/'samples/test.npz')]
    recipe=dict(variant='charm_in',seed=0,epochs=1,patience=1,hidden_dim=8,gnn_layers=2,batch_size=1,learning_rate=.001,fpr=.05,partitions=parts)
    write_json(prepared/'index.json',records)
    write_json(directory/'training.json',recipe)
    write_json(directory/'threshold.json',dict(value=.5))
    write_json(test/'predictions.json',test_samples)
    write_json(test/'prediction_settings.json',dict(variant='charm_in',seed=0))
    pair=dict(id='test',source_id='testsource',tier='cluster',error_start=12,normal_start=8,length=4)
    write_json(test/'cluster_audit/pairs.json',[pair])
    return root,prepared


def test_original_checkpoint_roundtrip(tmp_path):
    root,_=fixture_files(tmp_path)
    loaded,_=load_checkpoint(root/'charm_in/checkpoint.pt')
    graph,_=graph_sample()
    torch.testing.assert_close(model()(graph),loaded(graph))


def test_original_source_partition_guard(tmp_path):
    root,prepared=fixture_files(tmp_path)
    recipe=json.loads((root/'charm_in/training.json').read_text())
    recipe['partitions']['select']=['fit']
    with pytest.raises(ValueError,match='overlap'):
        original_parts(prepared,recipe)


def test_matching_is_score_blind_and_disjoint():
    graph,sample=graph_sample()
    first=match_answer(graph,sample)
    sample['score']=np.full(32,np.nan)
    second=match_answer(graph,sample)
    assert first==second
    assert len(first[0])==3
    assert all(p['normal_start']==8 for p in first[0])


def test_actual_report_cli_without_torch(tmp_path):
    root,prepared=fixture_files(tmp_path)
    code="import runpy,sys; sys.argv=['audit']+sys.argv[1:];runpy.run_module('experiments.charm_structure_audit.main',run_name='__main__');assert 'torch' not in sys.modules"
    command=[sys.executable,'-c',code,'--root',str(root),'--output',str(tmp_path/'report'),'--models','charm_in','--bootstrap','0']
    done=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    assert (tmp_path/'report/charm_in/positions.csv').exists()


def test_all_modes_and_input_immutability(tmp_path):
    root,prepared=fixture_files(tmp_path)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    common=['--root',str(root),'--prepared',str(prepared),'--device','cpu','--bootstrap','0','--ablations','no_graph','no_edge']
    for mode in ('report','match','ablate','train'):
        main(common+['--mode',mode,'--output',str(tmp_path/mode)])
    for p,(content,mtime) in before.items():
        assert p.read_bytes()==content and p.stat().st_mtime_ns==mtime
    assert (tmp_path/'ablate/paired_comparison.csv').exists()
    assert (tmp_path/'train/no_graph/threshold.json').exists()
    main(common+['--mode','ablate','--output',str(tmp_path/'ablate')])


def test_frozen_replay_mismatch_stops(tmp_path):
    root,prepared=fixture_files(tmp_path)
    path=root/'charm_in/test/samples/test.npz'
    with np.load(path) as saved:
        values={k:saved[k] for k in saved.files}
    values['score']=np.zeros(32)
    save_scores(path,**values)
    with pytest.raises(ValueError,match='replay'):
        main(['--mode','ablate','--root',str(root),'--prepared',str(prepared),'--device','cpu','--output',str(tmp_path/'bad')])


def test_simple_cluster_counterexample_has_poor_matched_ranking():
    graph,sample=graph_sample()
    score=np.full(32,.1)
    score[8:16]=.9
    from experiments.charm_structure_audit.data import token_frame
    table=token_frame(sample,score,.7)
    spans=pd.DataFrame([dict(id='a',start=12,end=16)])
    table=annotate(table,spans)
    pair=dict(id='a',source_id='asource',tier='cluster',error_start=12,normal_start=8,length=4)
    matched=paired_scores(table,[pair],.7)
    assert metrics(table.gold,table.score,.7)['auroc']>.9
    assert matched[matched.region=='all'].auroc.iloc[0]==.5


def test_position_denominators_and_short_spans():
    table=pd.DataFrame(dict(id=['a']*6,source_id=['s']*6,token=np.arange(6),
        text=['x']*6,gold=[0,1,1,0,1,0],score=[.1,.7,.8,.1,.9,.1],predicted=[0,0,1,0,1,0]))
    spans=pd.DataFrame([dict(id='a',start=1,end=3),dict(id='a',start=4,end=5)])
    table=annotate(table,spans)
    assert list(table.loc[table.gold==1,'offset'])==[0,1,0]
    rows,offsets,_=summarize(table)
    selected=rows[(rows.scope=='all')&rows.region.isin(['first','interior','last'])]
    assert selected.tokens.sum()==3
    assert selected.hits.sum()==2
    assert selected[selected.region=='interior'].tokens.iloc[0]==0


def test_mismatched_threshold_is_not_silently_used(tmp_path):
    table=pd.DataFrame(dict(id=['a','a'],source_id=['s','s'],token=[0,1],text=['x','y'],gold=[1,0],score=[.6,.2],predicted=[0,0]))
    spans=pd.DataFrame([dict(id='a',start=0,end=1)])
    with pytest.raises(ValueError,match='alarms'):
        analyze(table,spans,.5,[],tmp_path/'result',0)


def test_train_checkpointing_gradients_still_flow():
    graph,_=graph_sample()
    net=model().train()
    net(graph).sum().backward()
    assert all(p.grad is not None for p in net.parameters())
    assert net.mp_layers[0].msg_mlp[0].weight.grad.abs().sum()>0
