"""Numerical contracts and counterexamples for the message DAG model."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.tests.test_message_lineage import capture_fixture
from experiments.reanchor_flow.message_lineage import CheckpointWeights
from experiments.reanchor_flow.attention_audit import AuditConfig, capture_audit
from experiments.reanchor_flow.message_dag.cache import NativeCache, source_partition
from experiments.reanchor_flow.message_dag.graph import Tape, prepare, trace_target, trace_targets
from experiments.reanchor_flow.message_dag.operators import LayerOperator

torch.set_num_threads(2)


def compute(path, weights, target=7, **kw):
    with NativeCache(path,weights) as cache, TemporaryDirectory() as directory:
        tape=Tape(cache,directory)
        try:
            prepare(cache,tape,source_chunk=kw.pop('source_chunk',2))
            return trace_target(cache,tape,target,**kw)
        finally: tape.close()


@pytest.mark.parametrize('rule',['symmetric','up'])
def test_native_operators_and_adjoints_are_bilinear_with_gqa(tmp_path,rule):
    path,trace,weights=capture_fixture(tmp_path)
    with NativeCache(path,weights) as cache:
        op=LayerOperator(cache,1,rule,chunk=3)
        torch.manual_seed(40)
        x,y=torch.randn(2,cache.rows,op.d),torch.randn(2,cache.rows,op.d)
        empty=torch.zeros(2,len(trace['token_ids']))
        torch.testing.assert_close((op.attention(x,empty)*y).sum(),(x*op.attention_adjoint(y)).sum(),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close((op.mlp(x)*y).sum(),(x*op.mlp_adjoint(y)).sum(),atol=2e-6,rtol=2e-5)
        assert not op.values(x)[...,0,:].any() # P-1 is a prompt boundary, not a response carrier


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_source_sink_balance_and_display_budget_cannot_change_the_graph(tmp_path,dtype):
    path,trace,w=capture_fixture(tmp_path,dtype)
    a=compute(path,w,edge_budget=2)
    b=compute(path,w,edge_budget=30,source_chunk=1,query_chunk=3)
    np.testing.assert_allclose(a['source_output'].sum(),trace['residual_margin'][-1,2],atol=3e-7)
    np.testing.assert_allclose(a['root_credit'],a['source_output'],atol=2e-7,rtol=2e-5)
    assert a['balance_error'].max()<2e-5
    assert np.abs(a['head_relay'][:,1:]).max()>1e-8
    for key in ('source_output','head_relay','node_input','carrier_absolute'):
        np.testing.assert_allclose(a[key],b[key],atol=3e-7,rtol=2e-5)
    assert len(a['edge_index'])<len(b['edge_index'])
    assert not np.isin(b['edge_index'][:,2:], [0,4,8]).any()
    assert (b['edge_index'][:,2]<b['edge_index'][:,3]).all()
    # The same target cannot receive flow from its future, even though the
    # captured full sequence contains future tokens and their native states.
    assert not b['node_input'][...,trace['row_position']>6].any()


def test_material_units_remain_separate_and_add_back_to_the_merged_source(tmp_path):
    path,trace,w=capture_fixture(tmp_path)
    merged=compute(path,w)
    trace['source_unit_id'][3]=1
    np.savez_compressed(path,**trace)
    split=compute(path,w)
    assert split['source_names'][:2].tolist()==['material:0','material:1']
    for key in ('source_output','node_input','head_relay','root_credit'):
        np.testing.assert_allclose(split[key][:2].sum(0),merged[key][0],atol=3e-7,rtol=2e-5)


def test_target_blocks_share_layers_without_mixing_output_targets(tmp_path,monkeypatch):
    from experiments.reanchor_flow.message_dag import graph as module
    path,trace,w=capture_fixture(tmp_path)
    expected=[compute(path,w,t) for t in (6,7)]
    with NativeCache(path,w) as cache,TemporaryDirectory() as directory:
        tape=Tape(cache,directory);prepare(cache,tape)
        loads=[];constructor=module.LayerOperator
        def counted(*a,**kw): loads.append(a[1]);return constructor(*a,**kw)
        monkeypatch.setattr(module,'LayerOperator',counted)
        try: actual=trace_targets(cache,tape,[6,7])
        finally: tape.close()
    assert loads==[2,1,0]
    for a,b in zip(actual,expected):
        assert int(a['target'])==int(b['target'])
        for key in ('source_output','root_credit','node_input','head_relay','edge_source_credit'):
            np.testing.assert_allclose(a[key],b[key],atol=2e-7,rtol=1e-5)


def test_identical_attention_can_transmit_or_block_material_in_the_full_dag(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM
    outputs,attentions=[],[]
    for transform in (False,True):
        directory=tmp_path/str(transform);directory.mkdir()
        cfg=LlamaConfig(vocab_size=12,hidden_size=2,intermediate_size=4,num_hidden_layers=2,
                        num_attention_heads=1,num_key_value_heads=1)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
            model.model.embed_tokens.weight[:4,0]=1
            model.model.embed_tokens.weight[4:,1]=1
            model.model.norm.weight.fill_(1)
            model.lm_head.weight[7,1]=1
            for i,layer in enumerate(model.model.layers):
                layer.input_layernorm.weight.fill_(1);layer.post_attention_layernorm.weight.fill_(1)
                layer.self_attn.o_proj.weight.copy_(torch.eye(2))
                layer.self_attn.v_proj.weight[i,i]=1
            mlp=model.model.layers[0].mlp
            mlp.gate_proj.weight[0,1]=1;mlp.up_proj.weight[0,0]=1
            mlp.down_proj.weight[1,0]=float(transform)
        ids=np.arange(1,9);material=np.arange(8)==1;special=np.arange(8)==0
        path=directory/'sample.npz'
        trace=capture_audit(model,ids,3,material,special,np.where(material,0,-1),path,AuditConfig(query_chunk=3))
        trace['settings']=np.array(json.dumps({'local_window':10}))
        np.savez_compressed(path,**trace)
        model.save_pretrained(directory/'model')
        outputs.append(compute(path,CheckpointWeights(directory/'model'),target=6))
        with np.load(path.with_suffix('.history.npz')) as f: attentions.append([f['L0'],f['L1']])
    np.testing.assert_array_equal(attentions[0],attentions[1])
    assert outputs[0]['head_relay'][0,1,0]==0
    assert abs(outputs[1]['head_relay'][0,1,0])>1e-4


def test_pipeline_resume_and_portable_evaluation(tmp_path,monkeypatch):
    from experiments.reanchor_flow.message_dag import run as entry
    path,trace,w=capture_fixture(tmp_path)
    native=tmp_path/'native';native.mkdir()
    entries=[]
    for split in ('train','test'):
        for task in ('QA','Summary','Data2txt'):
            for cls in ('N','H'):
                relative=Path(split)/task/f'{cls}.npz';dest=native/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                for suffix in ('.npz','.history.npz','.qk.npz','.states.npz'): shutil.copyfile(path.with_suffix(suffix),dest.with_suffix(suffix))
                labels=np.zeros(8,int)
                if cls=='H': labels[[1,2,5]]=1
                np.savez_compressed(dest.with_suffix('.labels.npz'),labels=labels)
                entries.append(dict(split=split,task_type=task,sample_id=cls,source_id=f'{split}-{task}-{cls}',path=relative.as_posix(),response_tokens=8))
    original=dict(audit_schema=3,settings={'save_states':True,'model':str(w.directory)},samples=entries)
    (native/'index.json').write_text(json.dumps(original));before=(native/'index.json').read_bytes()
    argv=['--audit',str(native),'--device','cpu','--samples-per-group','0','--targets-per-sample','2','--bootstrap','0','--edge-budget','3']
    plan=entry.run(entry.parser().parse_args([*argv,'--plan-only']))
    assert len(plan['samples'])==12 and not (native/'message_dag_v2').exists()
    result=entry.run(entry.parser().parse_args(argv));output=native/'message_dag_v2'
    assert result['completed_targets']==24 and len(result['cohorts'])==6
    assert (native/'index.json').read_bytes()==before
    assert result['cohorts']['test/QA']['hallucinated']>0
    assert (output/'gallery.html').exists()
    def forbidden(*a,**kw): pytest.fail('completed targets must not replay source propagation')
    monkeypatch.setattr(entry,'prepare',forbidden)
    resumed=entry.run(entry.parser().parse_args(argv))
    assert resumed['completed_targets']==24
    # The report is portable: it needs only derived graphs, metadata and labels.
    for e in entries:
        for suffix in ('.npz','.history.npz','.qk.npz','.states.npz','.labels.npz'):
            (native/e['path']).with_suffix(suffix).unlink()
    shutil.rmtree(w.directory)
    last=next(output.rglob('target_*.npz'));last.rename(last.with_suffix('.tmp.npz'))
    partial=entry.run(entry.parser().parse_args(['--phase','evaluate','--output',str(output),'--bootstrap','0']))
    assert partial['completed_targets']==23 and partial['partial']
    assert sum(c['missing_graph'] for c in partial['cohorts'].values())==1
    # Interruption can also precede a sample's first metadata/target write.
    untouched=next(p for p in output.rglob('meta.npz') if p.parent!=last.parent)
    untouched.unlink()
    partial=entry.run(entry.parser().parse_args(['--phase','evaluate','--output',str(output),'--bootstrap','0']))
    assert partial['completed_targets']==21 and len(partial['missing_samples'])==1
