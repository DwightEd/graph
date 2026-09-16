"""Regression and actual tiny-CPU runs for the CHARM attribution audit."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.unsupervised_token_graph.data import ResponseCache
from experiments.charm_structure_audit.data import label_arrays, load_graph, partitions, prepare
from experiments.charm_structure_audit.experiment import audit, fit, load_predictions, compare_models
from experiments.charm_structure_audit.graph import build_graph, degree, prefix_graph, transform_graph
from experiments.charm_structure_audit.metrics import (boundary_summary, classification, group_report,
                                                     report, runs, span_rows, threshold_at_fpr)
from experiments.charm_structure_audit.model import CHARM, load_checkpoint, smooth_scores


def canonical(path, attention, p, **extra):
    l, h, n, _ = attention.shape
    pointer, columns, values = [0], [], []
    for layer in range(l):
        for head in range(h):
            for q in range(p, n):
                keys = np.flatnonzero(attention[layer, head, q, :q])
                columns.extend(keys); values.extend(attention[layer, head, q, keys]); pointer.append(len(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, token_ids=np.arange(n), response_idx=p,
                        attention_diagonal=np.diagonal(attention, axis1=2, axis2=3),
                        response_row_ptr=np.asarray(pointer), response_column_indices=np.asarray(columns, dtype=np.int64),
                        response_values=np.asarray(values, dtype=attention.dtype), **extra)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_canonical_union_matches_upstream_dense_definition(tmp_path, dtype):
    rng = np.random.default_rng(4)
    a = np.tril(rng.random((2, 2, 7, 7))).astype(dtype)
    a /= a.sum(-1, keepdims=True)
    a[0, 0, 3, 1] = .05
    path = tmp_path / 'sample.npz'
    canonical(path, a, 2, labels=np.array([{'never_read': True}], dtype=object))
    record = ResponseCache().load(path)
    graph = build_graph(record, .05)
    expected = {}
    for i in range(2, 7):
        for j in range(i):
            attr = a[:, :, i, j].reshape(-1).astype(np.float32)
            attr[attr <= .05] = 0
            if np.any(attr):
                expected[j, i] = attr
    actual = {tuple(edge): attr for edge, attr in zip(graph['edge_index'].T, graph['edge_attr'])}
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key])
    np.testing.assert_allclose(graph['x'], np.diagonal(a, axis1=2, axis2=3).reshape(4, 7).T)
    assert len(graph['x']) == 7  # No isolated node deletion.


def toy_graph():
    rng = np.random.default_rng(5)
    source, target = [], []
    for q in range(2, 10):
        for j in sorted({0, max(1, q-3), q-1}):
            source.append(j); target.append(q)
    source, target = np.asarray(source), np.asarray(target)
    return dict(x=rng.random((10, 4)).astype('float32'), edge_index=np.stack((source, target)),
                edge_attr=rng.random((len(source), 4)).astype('float32'),
                edge_mark=np.stack((source < 2, source >= 2), 1).astype('float32'), prompt_length=np.asarray(2))


@pytest.mark.parametrize('kind', ['local_in', 'rewire_in'])
def test_controls_preserve_target_counts_and_attributes_but_change_endpoints(kind):
    g = toy_graph()
    changed, stats = transform_graph(g, kind, seed=3)
    np.testing.assert_array_equal(g['edge_index'][1], changed['edge_index'][1])
    np.testing.assert_array_equal(g['edge_attr'], changed['edge_attr'])
    np.testing.assert_array_equal(g['edge_mark'], changed['edge_mark'])
    rp = g['edge_index'][0] < 2
    np.testing.assert_array_equal(g['edge_index'][:, rp], changed['edge_index'][:, rp])
    assert stats['changed_edges'] > 0
    assert np.all(changed['edge_index'][0] < changed['edge_index'][1])
    assert len(set(map(tuple, changed['edge_index'].T))) == changed['edge_index'].shape[1]
    if kind == 'rewire_in':
        bins = lambda edges: np.searchsorted([1, 4, 16, 64], edges[1]-edges[0], side='left')
        np.testing.assert_array_equal(bins(g['edge_index']), bins(changed['edge_index']))


@pytest.mark.parametrize('kind', ['real', 'node_only', 'no_rp', 'no_rr', 'zero_edge', 'zero_mark', 'shuffle_nodes', 'rewire_in'])
def test_controls_do_not_mutate_original_or_read_labels(kind):
    g = toy_graph(); before = {k: v.copy() for k, v in g.items()}
    g['gold'] = np.zeros(8, bool)
    a, _ = transform_graph(g, kind, seed=2)
    g['gold'] = np.ones(8, bool)
    b, _ = transform_graph(g, kind, seed=2)
    for name in ('x', 'edge_index', 'edge_attr', 'edge_mark'):
        np.testing.assert_array_equal(a[name], b[name]); np.testing.assert_array_equal(g[name], before[name])


def reference_forward(model, graph):
    """Literal upstream message/update algebra without the PyG dispatcher."""
    x = torch.from_numpy(graph['x'])
    source, target = torch.from_numpy(graph['edge_index'])
    attr, mark = torch.from_numpy(graph['edge_attr']), torch.from_numpy(graph['edge_mark'])
    h = model.in_proj(x).relu()
    divisor = torch.from_numpy(degree(graph, model.normalization))
    for layer in model.mp_layers:
        messages = layer.msg_mlp(torch.cat((h[source], attr, mark), -1))
        summed = torch.zeros_like(h).index_add(0, target, messages)
        h = (h + layer.up_mlp(torch.cat((h, summed / divisor[:, None]), -1))).relu()
    return model.pred(h).view(-1)


@pytest.mark.parametrize('norm', ['in', 'out'])
def test_chunked_model_matches_reference_logits_and_gradients(norm):
    torch.set_num_threads(1); torch.manual_seed(9)
    model = CHARM(4, 4, dict(hidden_dim=8, gnn_layers=3), norm, edge_chunk=2)
    model.pred[2].p = 0
    g = toy_graph()
    model.train()
    actual = model(g)
    expected = reference_forward(model, g)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(); reference_forward(model, g).sum().backward()
    for k, parameter in model.named_parameters():
        torch.testing.assert_close(gradients[k], parameter.grad)


def test_original_checkpoint_keys_load_strictly(tmp_path):
    torch.manual_seed(2)
    hp = dict(hidden_dim=8, gnn_layers=3, residual_mp=True)
    parent = CHARM(4, 4, hp, 'out').eval()
    path = tmp_path / 'grid_best_model.pt'
    torch.save(dict(model_state=parent.state_dict(), best_hp=hp), path)
    restored, _ = load_checkpoint(path)
    torch.testing.assert_close(parent(toy_graph()), restored(toy_graph()))


def test_indegree_is_prefix_invariant_outdegree_is_not():
    torch.manual_seed(6)
    g = toy_graph()
    model = CHARM(4, 4, dict(hidden_dim=8, gnn_layers=3), 'in').eval()
    prefix = prefix_graph(g, 6)
    torch.testing.assert_close(model(g)[:6], model(prefix))
    model.normalization = 'out'
    assert torch.max(abs(model(g)[:6] - model(prefix))) > 1e-5


def sample(y=(False, True, True, True, True, True, False), score=None):
    y = np.asarray(y, bool)
    if score is None:
        score = np.array([.1, .1, .9, .9, .9, .9, .1])
    spans = np.asarray(runs(y), int).reshape(-1, 2)
    onset = np.zeros(len(y), bool)
    for a, b in spans:
        onset[a] = True
    value = dict(id='1', source_id='s1', task='QA', generator='fixture',
                 gold=y, onset=onset, spans=spans, score=np.asarray(score, float),
                 offsets=np.stack((np.arange(len(y)), np.arange(1, len(y)+1)), 1),
                 response='abcdefg'[:len(y)])
    for name in ('in_rp','in_rr','out_degree','local_rr_fraction','self_attention_mean'):
        value['structure_' + name] = np.arange(len(y), dtype=float)
    return value


def test_missed_first_but_high_continuation_is_explicit():
    s = sample(); metrics, rows = boundary_summary([s], .5)
    first = metrics['first_error_spans']
    assert first['onset_recall'] == 0
    assert first['mean_token_coverage'] == pytest.approx(.8)
    assert first['missed_onset_later80_count'] == 1
    assert first['missed_onset_later_all_count'] == 1
    assert first['missed_onset_later_all_given_missed'] == 1
    assert rows[0]['delay_if_detected'] == 1
    assert rows[0]['continuation_coverage'] == 1


def test_zero_token_first_error_is_included():
    s = sample([True, True, False], [.1, .9, .1])
    metrics, _ = group_report([s], .5)
    assert metrics['token_scopes']['first_error_vs_normal']['positives'] == 1
    assert metrics['token_scopes']['first_error_vs_normal']['recall'] == 0


def test_end_spillover_is_a_false_positive_not_a_span_hit():
    s = sample([False, True, True, False, False], [.1, .9, .9, .9, .9])
    metrics, _ = boundary_summary([s], .5)
    assert metrics['post_end_5_normal']['fpr'] == 1
    assert metrics['annotation_spans']['mean_token_coverage'] == 1
    assert classification(s['gold'], s['score'], .5)['precision'] == .5


def test_singleton_is_excluded_from_later80_denominator():
    s = sample([False, True, False], [.1, .1, .1])
    metrics, _ = boundary_summary([s], .5)
    assert metrics['first_error_spans']['missed_onset_later80_denominator'] == 0
    assert metrics['first_error_spans']['mean_delay_detected_only'] is None
    assert metrics['first_error_spans']['undetected'] == 1


def test_all_auc_decomposition_uses_identical_normal_negatives():
    m, _ = group_report([sample()], .5)
    assert m['auc_decomposition']['reconstructed_all_auroc'] == pytest.approx(m['token_scopes']['all_error']['auroc'])


def test_ties_and_single_class_are_not_fabricated_success():
    value = classification([0,1,0,1], [.4]*4, .4)
    assert value['auroc'] == .5 and value['ap'] == .5
    assert value['recall'] == 0
    assert classification([0,0], [.8,.9], .5)['auroc'] is None


def test_threshold_uses_calibration_normals_and_strict_ties():
    s = sample([False, False, True, True], [.2,.2,.9,.8])
    threshold = threshold_at_fpr([s], .05)
    assert threshold['value'] == .2 and threshold['achieved_fpr'] == 0
    s['score'][2:] = [.01,.02]
    assert threshold_at_fpr([s], .05) == threshold


def test_scores_are_never_expanded_using_gold_boundaries():
    s = sample(); before = s['score'].copy()
    boundary_summary([s], .5)
    np.testing.assert_array_equal(s['score'], before)
    assert classification(s['gold'], s['score'], .5)['fn'] == 1


def test_span_mapping_retains_adjacent_annotations_and_zero_offsets():
    y, onset, spans = label_arrays(np.array([[0,0],[0,2],[2,4]]), [dict(start=0,end=2),dict(start=2,end=4)])
    assert y.tolist() == [False,True,True]
    assert onset.tolist() == [False,True,True]
    assert spans.tolist() == [[1,2],[2,3]]


def test_ewma_is_past_only():
    a, b = smooth_scores([.1,.8,.2]), smooth_scores([.1,.8,.2,.99])
    np.testing.assert_allclose(a, b[:3])


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'RAGTruth/attention/llama31_8b'
    dataset = tmp_path / 'RAGTruth/dataset'; dataset.mkdir(parents=True)
    annotations, sources = [], []
    rng = np.random.default_rng(8)
    for i in range(9):
        split = 'train' if i < 6 else 'test'
        response = 'abcdef'
        a = np.tril(rng.random((2,2,8,8))).astype(np.float16); a /= a.sum(-1, keepdims=True)
        canonical(root / split / 'attention' / f'attention_{i}.npz', a, 2,
                  id=str(i), source_id='s'+str(i), split=split, offsets=np.array([[j,j+1] for j in range(6)]),
                  response=response, task='QA', generator='fixture')
        annotations.append(dict(id=str(i), source_id='s'+str(i), response=response, split=split, model='fixture',
                                labels=[dict(start=2,end=5)]))
        sources.append(dict(source_id='s'+str(i), task_type='QA'))
    labels=dataset/'response.jsonl'; labels.write_text('\n'.join(map(json.dumps, annotations)))
    (dataset/'source_info.jsonl').write_text('\n'.join(map(json.dumps, sources)))
    output=tmp_path/'prepared'
    rows=prepare(root, labels, output, tasks=['QA'], generator='fixture')
    return root, labels, output, rows


def test_existing_npz_readers_source_id_and_resume(prepared):
    root, labels, out, rows=prepared
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'graphs').rglob('*.npz')}
    assert prepare(root, labels, out, tasks=['QA'], generator='fixture') == rows
    assert {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before} == before
    assert all(r['task'] == 'QA' for r in rows)


def test_partitions_are_source_disjoint(prepared):
    *_, rows=prepared
    part=partitions(rows)
    sets=[{r['source_id'] for r in v} for v in part.values()]
    for i in range(len(sets)):
        for j in range(i): assert not sets[i]&sets[j]


def test_real_cpu_training_checkpoint_audit_report(prepared, tmp_path):
    *_, rows=prepared; torch.set_num_threads(1)
    part=partitions(rows); outputs={}
    for variant in ('charm_in','node_only'):
        out=tmp_path/variant
        checkpoint=fit(part,out,variant,epochs=1,hidden_dim=8,layers=2,batch_size=2,edge_chunk=3)
        threshold=json.loads((out/'threshold.json').read_text())
        samples=audit(checkpoint,part['test'],out/'test',variant,threshold=threshold,edge_chunk=3,prefix_sites=2)
        result=report(samples,out/'test',threshold,dict(test='tiny_actual_training'),bootstrap=2)
        assert result['groups']['ALL']['responses'] == 3
        assert (out/'test/tokens.csv').exists() and (out/'test/gallery.html').exists()
        assert 'score_prefix' in result['interventions']
        assert result['interventions']['score_prefix']['max_absolute_score_change'] < 1e-5
        assert np.load(out/'test/samples/6.npz')['embedding'].shape == (6,8)
        outputs[variant]=out/'test'
    compared=compare_models(outputs,tmp_path/'comparison.json',bootstrap=2)
    assert 'node_only' in compared


def test_frozen_checkpoint_run_does_not_train_or_modify(prepared,tmp_path,monkeypatch):
    *_, rows=prepared
    model=CHARM(4,4,dict(hidden_dim=8,gnn_layers=2),'out')
    checkpoint=tmp_path/'original.pt'; torch.save(dict(model_state=model.state_dict(),best_hp=dict(hidden_dim=8,gnn_layers=2)),checkpoint)
    before=checkpoint.read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('audit must not train')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    samples=audit(checkpoint,[r for r in rows if r['split']=='test'],tmp_path/'audit',prefix_sites=1,
                  threshold=dict(value=.5,origin='predeclared'))
    assert len(samples)==3 and checkpoint.read_bytes()==before
    assert any(np.max(abs(s['score_no_rr']-s['score']))>0 for s in samples)
    assert any(np.nanmax(abs(s['score_prefix']-s['score']))>1e-6 for s in samples)
    paths=list((tmp_path/'audit/samples').glob('*.npz'))
    before={p:p.stat().st_mtime_ns for p in paths}
    audit(checkpoint,[r for r in rows if r['split']=='test'],tmp_path/'audit',prefix_sites=1,threshold=dict(value=.5,origin='predeclared'))
    assert {p:p.stat().st_mtime_ns for p in paths}==before


def test_report_html_escapes_text(tmp_path):
    s=sample();s['response']='<abc>&x'
    report([s],tmp_path,dict(value=.5),{},bootstrap=0)
    text=(tmp_path/'gallery.html').read_text()
    assert '&lt;' in text and '&gt;' in text and '&amp;' in text


def test_cli_existing_checkpoint_and_readonly_report(prepared,tmp_path):
    *_, data,rows=prepared
    hp=dict(hidden_dim=8,gnn_layers=2)
    model=CHARM(4,4,hp,'out')
    checkpoint=tmp_path/'parent.pt';torch.save(dict(model_state=model.state_dict(),best_hp=hp),checkpoint)
    repo=Path(__file__).resolve().parents[1]
    output=tmp_path/'cli'
    result=subprocess.run([sys.executable,'-m','experiments.charm_structure_audit.run','audit',
                           '--prepared',str(data),'--output',str(output),'--tasks','QA','--checkpoint',str(checkpoint),
                           '--device','cpu','--bootstrap','0','--prefix-sites','1'],cwd=repo,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert 'first_error_recall' in result.stdout
    pred=output/'QA/external_checkpoint'
    (pred/'predictions.json').unlink()
    result=subprocess.run([sys.executable,'-m','experiments.charm_structure_audit.run','report',
                           '--predictions',str(pred),'--output',str(tmp_path/'preview'),'--bootstrap','0','--completed-only'],
                          cwd=repo,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'preview/report.json').exists()


def test_checkpoint_accepts_numeric_numpy_metrics_without_unsafe_pickle(tmp_path):
    model=CHARM(4,4,dict(hidden_dim=8,gnn_layers=2),'out').eval()
    path=tmp_path/'grid_best.pt'
    torch.save(dict(model_state=model.state_dict(), best_hp=dict(hidden_dim=8,gnn_layers=2),
                    test_metrics=dict(auroc=np.float64(.8), aupr=np.float32(.2))), path)
    loaded,_=load_checkpoint(path)
    torch.testing.assert_close(model(toy_graph()),loaded(toy_graph()))


def test_model_never_reads_gold_labels():
    model=CHARM(4,4,dict(hidden_dim=8,gnn_layers=2),'in').eval()
    graph=toy_graph(); graph['gold']=np.zeros(8,bool)
    a=model(graph)
    graph['gold']=np.ones(8,bool)
    torch.testing.assert_close(a,model(graph))


def test_manifest_floor_above_threshold_is_rejected(prepared,tmp_path):
    root,labels,_,_=prepared
    (root/'manifest.json').write_text(json.dumps(dict(attention_floor=.1)))
    with pytest.raises(ValueError,match='floor exceeds'):
        prepare(root,labels,tmp_path/'new',tasks=['QA'],generator='fixture',tau=.05)


def test_label_overlap_does_not_inflate_auc_positive_count():
    offsets=np.array([[i,i+1] for i in range(6)])
    y,onset,spans=label_arrays(offsets,[dict(start=1,end=4),dict(start=2,end=5)])
    assert y.sum()==4 and onset.sum()==2 and len(spans)==2


def test_empty_graph_retains_all_token_predictions(tmp_path):
    a=np.zeros((1,1,5,5),np.float16)
    a[...,np.arange(5),np.arange(5)]=1
    path=tmp_path/'empty.npz';canonical(path,a,2)
    graph=build_graph(ResponseCache().load(path))
    assert graph['edge_index'].shape==(2,0)
    model=CHARM(1,1,dict(hidden_dim=8,gnn_layers=2),'out').eval()
    assert model(graph).shape==(5,)


@pytest.mark.parametrize('phase,with_checkpoint',[('prepare',False),('all',False),('all',True),('audit',True)])
def test_shell_forwarding_and_mode_selection(tmp_path,phase,with_checkpoint):
    import os
    repo=Path(__file__).resolve().parents[1]
    mock=tmp_path/'python mock'
    log=tmp_path/'calls.jsonl'
    mock.write_text('#!/usr/bin/env python\nimport json,sys,os\nwith open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    mock.chmod(0o755)
    env=os.environ.copy()
    env.update(PY=str(mock),CALL_LOG=str(log),PHASE=phase,CHECKPOINT=str(tmp_path/'old model.pt') if with_checkpoint else '',
               TASKS='QA',OUTPUT=str(tmp_path/'out dir'),DEVICE='cpu',ATTENTION_ROOT=str(tmp_path/'cache root'),
               ANNOTATIONS=str(tmp_path/'response.jsonl'),TOKENIZER=str(tmp_path/'tokenizer'),PREPARED=str(tmp_path/'prepared'),
               BOOTSTRAP='0',VARIANTS='charm_in node_only',SEEDS='0')
    result=subprocess.run(['bash','experiments/charm_structure_audit/run_all.sh'],cwd=repo,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    calls=[json.loads(x) for x in log.read_text().splitlines()]
    assert calls[0][:2]==['-m','pytest']
    assert not any('fit' in c for c in calls) if with_checkpoint else True
    if phase in ('all','prepare'):
        preparation=next(c for c in calls if 'prepare' in c)
        pos=preparation.index('--splits')
        assert preparation[pos+1]=='test' if with_checkpoint else preparation[pos+1:pos+3]==['train','test']
    if phase in ('all','audit'):
        execution=calls[-1]
        assert ('audit' in execution) == with_checkpoint
