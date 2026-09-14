import numpy as np
import pytest
from revisit_evaluate import cover, random_events, selector_counts


def test_lead_window_does_not_backfill():
    event = np.array([0,0,1,0,0,0],bool)
    np.testing.assert_equal(cover(event,2),[0,0,1,1,1,0])
    np.testing.assert_equal(cover(event,0),event)


def test_matching_preserves_block_counts_and_cold_start():
    event = np.zeros(100,bool)
    event[[18,21,35,68,99]]=True
    for mode in ('uniform','position_matched'):
        rng = np.random.default_rng(5)
        for _ in range(20):
            out = random_events(event,rng,mode)
            assert out.sum()==event.sum()
            assert not out[:17].any()
            if mode=='position_matched':
                for start in range(0,100,32):
                    assert out[start:start+32].sum()==event[start:start+32].sum()


def test_total_onsets_not_only_eligible():
    event = np.zeros(30,bool);event[20]=True
    onset = np.zeros(30,int);onset[[0,21]]=1
    result=selector_counts(event,onset,onset,4)
    assert result['onsets']==2 and result['onset_hits']==1
    assert result['eligible_onsets']==1
    assert result['covered_tokens']==5


def test_no_events_valid():
    event=np.zeros(3,bool)
    result=selector_counts(event,np.zeros(3),np.zeros(3),4)
    assert result['events']==result['onset_hits']==0


def test_evaluator_end_to_end_serialization(tmp_path,monkeypatch):
    import hashlib,json,sys,importlib.util
    from pathlib import Path
    from revisit_evaluate import main,digest
    helper=Path(__file__).resolve().parents[1]/'grounding-work/grounding_contrast_evaluate.py'
    if not helper.exists():
        helper=Path(__file__).resolve().parents[1]/'next_iteration/grounding_contrast_evaluate.py'
    spec=importlib.util.spec_from_file_location('next_iteration.grounding_contrast_evaluate',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,'next_iteration.grounding_contrast_evaluate',module)
    root=tmp_path/'predictions';root.mkdir()
    response='x'*40
    e=np.zeros(40,bool);e[[18,25]]=True
    h=np.arange(40,dtype=float)
    record=dict(id='one',source_id='source',task='QA',response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        offsets=[[i,i+1] for i in range(40)],event=e.tolist(),entropy_event=e.tolist(),mean_event=e.tolist(),
        scores={'native_entropy':h.tolist(),'revisit_magnitude':h.tolist()})
    graph=root/'attention_one.npz';graph.write_bytes(b'fixture-only-hash-not-native-attention')
    record['graph_sha256']=digest(graph)
    pred=root/'response_one.json';pred.write_text(json.dumps(record))
    (root/'manifest.json').write_text(json.dumps(dict(complete=True,settings={'phase':'development'},
        planned_ids=['one'],completed_ids=['one'],output_sha256={pred.name:digest(pred)},graph_sha256={graph.name:digest(graph)})))
    annotations=tmp_path/'labels.jsonl'
    annotations.write_text(json.dumps(dict(id='one',source_id='source',response=response,labels=[{'start':19,'end':21}])))
    output=tmp_path/'audit.json'
    monkeypatch.setattr(sys,'argv',['audit','--predictions',str(root),'--annotations',str(annotations),'--output',str(output)])
    main()
    report=json.loads(output.read_text())
    assert report['onsets']==1
    assert report['per_response'][0]['matched']['4']['onset_event_positions']==[18]
