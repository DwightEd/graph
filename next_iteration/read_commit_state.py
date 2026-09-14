"""P4 causal delay between reading-event detection and uncertainty update."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def delayed_state(entropy,events,delay=8):
    h,e=np.asarray(entropy,float),np.asarray(events,bool)
    if h.ndim!=1 or h.shape!=e.shape or not np.isfinite(h).all() or delay<0:
        raise ValueError('invalid aligned state inputs')
    score=h.copy();state=None;until=-1
    for t in range(len(h)):
        if e[t]:
            state=h[t];until=t+delay
        if state is not None:
            if t<=until:state=max(state,h[t])
            score[t]=state
    return score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    from next_iteration.grounding_contrast import digest,write_json
    root=Path(args.predictions);out=Path(args.output)
    prior=json.loads((root/'manifest.json').read_text())
    if not prior['complete'] or prior.get('error') or prior['settings']['phase']!='development':
        raise ValueError('requires completed P3 development')
    if prior['completed_ids']!=prior['planned_ids']:
        raise ValueError('incomplete source roster')
    out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    protocol=Path(__file__).resolve().parent.parent/'docs/P4_READ_COMMIT_PROTOCOL.md'
    (out/'protocol.md').write_bytes(protocol.read_bytes())
    (out/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(complete=False,settings=dict(phase='development',**vars(args)),
        planned_ids=prior['planned_ids'],completed_ids=[],model_forwards=0,labels_used=False,
        code_sha256=digest(__file__),protocol_sha256=digest(protocol),
        input_prediction_manifest_sha256=digest(root/'manifest.json'),
        primary_score='read_commit_state',delay=8,
        development_selection='delay8 chosen after P3 development; not an independent replication',
        timing='strictly causal pre-token; no retrospective backfill')
    write_json(out/'manifest.json',manifest)
    for rid in prior['planned_ids']:
        path=root/f'response_{rid}.json'
        if digest(path)!=prior['output_sha256'][path.name]:raise ValueError('source prediction hash changed')
        r=json.loads(path.read_text());h=np.array(r['scores']['native_entropy']);n=len(h)
        event=np.array(r['event'],bool);entropy_event=np.array(r['entropy_event'],bool)
        periodic=(np.arange(n)>=17)&((np.arange(n)-17)%16==0)
        scores=dict(native_entropy=h,
            read_commit_state=delayed_state(h,event),
            entropy_commit_control=delayed_state(h,entropy_event),
            periodic_commit_control=delayed_state(h,periodic),
            rolling_max9=np.array([max(h[max(0,t-8):t+1]) for t in range(n)]))
        result={k:r[k] for k in ('id','source_id','task','split','response_sha256','prompt_sha256','offsets','token_ids')}
        result['scores']={k:v.tolist() for k,v in scores.items()}
        write_json(out/f'response_{rid}.json',result)
        manifest['completed_ids'].append(rid)
    manifest.update(complete=True,elapsed_seconds=time.time()-started,
        output_sha256={p.name:digest(p) for p in sorted(out.glob('response_*.json'))})
    write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(complete=True,responses=len(manifest['completed_ids']),elapsed=manifest['elapsed_seconds'])))


if __name__=='__main__':
    main()
