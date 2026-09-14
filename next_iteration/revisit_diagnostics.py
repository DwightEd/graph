"""Post-hoc P3 destination-gain and P4 leave-one-source-out diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--p3',required=True)
    p.add_argument('--p4-localization',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.p3)
    m=json.loads((root/'manifest.json').read_text())
    a=json.loads((root/'event_audit_v2.json').read_text())
    if not m['complete'] or sha(root/'manifest.json')!=a['manifest_sha256']:raise ValueError('audit manifest mismatch')
    events=[]
    for r in a['per_response']:
        path=root/f"attention_{r['id']}.npz"
        if sha(path)!=m['graph_sha256'][path.name]:raise ValueError('graph hash mismatch')
        with np.load(path) as g:
            attention=g['mean_attention'];P=len(g['source_mask']);source=g['source_mask']
            allowed=~g['special_mask'];allowed[:P]&=source
            onset={c['token_index'] for c in r['onset_cases']}
            for t in np.flatnonzero(g['event']):
                q=P+t-1
                current=attention[t,:q-1].astype(float)*allowed[:q-1]
                previous=attention[t-1,:q-1].astype(float)*allowed[:q-1]
                csum,psum=current.sum(),previous.sum()
                if csum:current/=csum
                if psum:previous/=psum
                gain=np.maximum(0,current-previous)
                keys=np.arange(q-1)
                masks=dict(source=keys<P,far_history=(keys>=P)&(q-keys>16),local_history=(keys>=P)&(q-keys<=16))
                values={k:float(gain[v].sum()) for k,v in masks.items()}
                kind=max(values,key=values.get) if csum>0 and psum>0 and sum(values.values())>1e-10 else 'unresolved'
                events.append(dict(id=r['id'],source_id=r['source_id'],t=int(t),kind=kind,gain=values,
                    onset_within8=any(t<=s<=t+8 for s in onset)))
    local_path=Path(args.p4_localization);l=json.loads(local_path.read_text())
    rows=l['per_response'];names=list(rows[0]['auroc']);total_pairs=sum(r['pairs'] for r in rows)
    loso={}
    for source in sorted(set(str(r['source_id']) for r in rows)):
        kept=[r for r in rows if str(r['source_id'])!=source];den=sum(r['pairs'] for r in kept)
        loso[source]=dict(remaining_pairs=den,removed_pair_fraction=1-den/total_pairs,
            within_pair_auc={name:sum(r['auroc'][name]*r['pairs'] for r in kept)/den for name in names})
    counts={kind:dict(events=sum(e['kind']==kind for e in events),onset_windows=sum(e['kind']==kind and e['onset_within8'] for e in events)) for kind in ('source','far_history','local_history','unresolved')}
    report=dict(scope='post-hoc descriptive controls; no new detector/no causal or semantic attribution',
        code_sha256=sha(__file__),p3_manifest_sha256=sha(root/'manifest.json'),
        p3_event_audit_sha256=sha(root/'event_audit_v2.json'),p4_localization_sha256=sha(local_path),
        destination_rule='positive gain in mean common-key conditional attention; argmax source/far-history/local-history; far distance>16; NOT a headwise causal contribution',
        destination_counts=counts,event_destinations=events,
        leave_one_mixed_class_source_out=loso,
        limitation='LOSO is post-hoc on previously seen development. Classes are attention destinations, not semantic evidence owners.')
    text=json.dumps(report,indent=2,allow_nan=False)
    with Path(args.output).open('x') as f:f.write(text+'\n')
    Path(str(args.output)+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(destination_counts=counts,leave_one_mixed_class_source_out=loso)))


if __name__=='__main__':
    main()
