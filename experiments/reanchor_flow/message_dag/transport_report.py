"""A frozen, label-free opposition score; labels enter only the final report."""
from pathlib import Path

import numpy as np

from ..attention_rhythm_report import save_json
from ..detection_metrics import detection_report, render_detection_report

SCORES = ('opposition','direct_only','value_only','negative_logprob','relative_position')


def opposition(positive, negative):
    total = positive+negative
    return np.divide(negative,total,out=np.full_like(total,np.nan,dtype=float),where=total>0)


def sample_scores(folder, scan):
    """The latest strictly preceding PLANNED event, without label selection.

    A missing latest event remains missing; an older complete event must not
    silently change the detector. Explicit truth contrasts are excluded.
    """
    rows = np.asarray(scan['row_position'])[:-1]
    targets = rows+1
    events = np.unique(scan['event_index'][:,2])
    previous = np.searchsorted(events,np.arange(len(rows)),side='left')-1
    selected = np.full(len(rows),-1,int)
    available = previous>=0
    selected[available] = events[previous[available]]
    risk = np.full((len(SCORES),len(rows)),np.nan)
    reason = np.full(len(rows),'no_prior_event',dtype='<U24')
    for row in np.unique(selected[available]):
        take = selected==row
        reason[take] = 'event_not_traced'
        path = Path(folder)/f"event_{int(scan['row_position'][row])}.npz"
        if not path.exists(): continue
        with np.load(path,allow_pickle=False) as event:
            if 'cut_hop_positive' not in event:
                reason[take] = 'cut_not_available';continue
            p,n = event['cut_hop_positive'],event['cut_hop_negative']
            values = np.stack((opposition(p.sum((0,2)),n.sum((0,2))),
                               opposition(p[0].sum(-1),n[0].sum(-1)),
                               opposition(p[...,0].sum(0),n[...,0].sum(0))))
            explicit = event['explicit_contrast']
            valid = take & ~explicit
            risk[:3,valid] = values[:,valid]
            reason[valid] = np.where(np.isfinite(risk[0,valid]),'scored','zero_path_response')
            reason[take & explicit] = 'explicit_contrast'
    special = scan['special_mask'][rows] | scan['special_mask'][targets]
    reason[special] = 'special_token'
    risk[:3,special] = np.nan
    if 'predictor_logprob' in scan: risk[3] = -scan['predictor_logprob'][:-1]
    risk[4] = np.arange(len(rows))/max(1,len(rows)-1)
    return dict(target_position=targets,event_row=selected,score_names=np.array(SCORES),risk=risk,
                reason=reason,labels_used=np.array(False),primary=np.array('opposition'))


def report_event_incidence(output, manifest, *, bootstrap=200):
    """Report the hurdle's first stage when legacy cuts have no transport score."""

    from .event_report import labels_for
    from .hurdle import summarize_hurdle

    output = Path(output)
    samples = []
    for entry in manifest["samples"]:
        folder = output / entry["folder"]
        scan_path = folder / "scan.npz"
        if not scan_path.exists():
            continue
        with np.load(scan_path, allow_pickle=False) as archive:
            scan = dict(archive)
        rows = np.asarray(scan["row_position"])[:-1]
        targets = rows + 1
        event_rows = np.unique(scan["event_index"][:, 2])
        prior_event = np.searchsorted(event_rows, np.arange(len(rows)), side="left") > 0
        special = scan["special_mask"]
        ordinary = ~special[rows] & ~special[targets]
        samples.append(
            {
                "group": entry["split"] + "/" + entry["task_type"],
                "source": str(entry["source_id"]),
                "labels": labels_for(folder, scan),
                "ordinary": ordinary,
                "event": prior_event,
                "transport": np.full(len(rows), np.nan),
            }
        )
    result = summarize_hurdle(samples, bootstrap=bootstrap)
    save_json(output / "transport_hurdle.json", result)
    return result


def report_transport(output, manifest, *, bootstrap=200):
    from .event_report import labels_for
    from .hurdle import summarize_hurdle
    output = Path(output)
    ys,values,sources,tasks,coverage,hurdle_samples = [],[],[],[],[],[]
    for entry in manifest['samples']:
        folder = output/entry['folder']
        if not (folder/'scan.npz').exists(): continue
        with np.load(folder/'scan.npz',allow_pickle=False) as archive: scan = dict(archive)
        result = sample_scores(folder,scan)
        np.savez_compressed(folder/'transport_scores.npz',**result)
        # Only this reporting stage reads outcome annotations.
        labels = labels_for(folder,scan)
        common = np.isfinite(result['risk']).all(0)
        ordinary = result['reason']!='special_token'
        common &= ordinary
        primary = np.isfinite(result['risk'][0]) & ordinary
        unique,counts = np.unique(result['reason'],return_counts=True)
        coverage.append(dict(sample=f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}",
                             ordinary_tokens=int(ordinary.sum()),primary_scored=int(primary.sum()),
                             ordinary_H=int((ordinary & (labels==1)).sum()),
                             ordinary_N=int((ordinary & (labels==0)).sum()),
                             ordinary_unknown=int((ordinary & ~np.isin(labels,[0,1])).sum()),
                             common_scored=int(common.sum()),scored_H=int((common & (labels==1)).sum()),
                             scored_N=int((common & (labels==0)).sum()),
                             scored_unknown=int((common & ~np.isin(labels,[0,1])).sum()),
                             reasons=dict(zip(unique,counts.tolist()))))
        hurdle_samples.append({
            'group': entry['split']+'/'+entry['task_type'],
            'source': str(entry['source_id']),
            'labels': labels,
            'ordinary': ordinary,
            'event': result['event_row'] >= 0,
            'transport': result['risk'][0],
        })
        ys.append(labels[common]);values.append(result['risk'][:,common])
        sources.extend([str(entry['source_id'])]*int(common.sum()))
        tasks.extend([entry['split']+'/'+entry['task_type']]*int(common.sum()))
    y = np.concatenate(ys) if ys else np.empty(0,int)
    x = np.concatenate(values,axis=1) if values else np.empty((len(SCORES),0))
    scores = dict(zip(SCORES,x))
    task_array = np.asarray(tasks,dtype=str)
    report = detection_report(y,scores,np.asarray(sources,dtype=str),task_array,
                               primary='opposition',bootstrap=bootstrap)
    hurdle = summarize_hurdle(hurdle_samples,bootstrap=bootstrap)
    save_json(output/'transport_hurdle.json',hurdle)
    report.update(status='two_stage_unvalidated_candidate',
                  event_selection='latest_detected_event_strictly_before_query',
                  comparison_scope='same_finite_tokens_for_all_scores',
                  labels_used_for_score=False,coverage=coverage,
                  hurdle=hurdle,
                  ordinary_tokens=sum(c['ordinary_tokens'] for c in coverage),
                  primary_scored=sum(c['primary_scored'] for c in coverage),
                  common_scored=len(y))
    save_json(output/'transport_detection.json',report)
    render_detection_report(output/'transport_detection.png',y,scores,task_array,primary='opposition')
    print(f"transport scores: primary={report['primary_scored']}/{report['ordinary_tokens']} ordinary tokens; "
          f"paired comparison={len(y)}; coverage is conditional, not full-token detection",flush=True)
    for group,stats in report['groups'].items():
        metric = stats['scores']['opposition']
        print(f"  {group}: AUROC={metric['auroc']} AP={metric['auprc']} "
              f"prevalence={stats['prevalence']} sources={stats['sources']}",flush=True)
    return dict(primary='opposition',ordinary_tokens=report['ordinary_tokens'],
                primary_scored=report['primary_scored'],common_scored=len(y),hurdle=hurdle)
