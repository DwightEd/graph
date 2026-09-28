"""Preserve confident rejection events instead of letting the joint readout erase them."""
import argparse
from pathlib import Path
import numpy as np
from .run import read_json,write_json,allowed_mask,exposed_sources,evaluate_frozen
from experiments.probabilistic_detection.data import load_pack
from experiments.probabilistic_detection.evaluation import threshold_at_fpr


def tail_score(reference,values):
    ordered=np.sort(reference)
    tail=(1+len(ordered)-np.searchsorted(ordered,values,side='left'))/(len(ordered)+1)
    return -np.log(tail)


def span_recall(labels,onsets,answers,alarm,eligible):
    hits=[]
    for start in np.flatnonzero(onsets & eligible):
        stop=start+1
        while stop<len(labels) and labels[stop] and answers[stop]==answers[start] and not onsets[stop]:
            stop+=1
        hits.append(bool(alarm[start:stop].any()))
    return float(np.mean(hits))


def candidate_scores(saved,references):
    rejection=saved['surprisal']-saved['entropy']
    scores={name:tail_score(references[name],values) for name,values in
        dict(joint=saved['joint'],static=saved['static'],rejection=rejection,entropy=saved['entropy']).items()}
    result=dict(joint=scores['joint'],static=scores['static'],confident_rejection=scores['rejection'])
    for weight in (.25,1.,4.):
        name=f'joint_rejection_{weight:g}'
        result[name]=np.maximum(scores['joint'],scores['rejection']+np.log(weight))
        result[name+'_entropy']=np.maximum(result[name],scores['entropy']+np.log(.25))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packs','frozen','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    excluded=exposed_sources(args.packs)
    for task in ('QA','Summary','Data2txt'):
        pack,meta=load_pack(args.packs,task,'train')
        allowed=allowed_mask(meta,excluded,len(pack['target']))
        dev=allowed & pack['development']
        normal=dev & (pack['labels']==0)
        with np.load(args.frozen/task/'train_scores.npz') as data:
            saved={name:data[name] for name in ('joint','static','entropy','surprisal')}
        references=dict(joint=saved['joint'][normal],static=saved['static'][normal],
            rejection=(saved['surprisal']-saved['entropy'])[normal],entropy=saved['entropy'][normal])
        scores=candidate_scores(saved,references)
        thresholds={name:threshold_at_fpr(pack['labels'][dev],score[dev]) for name,score in scores.items()}
        recalls={name:span_recall(pack['labels'],pack['onsets'],pack['answer_index'],score>thresholds[name],dev) for name,score in scores.items()}
        selected=max(scores,key=lambda name:recalls[name])
        directory=args.output/task
        directory.mkdir()
        write_json(directory/'selection.json',dict(selected=selected,thresholds=thresholds,
            development_span_recall=recalls,labels_used_for_selection=True,excluded_sources=sorted(excluded),
            rule='highest dev span recall at 5% normal token FPR; source excluded; no case-specific thresholds'))
        np.savez_compressed(directory/'references.npz',**references)
        np.savez_compressed(directory/'train_scores.npz',**scores)
        with np.load(args.frozen/task/'test_scores.npz') as data:
            scores=candidate_scores(data,references)
        np.savez_compressed(directory/'test_scores.npz',**scores)
        write_json(directory/'test_frozen.json',dict(status='frozen'))
    write_json(args.output/'SCORING_FREEZE.json',dict(status='all tasks frozen'))
    evaluate_frozen(args.packs,args.output)
    print('FUSION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
