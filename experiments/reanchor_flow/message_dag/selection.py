"""Small label-stratified comparisons; labels never enter DAG operators.

Content is a disclosed lexical heuristic, not a factuality classifier. Select
the first token of a content word/number, excluding scaffolding and citations.
Freeze same-answer, same-token-class N-H-N triples before examining graph data.
"""
from collections import defaultdict
import re

import numpy as np
from tqdm.auto import tqdm

from ..attention_audit_stats import bracket_positions, token_classes
from .cache import read_trace, target_positions

STOP = set('a an the and or but as at by for from in into of on to with without '
           'is are was were be been being it its this that these those there here '
           'i you he she we they them their his her our your my me us which who what '
           'how when where why can could may might will would should do does did '
           'not no yes some any all also than then so based provided passage passages '
           'source sources document documents reference references according answer '
           'question therefore unable following given output'.split())
WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*",re.UNICODE)


def content_positions(trace):
    start = int(trace['response_start'])
    pieces = trace['token_text'][start:]
    text = ''.join(pieces)
    offsets = np.r_[0,np.cumsum([len(str(s)) for s in pieces])]
    eligible = set(target_positions(trace,0).tolist())
    result = set()
    for word in WORDS.finditer(text):
        value = word.group().casefold()
        if value in STOP: continue
        if value.isdigit():
            before = text[:word.start()]
            line = before[before.rfind('\n')+1:]
            if re.search(r'\b(?:passage|source|document|reference)\s*$',before,re.I): continue
            if not line.strip() and re.match(r'[.)](?:\s|$)',text[word.end():]): continue
        position = start + int(np.searchsorted(offsets,word.start(),side='right')-1)
        if position in eligible: result.add(position)
    return np.array(sorted(result),int)


def read_labels(path, trace):
    with np.load(path.with_suffix('.labels.npz'),allow_pickle=False) as data:
        labels = data['labels']
    n = len(trace['token_ids'])-int(trace['response_start'])
    if labels.shape!=(n,) or not np.isin(labels,(-1,0,1)).all():
        raise ValueError(f'{path}: labels must cover the original response with -1/0/1')
    return labels


def inventory(audit, entries):
    records=[]
    for e in tqdm(entries,desc='comparison inventory (text + labels only)',unit='sample'):
        path=audit/e['path']; trace=read_trace(path); labels=read_labels(path,trace)
        start=int(trace['response_start']); ordinary=~trace['special_mask'][start:]
        content=content_positions(trace); valid=np.zeros(len(labels),bool);valid[content-start]=True
        pairs=bracket_positions(labels,token_classes(trace['token_text'][start:]),valid,32)+start
        counts={name:int(((labels==y)&ordinary).sum()) for name,y in (('N',0),('H',1),('unknown',-1))}
        cls='H' if counts['H'] else 'N' if counts['N'] and not counts['unknown'] else 'unknown'
        records.append(dict(entry=dict(e),classification=cls,counts=counts,content=content,
                            pairs=pairs,labels=labels,trace=trace))
    return records


def uniform(items, count):
    if count and len(items)>count:
        return [items[i] for i in np.linspace(0,len(items)-1,count).round().astype(int)]
    return list(items)


def target_details(trace, targets, labels=None):
    start=int(trace['response_start']);text=trace.get('token_text',np.full(len(trace['token_ids']),''))
    content=set(content_positions(trace).tolist()) if 'token_text' in trace else set()
    return [dict(position=int(t),text=str(text[t]),content_candidate=t in content,
                 label=int(labels[t-start]) if labels is not None else -1) for t in targets]


def paired_plan(audit, entries, sample_budget, target_budget):
    if sample_budget and (sample_budget<2 or sample_budget%2):
        raise ValueError('paired selection needs an even --samples-per-group >=2 (or 0 for all eligible sources)')
    if target_budget and target_budget<3:
        raise ValueError('paired selection needs --targets-per-sample >=3 for N-H-N triples (or 0)')
    records=inventory(audit,entries);groups=defaultdict(list)
    for r in records:groups[r['entry']['split']+'/'+r['entry']['task_type']].append(r)
    selected=[];coverage={};skipped=[]
    for name,rows in groups.items():
        pools={cls:[] for cls in ('N','H')}
        # At most one answer per source in each class; no graph-based ranking.
        for cls in pools:
            seen=set()
            for r in sorted(rows,key=lambda r:(str(r['entry']['source_id']),str(r['entry']['sample_id']))):
                source=str(r['entry']['source_id'])
                usable=len(r['pairs'])>0 if cls=='H' else len(r['content'])>0
                if r['classification']==cls and usable and source not in seen:
                    seen.add(source);pools[cls].append(r)
        c={cls+'_answers':sum(r['classification']==cls for r in rows) for cls in ('N','H','unknown')}
        c.update(N_sources=len(pools['N']),H_sources_with_content_pairs=len(pools['H']),
                 added_control_positions=sum(len(r['trace']['added_special_positions']) for r in rows))
        c.update({cls+'_content_tokens':sum(int((r['labels'][r['content']-int(r['trace']['response_start'])]==y).sum())
                                            for r in rows) for cls,y in (('N',0),('H',1))})
        print(f'coverage {name}: {c}',flush=True)
        coverage[name]=c
        if not pools['N'] or not pools['H']:
            skipped.append(name)
            print(f'skip {name}: need both full-N answers and H content targets with two normal controls; no GPU work for this group',flush=True)
            continue
        count=min(len(pools['N']),len(pools['H']),sample_budget//2 if sample_budget else len(rows))
        chosen=uniform(pools['H'],count)+uniform(pools['N'],count)
        c['selected_per_class']=count
        for r in chosen:
            trace=r['trace'];start=int(trace['response_start']);e=r['entry']
            if r['classification']=='H':
                pairs=np.asarray(uniform(r['pairs'],target_budget//3 if target_budget else 0),int).reshape(-1,3)
                targets=np.unique(pairs) if target_budget else r['content']
            else:
                pairs=np.empty((0,3),int);targets=np.array(uniform(r['content'],target_budget),int)
            e.update(targets=targets.tolist(),eligible_targets=len(target_positions(trace,0)),
                     comparison_class=r['classification'],audit_pairs=pairs.tolist(),
                     target_details=target_details(trace,targets,r['labels']),
                     added_special_positions=trace['added_special_positions'].tolist())
            selected.append(e)
    summary=dict(mode='paired',labels_used_for_selection=True,labels_used_for_graph=False,
                 target_policy='first token of lexical content words/numbers; not a semantic fact selector',
                 match_policy='same answer, same coarse token class, N on both sides within 32 tokens',
                 groups=coverage,skipped_groups=skipped)
    if not selected:
        raise ValueError(f'No supported N/H content comparison in completed caches: {coverage}. '
                         'No model was loaded. Do not rerun all targets of all-N answers; supply an H-bearing completed scope.')
    return selected,summary
