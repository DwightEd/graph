"""Offline, source-balanced comparisons; labels never define lookback events."""
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..attention_audit_stats import (
    bracket_positions,
    matched_difference,
    mean,
    token_classes,
)
from ..attention_rhythm_report import save_json
from .artifacts import atomic_path

METRICS = ('local_mass','remote_mass','time_tv','remote_gain')
COHORTS = ('all_H','start_H','continuing_H')
HORIZONS = [(str(k),k,k) for k in range(17)]+[('17-64',17,64),('65+',65,None)]


def source_summary(values, bootstrap=200):
    """Each input row is one independent source, not a head/event/token."""
    data = np.asarray(values,float)
    valid = np.isfinite(data)
    average = mean(data,0)
    ci = np.full((2,*average.shape),np.nan)
    if len(data)>1 and bootstrap:
        rng = np.random.default_rng(82)
        weights = rng.multinomial(len(data),np.full(len(data),1/len(data)),size=bootstrap)
        denominator = weights@valid.reshape(len(data),-1)
        sums = weights@np.where(valid,data,0).reshape(len(data),-1)
        draws = np.divide(sums,denominator,out=np.full_like(sums,np.nan),where=denominator>0)
        for i in range(draws.shape[1]):
            finite = draws[np.isfinite(draws[:,i]),i]
            if len(finite) and valid.reshape(len(data),-1)[:,i].sum()>1:
                ci.reshape(2,-1)[:,i] = np.quantile(finite,[.025,.975])
    return average,ci,valid.sum(0)


def labels_for(folder, scan):
    n = len(scan['token_ids'])-int(scan['response_start'])
    path = folder/'labels.npz'
    if not path.exists(): return np.full(n,-1,int)
    with np.load(path,allow_pickle=False) as saved: labels = saved['labels']
    if labels.shape!=(n,) or not np.isin(labels,(-1,0,1)).all():
        raise ValueError(f'{path}: invalid labels for this response')
    return labels


def plot_cohort(path, values, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,4,figsize=(13,8),layout='constrained')
    for j,metric in enumerate(METRICS):
        limit = np.nanmax(np.abs(values[j])) if np.isfinite(values[j]).any() else 1
        limit = max(limit,1e-8)
        for i,cohort in enumerate(COHORTS):
            ax = axes[i,j]
            im = ax.imshow(values[j,i],aspect='auto',origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_title(f'{cohort}: {metric}\n{counts[i]} independent sources',fontsize=9)
            ax.set_xlabel('physical head');ax.set_ylabel('physical layer')
        fig.colorbar(im,ax=axes[:,j],shrink=.7,label='matched H minus normal')
    fig.suptitle('One cell = one physical layer/head; rows are cohorts, not individual tokens\n'
                 'Same-response, same-token-class normal controls on both sides; source-balanced means',fontsize=10)
    fig.savefig(path,dpi=150);plt.close(fig)


def evaluate(output, *, bootstrap=200):
    from .event_view import render_event
    from .reanchor import REQUIRED_SCAN_FIELDS, ReanchorProfiler
    from .reanchor_report import summarize_reanchor
    from .transport_report import report_event_incidence, report_transport
    output = Path(output)
    manifest = json.loads((output/'index.json').read_text())
    heads = defaultdict(lambda:defaultdict(list))
    trajectories = defaultdict(lambda:defaultdict(list))
    candidate_states = defaultdict(lambda:defaultdict(list))
    observations = defaultdict(int)
    coverage = [];preview_links = []
    reanchor_records = [];reanchor_missing = []
    match_counts = defaultdict(lambda:np.zeros(3,int))
    for e in manifest['samples']:
        folder = output/e['folder'];scan_path = folder/'scan.npz'
        record = dict(sample=f"{e['split']}/{e['task_type']}/{e['sample_id']}",scanned=False)
        coverage.append(record)
        if not scan_path.exists():
            reanchor_missing.append(record['sample'])
            continue
        with np.load(scan_path,allow_pickle=False) as data: scan = dict(data)
        labels = labels_for(folder,scan)
        if all(name in scan for name in REQUIRED_SCAN_FIELDS):
            profile = ReanchorProfiler().run(scan)
            serializable_profile = {
                name: (
                    np.array(json.dumps(value, sort_keys=True))
                    if isinstance(value, dict)
                    else value
                )
                for name, value in profile.items()
            }
            with atomic_path(folder/'reanchor_profile.npz') as temporary:
                np.savez_compressed(temporary, **serializable_profile)
            reanchor_records.append({
                'group': e['split']+'/'+e['task_type'],
                'source': str(e['source_id']),
                'profile': profile,
                'labels': labels,
                'response_start': int(scan['response_start']),
                'special_mask': scan['special_mask'],
                'predictor_logprob': scan.get(
                    'predictor_logprob', np.full(len(scan['row_position']), np.nan)
                ),
            })
        else:
            reanchor_missing.append(record['sample'])
        start = int(scan['response_start']);rows = scan['row_position']
        special = scan['special_mask']
        targets = rows[:-1]+1
        if not np.array_equal(targets,np.arange(start,len(special))):
            raise ValueError(f'{scan_path}: response prediction coordinates are not contiguous')
        target_valid = ~special[targets] & ~special[rows[:-1]]
        valid = target_valid.copy()
        valid[1:] &= ~special[rows[:-2]]  # adjacent ordinary queries for time changes
        group = e['split']+'/'+e['task_type'];source = str(e['source_id'])
        pairs = bracket_positions(labels,token_classes(scan['token_text'][start:]),valid,32)
        previous = np.r_[-1,labels[:-1]]
        subsets = (pairs,pairs[previous[pairs[:,0]]==0],pairs[previous[pairs[:,0]]==1])
        for i,pair in enumerate(subsets):
            match_counts[group][i] += len(pair)
            if not len(pair): continue
            for metric in METRICS:
                difference = mean(matched_difference(scan[metric][...,:-1],pair),-1)
                heads[group,metric,COHORTS[i]][source].append(difference)
        event_rows = np.unique(scan['event_index'][:,2])
        record.update(scanned=True,response_tokens=len(labels),ordinary_tokens=int((~special[start:]).sum()),
                      H=int(((labels==1)&~special[start:]).sum()),N=int(((labels==0)&~special[start:]).sum()),
                      unknown=int(((labels==-1)&~special[start:]).sum()),events=len(event_rows),
                      read_sites=len(scan['event_index']),traced=0,missing_event_positions=[])
        sample_stats = defaultdict(list)
        first = None
        for row in event_rows:
            position = int(rows[row]);path = folder/f'event_{position}.npz'
            if not path.exists(): record['missing_event_positions'].append(position);continue
            with np.load(path,allow_pickle=False) as data: event = dict(data)
            record['traced'] += 1
            if first is None: first = event
            if not np.array_equal(event['target_position'],targets): raise ValueError(f'{path}: target mismatch')
            offset = targets-position-1  # offset 0 is the first token predicted by the carrier
            response = event['margin_response'].transpose(2,0,1)
            for name,lo,hi in HORIZONS:
                eligible = target_valid & (offset>=lo)
                if hi is not None: eligible &= offset<=hi
                for label_name,label in (('N',0),('H',1),('unknown',-1)):
                    for kind,explicit in (('observed_runner',False),('explicit_candidates',True)):
                        mask = eligible & (labels==label) & (event['explicit_contrast']==explicit)
                        if not mask.any(): continue
                        key = (group,name,label_name,kind)
                        baseline = event['baseline_margin'][mask]
                        helping_behind = (baseline<0)&(response[mask,0].sum(-1)>0)
                        sample_stats[key].append((response[mask].sum(0),int(mask.sum()),
                                                 np.array([baseline.sum(),helping_behind.sum()])))
                        observations[key] += int(mask.sum())
        for key,items in sample_stats.items():
            trajectories[key][source].append(sum(x[0] for x in items)/sum(x[1] for x in items))
            candidate_states[key][source].append(sum(x[2] for x in items)/sum(x[1] for x in items))
        if first is not None:
            render_event(folder,first,scan,labels,folder/'preview.html')
            preview_links.append((record['sample'],e['folder']+'/preview.html',int(first['event_position'])))
    tables = output/'cohorts';tables.mkdir(exist_ok=True)
    # Relabeling may remove a cohort; do not leave stale inference tables behind.
    for pattern in ('*_heads.npz','*_heads.png'):
        for path in tables.glob(pattern): path.unlink()
    head_summary = []
    for group in sorted({k[0] for k in heads}):
        example = next(v for k,v in heads.items() if k[0]==group)
        shape = next(iter(example.values()))[0].shape
        effect = np.full((4,3,*shape),np.nan);ci = np.full((4,3,2,*shape),np.nan)
        support = np.zeros((4,3,*shape),int);sources = np.zeros(3,int)
        for j,metric in enumerate(METRICS):
            for i,cohort in enumerate(COHORTS):
                by_source = heads.get((group,metric,cohort),{})
                if not by_source: continue
                values = [mean(v,0) for v in by_source.values()]
                effect[j,i],ci[j,i],support[j,i] = source_summary(values,bootstrap)
                sources[i] = len(by_source)
        name = group.replace('/','_')
        np.savez_compressed(tables/f'{name}_heads.npz',metrics=METRICS,cohorts=COHORTS,
                            matched_difference=effect,ci95=ci,valid_sources=support,
                            matched_targets=match_counts[group],sources=sources)
        plot_cohort(tables/f'{name}_heads.png',effect,sources)
        head_summary.append(dict(group=group,sources=sources,matched_targets=match_counts[group]))
    curve_rows = []
    for (group,horizon,label,kind),by_source in sorted(trajectories.items()):
        avg,ci,n = source_summary([mean(v,0) for v in by_source.values()],bootstrap)
        for v,variant in enumerate(('full','fixed_qk','no_mlp_paths')):
            for h,hop in enumerate(('0','1','2+')):
                curve_rows.append(dict(group=group,offset=horizon,label=label,contrast=kind,variant=variant,hop=hop,
                                       sources=int(n[v,h]),event_target_pairs=observations[group,horizon,label,kind],
                                       mean=avg[v,h],ci_low=ci[0,v,h],ci_high=ci[1,v,h]))
    columns = ['group','offset','label','contrast','variant','hop','sources','event_target_pairs','mean','ci_low','ci_high']
    with (tables/'event_outcomes.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,columns);writer.writeheader();writer.writerows(curve_rows)
    state_rows = []
    for key,by_source in sorted(candidate_states.items()):
        group,horizon,label,kind = key
        avg,ci,n = source_summary([mean(v,0) for v in by_source.values()],bootstrap)
        state_rows.append(dict(group=group,offset=horizon,label=label,contrast=kind,sources=int(n[0]),
                               event_target_pairs=observations[key],baseline_mean=avg[0],baseline_ci_low=ci[0,0],
                               baseline_ci_high=ci[1,0],positive_gain_while_behind_rate=avg[1],
                               rate_ci_low=ci[0,1],rate_ci_high=ci[1,1]))
    columns = ['group','offset','label','contrast','sources','event_target_pairs','baseline_mean',
               'baseline_ci_low','baseline_ci_high','positive_gain_while_behind_rate','rate_ci_low','rate_ci_high']
    with (tables/'candidate_states.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,columns);writer.writeheader();writer.writerows(state_rows)
    summary = dict(native_coverage=manifest['analysis_coverage'],samples=coverage,head_comparisons=head_summary,
                   scanned=sum(r['scanned'] for r in coverage),events=sum(r.get('events',0) for r in coverage),
                   traced=sum(r.get('traced',0) for r in coverage),
                   zero_event_samples=sum(r['scanned'] and r.get('events')==0 for r in coverage),
                   bootstrap=bootstrap,labels_used_for_events=False,labels_used_for_dag=False)
    signed_edges = manifest.get('event_settings',{}).get('schema',1)>=2
    summary['transport'] = (
        report_transport(output,manifest,bootstrap=bootstrap)
        if signed_edges
        else {
            'status': 'not_available_v1',
            'hurdle': report_event_incidence(output,manifest,bootstrap=bootstrap),
        }
    )
    if reanchor_records:
        reanchor = summarize_reanchor(reanchor_records, bootstrap=bootstrap)
        reanchor.update(
            status='partial' if reanchor_missing else 'complete',
            missing_scans=reanchor_missing,
        )
    else:
        reanchor = {
            'status': 'not_available_rescan_required',
            'missing_scans': reanchor_missing,
            'labels_used_for_profile': False,
            'labels_used_for_outcomes': False,
        }
    summary['reanchor'] = reanchor
    save_json(output/'reanchor_summary.json', reanchor)
    save_json(output/'summary.json',summary)
    lines = ['# 内部回看事件审计','',
             f"请求原生样本 {summary['native_coverage']['planned_samples']}；完成扫描 {summary['scanned']}；"
             f"回看位置 {summary['events']}；完成传播 {summary['traced']}；无事件样本 {summary['zero_event_samples']}。",'',
             '所有可用事件均计划传播；缺失缓存及未完成事件见 summary.json。源可以是 prompt 或旧回答。',
             '预测目标 t 使用 query t−1；回看载体 b 本身的标签不代替其后续目标标签。',
             'offset=0 指 b 预测的第一个 token b+1。未来不足记为缺失，不填零。',
             '后续统计使用每个实际可用的普通 query/target；17–64、65+ 区间按可用目标汇总，需同时看暴露数量。','',
             '## 结构复核','',
             'cohorts/*_heads.png 的每格是一个物理 layer/head 的 H−N 差，绝非一个幻觉 token。',
             '三排分别是全部匹配 H、已知 N→H 起点、已知 H→H 延续；不将未知边界算成起点或延续。',
             'time_tv 是同一 head 在相邻 query token 间的 attention 总变差，不是层间或头间变化。',
             '同回答、同粗 token 类别、两侧32 token 内正常对照插值；先回答内，再 source 等权。',
             '95%区间对 source 重抽样。它不是全 head 多重检验后的显著性声明；没有匹配支持则无图。','',
             '## 事件传播','',
             'cohorts/event_outcomes.csv 汇总事件之后的 N/H 响应，先回答内再 source 等权；重复事件不作为独立 source。',
             '同一 source 的完整结果保存在每个 event_<绝对位置>.npz；所有实际后续 token 均参与传播。',
             'observed_runner 的正负仅表示对当前 token 相对 runner 的支持，不能自动当成正确／错误证据。',
             'explicit_candidates 单独汇总，不与 observed_runner 混合；单 token 候选也不等于完整事实判定。',
             'cohorts/candidate_states.csv 保存当前候选差，以及 M<0 且完整消息响应 C>0 的 source 等权比例。',
             '该比例描述名义正候选落后而消息增强有正向局部增益；只有候选真值独立确定，才可联系正确／错误证据。',
             '0/1/2+ 跳相加还原总局部响应；full−fixed_qk 与 full−no_mlp_paths 两个路径差有重叠。',
             '状态范数、MLP方向抵消不直接等于信息量或语义丢失。teacher forcing 不证明自由生成闭环。','']
    lines += (['## 有符号传播图与固定检测候选','',
              '`edges_<b>.npz`保存全部物理层/头、中继、目标的V/K边，未用top-k截断。',
              '每条路径按最后跨位置边计入一次；`cut_closure_error`核对直接/多跳之和与原响应一致。',
              '这分解的是候选差的局部导数，不是候选差本身，也不是已经解耦的正确事实流量。',
              '`transport_detection.json/png`报告固定 opposition 分数及直接路径、V-only、置信度、位置对照。',
              '比较使用相同有效token并按source配对重抽样；具体条件覆盖与缺失原因必须同时查看。',
              '没有此前事件、零作用、未完成传播或明确真值候选的位置不填零，不算入无监督检测。',
              '该分数尚需真实8B数据检验，不将正负路径作用直接命名为幻觉机制。',''] if signed_edges
              else ['当前是v1跳数审计；没有v2逐边传播图或opposition检测分数。',''])
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    links = '\n'.join(f'<li><a href="{html.escape(link,quote=True)}">{html.escape(name)} · b={b}</a></li>' for name,link,b in preview_links)
    figures = ''.join(f'<p>{html.escape(x["group"])}</p><img style="max-width:100%" src="cohorts/{x["group"].replace("/","_")}_heads.png">' for x in head_summary)
    transport_link = ('<p><a href="transport_detection.json">有符号传播：AUROC/AP、配对区间与覆盖</a> · '
                      '<a href="transport_detection.png">同token检测对照曲线</a></p>' if signed_edges
                      else '<p>v1跳数审计；不包含v2逐边传播图及检测分数。</p>')
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>内部回看审计</title>'
        '<main style="max-width:1100px;margin:35px auto;font:16px/1.6 system-ui"><h1>内部回看审计</h1>'
        f'<p>扫描 {summary["scanned"]} 个样本；{summary["events"]} 个回看位置；已传播 {summary["traced"]} 个。</p>'
        '<p>每个样本预览最早完成的内部事件，未按 H 标签挑选。其他事件可用 event_view --position 打开。</p>'
        '<p>下图三排为全部匹配 H、已知起点、延续位置。每格是一个物理 layer/head 的平均差；不是单 token。</p>'
        f'{transport_link}{figures}<ul>{links}</ul><a href="summary.md">定义、覆盖与解释边界</a></main>',encoding='utf-8')
    print(f"event report: scanned={summary['scanned']}, events={summary['events']}, traced={summary['traced']}; {output/'gallery.html'}",flush=True)
    return summary
