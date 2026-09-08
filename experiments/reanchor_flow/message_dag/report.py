"""Source-level N/H structure comparisons after label-blind graph construction."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..attention_audit_stats import Moments, bracket_positions, matched_difference, mean, token_classes
from ..attention_rhythm_report import save_json
from ..detection_metrics import detection_report
from .structure import NAMES, HEAD_NAMES, SCORES
from .selection import content_positions


def evaluate(output, manifest, *, bootstrap=200):
    output = Path(output)
    selection=manifest.get('selection',{'mode':'uniform','labels_used_for_selection':False})
    paired=selection['mode']=='paired'
    grouped = defaultdict(lambda:defaultdict(list))
    for e in manifest['samples']: grouped[e['split']+'/'+e['task_type']][e['source_id']].append(e)
    report = {'labels_used_for_graph':False,'trained_graph':False,'cohorts':{},'detection_diagnostics':{},
              'mechanism_status':'conditional source attribution; factual semantics and causal effects not established',
              'planned_targets':sum(len(e['targets']) for e in manifest['samples']), 'completed_targets':0,
              'selected_samples':len(manifest['samples']),
              'completed_native_samples':manifest['analysis_coverage']['completed_samples'],
              'missing_samples':[],
              'selection':selection,'labels_used_for_selection':paired,
              'eligible_targets_in_selected_samples':sum(e['eligible_targets'] for e in manifest['samples']),
              'sampling':('label-stratified lexical-content cases; not population detection evaluation' if paired else
                          'uniform label-blind targets; nonzero budgets are not full-token population evaluation')}
    tokens = defaultdict(lambda:defaultdict(list))
    gallery = []
    for group,sources in grouped.items():
        cohort = defaultdict(Moments)
        count = dict(normal=0,hallucinated=0,unknown=0,missing_graph=0,missing_labels=0,matched=0,
                     missing_pairs=0,normal_content=0,hallucinated_content=0,corrected_control_positions=0)
        for source,entries in tqdm(sources.items(),desc=f'DAG N/H {group}',unit='source'):
            acc = defaultdict(Moments)
            for e in entries:
                folder = output/e['folder']
                if not (folder/'meta.npz').exists():
                    count['missing_graph']+=len(e['targets'])
                    count['missing_pairs']+=len(e.get('audit_pairs',[]))
                    report['missing_samples'].append(e['folder'])
                    continue
                with np.load(folder/'meta.npz') as data: meta=dict(data)
                content=set(content_positions(meta).tolist())
                count['corrected_control_positions']+=len(meta.get('added_special_positions',[]))
                n = len(meta['token_ids'])-int(meta['response_start'])
                labels = np.full(n,-1,int)
                if (folder/'labels.npz').exists():
                    with np.load(folder/'labels.npz') as data: labels=data['labels']
                    if labels.shape!=(n,) or not np.isin(labels,(-1,0,1)).all(): raise ValueError('unaligned labels')
                else: count['missing_labels']+=1
                values = np.full((len(NAMES),n),np.nan)
                heads = np.full((len(HEAD_NAMES),int(meta['layers']),int(meta['heads']),n),np.nan)
                selected = np.zeros(n,bool)
                ready = []
                for target in e['targets']:
                    file=folder/f'target_{target}.npz'
                    if not file.exists(): count['missing_graph']+=1; continue
                    with np.load(file) as data:
                        i=target-int(meta['response_start']); y=int(labels[i])
                        values[:,i],heads[...,i]=data['metrics'],data['head_metrics']
                        selected[i]=True; ready.append(target)
                        report['completed_targets']+=1
                        count['normal' if y==0 else 'hallucinated' if y==1 else 'unknown']+=1
                        if y in (0,1) and target in content:
                            count['normal_content' if y==0 else 'hallucinated_content']+=1
                        if y>=0 and np.isfinite(data['scores']).all():
                            arrays=tokens[e['split']]
                            for name,value in zip(SCORES,data['scores']): arrays[name].append(float(value))
                            for name,value in (('labels',y),('source_id',source),('sample_id',e['sample_id']),
                                               ('task_type',e['task_type']),('target',target)):
                                arrays[name].append(value)
                valid=selected & np.isin(labels,(0,1))
                if paired:
                    planned=np.asarray(e.get('audit_pairs',[]),int).reshape(-1,3)-int(meta['response_start'])
                    complete=valid[planned].all(1)
                    count['missing_pairs']+=int((~complete).sum())
                    pairs=planned[complete]
                    if len(pairs) and not ((labels[pairs[:,0]]==1)&(labels[pairs[:,1]]==0)&(labels[pairs[:,2]]==0)).all():
                        raise ValueError('frozen N-H-N labels changed; do not silently rematch controls')
                else:
                    pairs=bracket_positions(labels,token_classes(meta['token_text'][int(meta['response_start']):]),valid,32)
                count['matched']+=len(pairs)
                for name,x in (('structure',values),('head',heads)):
                    acc[name+'_raw'].add(np.stack([mean(x[...,valid & (labels==y)],-1) for y in (0,1)]))
                    acc[name+'_matched'].add(mean(matched_difference(x,pairs),-1))
                if ready:
                    from .view import render_sample
                    page=render_sample(folder,ready)
                    gallery.append((group,str(e['sample_id']),page.relative_to(output).as_posix()))
            for name,value in acc.items():
                if value.total is not None: cohort[name].add(value.finish()['mean'])
        if not cohort: continue
        statistics={name+'_'+key:value for name,acc in cohort.items()
                    for key,value in acc.finish(inference=True).items()}
        statistics.update(metric_names=np.array(NAMES),head_metric_names=np.array(HEAD_NAMES))
        folder=output/'cohorts';folder.mkdir(exist_ok=True)
        file=folder/(group.replace('/','_')+'.npz')
        np.savez_compressed(file,**statistics)
        plot_cohort(statistics,file.with_suffix('.png'),group)
        plot_heads(statistics,file.with_name(file.stem+'_heads.png'),group)
        report['cohorts'][group]={**count,'sources':len(sources),'statistics':file.relative_to(output).as_posix()}
    for split,lists in tokens.items():
        data={key:np.asarray(value) for key,value in lists.items()}
        np.savez_compressed(output/f'{split}_tokens.npz',**data)
        if not paired:
            report['detection_diagnostics'][split]=detection_report(data['labels'],{k:data[k] for k in SCORES},
                             data['source_id'],data['task_type'],primary=SCORES[0],bootstrap=bootstrap)
    report['detection_scope']='not_run_for_label_selected_cases' if paired else 'diagnostic_on_selected_targets'
    report['comparison_ready']=any(c['normal'] and c['hallucinated'] and c['matched'] for c in report['cohorts'].values())
    report['partial']=(report['completed_targets']!=report['eligible_targets_in_selected_samples']
                       or report['selected_samples']!=report['completed_native_samples'])
    report['replication']={}
    for task in sorted({e['task_type'] for e in manifest['samples']}):
        if all(s+'/'+task in report['cohorts'] for s in ('train','test')):
            with np.load(output/report['cohorts']['train/'+task]['statistics']) as a,np.load(output/report['cohorts']['test/'+task]['statistics']) as b:
                valid=np.isfinite(a['head_matched_q_by']) & np.isfinite(b['head_matched_q_by'])
                same=a['head_matched_mean']*b['head_matched_mean']>0
                confirmed=valid & same & (a['head_matched_q_by']<.05) & (b['head_matched_q_by']<.05)
                report['replication'][task]={'tested_head_entries':int(valid.sum()),'same_sign_BY_entries':int(confirmed.sum())}
    save_json(output/'summary.json',report)
    lines=['# 消息 DAG：结构与正常／幻觉比较','',
           f"选中 {report['selected_samples']}/{report['completed_native_samples']} 个已完成原生样本；已完成 {report['completed_targets']}/{report['planned_targets']} 个计划目标；所选样本共有 {report['eligible_targets_in_selected_samples']} 个普通目标。",
           '图按来源单位传播，并从同一个目标反向计入全部后续路径。稀疏边仅用于显示。',
           ('本批按标签选取内容目标及冻结 N-H-N 对照；标签不进入图算子。该富集小批不输出总体检测 AUROC/AP。' if paired else
            '本批均匀抽样；若没有 H 或正常对照，只能检查计算与单类案例。'),
           '这些是条件归因与探索性结构差异，不能直接命名为错误捷径或事实约束失败。','',
           '| cohort | N | H | N内容 | H内容 | matched | missing graphs | missing pairs |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in report['cohorts'].items():
        lines.append(f"| {name} | {c['normal']} | {c['hallucinated']} | {c['normal_content']} | {c['hallucinated_content']} | {c['matched']} | {c['missing_graph']} | {c['missing_pairs']} |")
    lines += ['', 'cohorts/*.npz：来源等权的结构与逐 head N/H 均值、位置配对差异、区间及 BY 校正。',
              '稀疏目标预算可能没有两侧正常对照；缺失显示为缺失，不解释为无差异。',
              ('按标签选取的小批只比较结构与配对差异；总体检测指标需另外使用均匀或全量的固定评估样本。' if paired else
               'summary.json 的 detection_diagnostics 保留材料支持不足这一固定对照及直接读出、logprob、位置的 AUROC/AP。'),
              '该分数不是完整图模型，也不是已发现机制；多跳路径贡献不能沿所有边相加当作一次输出。']
    for split,d in report['detection_diagnostics'].items():
        lines+=['',f'## {split} 固定检测对照','', '| task | score | AUROC | AP |','|---|---|---:|---:|']
        for task,row in d['groups'].items():
            for score,v in row['scores'].items():
                number=lambda x: 'NA' if x is None else f'{x:.4f}'
                lines.append(f"| {task} | {score} | {number(v['auroc'])} | {number(v['auprc'])} |")
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    from html import escape
    links=''.join(f'<li><a href="{escape(path)}">{escape(group)}/{escape(sample)}</a></li>' for group,sample,path in gallery)
    figures=''.join(f'<p>{escape(group)}</p><img width="1100" src="{c["statistics"].replace(".npz",".png")}"><img width="1100" src="{c["statistics"].replace(".npz","_heads.png")}">' for group,c in report['cohorts'].items())
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><h1>消息 DAG</h1><p><a href="summary.md">结构／检测表</a></p>'
                                      +f'<p>目标 {report["completed_targets"]}/{report["planned_targets"]}；红=H，绿=N，灰=特殊。'
                                      +('有 N/H 配对支持。' if report['comparison_ready'] else '当前缺少 N/H 配对支持，不能解释为没有差异。')
                                      +'</p><ul>'+links+'</ul>'+figures,encoding='utf-8')
    import shutil
    shutil.copyfile(Path(__file__).with_name('view.ipynb'),output/'view.ipynb')
    totals={k:sum(c[k] for c in report['cohorts'].values()) for k in ('normal','hallucinated','matched')}
    print(f"DAG report: {report['completed_targets']}/{report['planned_targets']} targets; "
          f"N={totals['normal']} H={totals['hallucinated']} matched={totals['matched']}; {output/'gallery.html'}",flush=True)
    return report


def plot_heads(data,path,title):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes=plt.subplots(len(HEAD_NAMES),3,figsize=(12,9))
    for k,name in enumerate(HEAD_NAMES):
        raw=data['head_raw_mean'][:,k]
        raw_scale=max(1e-9,float(np.nanmax(np.abs(raw)))) if np.isfinite(raw).any() else 1
        for j in range(3):
            value=raw[j] if j<2 else data['head_matched_mean'][k]
            scale=raw_scale if j<2 else max(1e-9,float(np.nanmax(np.abs(value)))) if np.isfinite(value).any() else 1
            ax=axes[k,j]
            m=ax.imshow(value,origin='lower',aspect='auto',cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#eee'),vmin=-scale,vmax=scale)
            ax.set(title=name+': '+('N','H','matched H-N')[j],xlabel='Physical head',ylabel='Layer')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if j==2:
                l,h=np.where(data['head_matched_q_by'][k]<.05);ax.scatter(h,l,s=6,c='black')
            if np.isfinite(value).any():fig.colorbar(m,ax=ax,shrink=.7)
            else:ax.text(.5,.5,'No matched support' if j==2 else 'No eligible targets',transform=ax.transAxes,ha='center')
    fig.suptitle(title+' | same-target all-path signed contributions; dots: BY q < .05')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def plot_cohort(data,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for k,(name,ax) in enumerate(zip(NAMES,axes.flat)):
        values=data['structure_raw_mean'][:,k]
        intervals=data['structure_raw_ci95'][:,:,k]
        for y in (0,1):
            if np.isfinite(values[y]):
                ax.bar(y,values[y],color=('#4c9c75','#c95858')[y])
                if np.isfinite(intervals[:,y]).all(): ax.plot([y,y],intervals[:,y],color='black')
        ax.set(xticks=[0,1],xticklabels=['N','H'],title=name)
        effect=data['structure_matched_mean'][k]
        ax.set_xlabel(f'matched H-N={effect:.3g}' if np.isfinite(effect) else 'No matched support')
    fig.suptitle(title+' | source-balanced; intervals are descriptive, BY in NPZ')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
