"""Physical-head cohort plots, label-balanced examples and complete token text."""
from __future__ import annotations

import html
from pathlib import Path
import numpy as np

from .attention_audit_stats import read_metrics, ratio
from .attention_rhythm_report import save_json


def _pyplot():
    # Preserve Jupyter's inline backend; headless CLI Matplotlib defaults to Agg.
    import matplotlib.pyplot as plt
    return plt


def heatmap(ax, values, title, *, limit=None, signed=False, pair_heads=None):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    limit = limit or (float(np.max(np.abs(finite))) if len(finite) else 1.) or 1.
    im = ax.imshow(values, origin='upper', interpolation='nearest', aspect='auto',
                   cmap='RdBu_r' if signed else 'viridis', vmin=-limit if signed else 0, vmax=limit)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Reader L*H+h' if pair_heads else 'Head')
    ax.set_ylabel('Writer L*H+h' if pair_heads else 'Layer')
    if pair_heads:
        n = len(values)
        ticks = np.unique(np.linspace(0, n-1, 5, dtype=int))
        labels = [f'L{x//pair_heads}H{x%pair_heads}' for x in ticks]
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_yticks(ticks, labels, fontsize=7)
    else:
        ax.set_xticks(np.unique(np.linspace(0, values.shape[1]-1, 5, dtype=int)))
        ax.set_yticks(np.unique(np.linspace(0, values.shape[0]-1, 5, dtype=int)))
    return im


def plot_cohort(path, metric_names=None):
    plt = _pyplot()
    with np.load(path, allow_pickle=False) as z:
        names = z['metric_names'].tolist()
        selected = metric_names or ['evidence_share', 'far_history_share', 'distance_clip10',
                                    'distance_full', next(n for n in names if n.startswith('carrier_enrichment')), 'head_margin']
        fig, axes = plt.subplots(len(selected), 3, figsize=(12.5, 2.8*len(selected)), squeeze=False,
                                  layout='constrained')
        for row, name in enumerate(selected):
            i = names.index(name)
            raw = z['raw_mean'][:, i]
            finite = raw[np.isfinite(raw)]
            bound = float(np.max(np.abs(finite))) if len(finite) else 1.
            for j, label in enumerate(('N: normal', 'H: hallucinated')):
                im = heatmap(axes[row,j], raw[j], name+'\n'+label, limit=bound,
                             signed=bool(len(finite) and np.min(finite)<0))
                fig.colorbar(im, ax=axes[row,j], shrink=.75)
            n = z['matched_sources'][i]
            im = heatmap(axes[row,2], z['matched_mean'][i],
                         f'Matched H - N; sources {n.min()}..{n.max()}', signed=True)
            fig.colorbar(im, ax=axes[row,2], shrink=.75)
        fig.suptitle(Path(path).stem + '\nEach cell is one actual head; carrier_* uses carrier labels.', fontsize=12)
        destination=Path(path).with_suffix('.reads.png') if metric_names is None else Path(path).with_name(
            Path(path).stem+'.'+'_'.join(selected)+'.reads.png')
        fig.savefig(destination, dpi=135)
        plt.close(fig)
        if metric_names is not None:
            return destination
        fig, axes = plt.subplots(5, 3, figsize=(13, 15), layout='constrained')
        for s, name in enumerate(z['states']):
            for c in range(3):
                values = z['joint_raw_mean'][c,s] if c<2 else z['joint_matched_mean'][s]
                im = heatmap(axes[s,c], values, str(name)+'\n'+('carrier N','carrier H','matched carrier H - N')[c],
                             signed=c==2, limit=1 if c<2 else None, pair_heads=int(z['heads']))
                fig.colorbar(im, ax=axes[s,c], shrink=.75)
        for c, name in enumerate(('chain_matched_mean', 'chain_excess_matched_mean')):
            im = heatmap(axes[4,c], z[name], name+'\nlabels belong to the final predicted token',
                         signed=True, pair_heads=int(z['heads']))
            fig.colorbar(im, ax=axes[4,c], shrink=.75)
        axes[4,2].axis('off')
        axes[4,2].text(0, .9, 'All legal writer/reader pairs.\n\nEntry: evidence gain >= .10\nand local/self drop >= .10.\nReuse: > uniform opportunity x 2.\nNo peak-intersection selection.\n\nComplete estimates, counts,\nintervals and BY q in NPZ.\n\nCoefficient paths do not prove\nfactual mediation.', va='top', fontsize=10)
        fig.suptitle(Path(path).stem + ': entry, reuse and target-aligned two-hop structure', fontsize=12)
        fig.savefig(Path(path).with_suffix('.chains.png'), dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
        for i, label in enumerate(('attention write', 'MLP write', 'rounding remainder')):
            x = np.arange(int(z['layers']))
            axes[i].plot(x, z['writes_raw_mean'][0,i], label='N', color='#1b6e58')
            axes[i].plot(x, z['writes_raw_mean'][1,i], label='H', color='#bf2440')
            axes[i].plot(x, z['writes_matched_mean'][i], label='matched H-N', color='#334f96')
            ci = z['writes_matched_ci95'][:,i]
            axes[i].fill_between(x, ci[0], ci[1], color='#334f96', alpha=.15)
            axes[i].axhline(0, color='grey', lw=.6)
            axes[i].set(title=label, xlabel='Layer', ylabel='Observed-vs-runner margin contribution')
        axes[0].legend(fontsize=8)
        fig.suptitle('Native writes: observed token may itself be hallucinated')
        fig.savefig(Path(path).with_suffix('.writes.png'), dpi=140)
        plt.close(fig)


def backtrace(trace, query, limit=3):
    """Label-free backward path ranking at an explicitly requested query.

    Candidates use ordinary strict-history carriers and evidence-token roots,
    with all writer heads in shallower layers. No WAAD/FAI peak prerequisite.
    These are top-k display witnesses, never the full-population estimator.
    """
    start, heads = int(trace['response_start']), trace['ordinary_mass'].shape[1]
    row = query - start + 1
    if not 0 <= row < len(trace['row_position'])-1 or trace['special_mask'][query]:
        return []
    positions, weights = trace['top_position'], trace['top_attention']
    candidates = []
    for rl in range(1, positions.shape[0]):
        for rh in range(heads):
            for b, read in zip(positions[rl,rh,row,2], weights[rl,rh,row,2]):
                if b < start or b >= query or trace['special_mask'][b]:
                    continue
                slot = b-start+1
                source = positions[:rl,:,slot,1,0]
                write = weights[:rl,:,slot,1,0]
                eligible=np.flatnonzero(((source>=0)&(write>0)).ravel())
                keep=min(limit,len(eligible))
                if not keep:
                    continue
                # Only the best `limit` writers of this carrier/reader branch
                # can enter the global top `limit`; consider all writer heads
                # numerically without creating millions of Python dictionaries.
                ids=eligible[np.argpartition(write.ravel()[eligible],-keep)[-keep:]]
                for index in ids:
                    wl,wh=divmod(int(index),heads)
                    s = int(source[wl,wh])
                    if not trace['evidence_mask'][s] or trace['special_mask'][s]:
                        continue
                    candidates.append(dict(source=s, writer=[int(wl),int(wh)], carrier=int(b),
                                           reader=[rl,rh], query=int(query), predicted_token=int(query+1),
                                           write_attention=float(write[wl,wh]), read_attention=float(read),
                                           score=float(write[wl,wh]*read)))
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:limit]


def write_sample_text(path, trace, labels):
    start = int(trace['response_start'])
    spans = []
    for i, text in enumerate(trace['token_text']):
        label = int(labels[i-start]) if i>=start else -1
        cls = 'special' if trace['special_mask'][i] else ('hall' if label==1 else 'normal' if label==0 else 'prompt')
        spans.append(f'<span class="{cls}" title="absolute={i}; response={i-start}; label={label}">{html.escape(str(text))}</span>')
    text = ('<!doctype html><meta charset="utf-8"><style>body{max-width:1050px;margin:35px auto;font:16px/1.8 sans-serif}'
            '.hall{background:#ffd6df;border-bottom:2px solid #b7193d}.normal{background:#e5f3ee}'
            '.special{color:#999;text-decoration:line-through}.prompt{color:#555}pre{white-space:pre-wrap}</style>'
            f'<h2>{html.escape(str(trace["sample_id"]))}: full prompt and response</h2>'
            '<p>Red = hallucinated; green = normal; grey strike-through = excluded special token. Hover for exact token coordinates.</p>'
            '<p>Labels describe response tokens; prediction of token b uses query b-1.</p><pre>'+''.join(spans)+'</pre>')
    Path(path).with_suffix('.review.html').write_text(text, encoding='utf-8')


def plot_sample(path, layer=None, head=None, title=''):
    from .attention_audit import reconstruct_attention
    plt = _pyplot()
    with np.load(path, allow_pickle=False) as z:
        trace = dict(z)
    with np.load(Path(path).with_suffix('.audit.npz'), allow_pickle=False) as z:
        audit = dict(z)
    labels = audit['labels']
    start, layers, heads = int(trace['response_start']), trace['ordinary_mass'].shape[0], trace['ordinary_mass'].shape[1]
    queries = []
    for y in (0, 1):
        eligible = np.flatnonzero((labels==y) & audit['valid_target'])
        if len(eligible):
            queries.append(start + int(eligible[len(eligible)//2])-1)
    paths = [] if layer is not None else [dict(p, label=int(labels[p['predicted_token']-start])) for q in queries for p in backtrace(trace,q)]
    if layer is None:
        save_json(Path(path).with_suffix('.paths.json'), paths)
    if layer is not None and head is not None:
        selected = [(int(layer),int(head))]
    elif paths:
        selected = list(dict.fromkeys(tuple(p['writer']) for p in paths))[:2]
        selected += [x for x in dict.fromkeys(tuple(p['reader']) for p in paths) if x not in selected][:2]
    else:
        selected = [(0,0), (layers-1,heads-1)]
    metrics = read_metrics(trace)
    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 3.1*len(selected)), squeeze=False, layout='constrained')
    x = np.arange(len(labels))
    for row, (l,h) in enumerate(selected):
        for name in ('evidence_share','far_history_share','local_history_share'):
            axes[row,0].plot(x, metrics[name][l,h], label=name, lw=.9)
        axes[row,1].plot(x, metrics['distance_clip10'][l,h], label='WAAD10', lw=.9)
        other = axes[row,1].twinx()
        other.plot(x, audit['enrichment'][l,h,:,0], color='#805bad', label='carrier reuse enrichment', alpha=.7)
        other.set_ylabel('Carrier reuse enrichment')
        observed = audit['enrichment'][l,h,:,0]
        finite = observed[np.isfinite(observed)]
        other.set_ylim(0,max(2, float(finite.max())*1.05 if len(finite) else 2))
        other.axhline(1,color='#805bad',ls=':',lw=.6)
        matrix = reconstruct_attention(path,l,h,np.arange(len(labels)))
        matrix = np.where(~trace['special_mask'][None], matrix, np.nan)
        matrix = np.where(~trace['special_mask'][trace['row_position'][:-1]][:,None], matrix, np.nan)
        im=axes[row,2].imshow(matrix, origin='upper', aspect='auto', cmap='magma', interpolation='nearest',
                           extent=(-.5,len(trace['token_ids'])-.5,len(labels)-.5,-.5))
        fig.colorbar(im,ax=axes[row,2],shrink=.65,label='Attention (special columns hidden)')
        axes[row,2].axvline(start-.5,color='#2accba',ls='--',lw=.8)
        axes[row,2].set(xlabel='Absolute source token; dashed = response start', ylabel='Predicted response token index')
        for ax in axes[row,:2]:
            for begin,end in np.flatnonzero(np.diff(np.r_[False,labels==1,False])).reshape(-1,2):
                ax.axvspan(begin-.5,end-.5,color='#ed6480',alpha=.23)
            ax.set_xlim(-.5,len(labels)-.5)
            ax.set_xlabel('Response token index; red = H')
            ax.legend(fontsize=7,loc='upper right')
        # Label strips are placed in actual matrix query and source coordinates.
        hy = np.flatnonzero(labels==1)
        axes[row,2].scatter(np.full(len(hy),len(trace['token_ids'])-1),hy,c='#ed3159',s=12,marker='s')
        axes[row,2].scatter(start+hy,np.full(len(hy),len(labels)-1),c='#ed3159',s=12,marker='s')
        for ax in axes[row]:
            ax.set_title(f'L{l}H{h}',fontsize=10)
    fig.suptitle(title+'\nReads at predictor b-1; reuse belongs to carrier b. Full source map from saved Q/K; red strips = H.',fontsize=12)
    destination = Path(path).with_suffix('.review.png') if layer is None else Path(path).with_name(
        Path(path).stem+f'.L{layer}H{head}.review.png')
    fig.savefig(destination,dpi=140)
    plt.close(fig)
    return destination


def source_unit_paths(path, writer, reader):
    """All source-unit path coefficients for one strict cross-layer head pair."""
    if writer[0] >= reader[0]:
        raise ValueError('writer must be in a strictly shallower layer')
    with np.load(path, allow_pickle=False) as z:
        rows, special = z['row_position'], z['special_mask']
        w = ratio(z['unit_mass'][writer[0],writer[1],1:], z['ordinary_mass'][writer[0],writer[1],1:,None])
        direct = ratio(z['unit_mass'][reader[0],reader[1],:-1],z['ordinary_mass'][reader[0],reader[1],:-1,None])
        reader_mass = z['ordinary_mass'][reader[0],reader[1],:-1]
        names=z['unit_names']
    with np.load(Path(path).with_suffix('.history.npz')) as z:
        a=ratio(z[f'L{reader[0]}'][reader[1],:-1,1:],reader_mass[:,None])
    legal=(rows[:-1,None]>rows[None,1:]) & ~special[rows[None,1:]]
    weights=np.where(legal,a,0)
    paths=np.nan_to_num(weights) @ np.nan_to_num(w)
    invalid=(~np.isfinite(w)).astype(float).T @ (np.nan_to_num(weights)>0).T > 0
    missing_query=~np.isfinite(reader_mass) | (reader_mass<=0) | special[rows[:-1]]
    paths=np.where(invalid.T | missing_query[:,None],np.nan,paths)
    direct[missing_query]=np.nan
    return {'unit_names':names,'two_hop':paths,'direct':direct}


def plot_onset(path, metric, layer, head):
    plt=_pyplot()
    with np.load(path) as z:
        k=z['metric_names'].tolist().index(metric)
        x=z['onset_offsets']
        raw=z['onset_raw_mean'][:,k,layer,head]
        gap=z['onset_mean'][k,layer,head]
        ci=z['onset_ci95'][:,k,layer,head]
        count=z['onset_sources'][k,layer,head]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    axes[0].plot(x,raw[0],color='#1b6e58',label='matched normal window')
    axes[0].plot(x,raw[1],color='#bf2440',label='hallucination onset window')
    axes[0].legend(fontsize=8)
    axes[1].plot(x,gap,color='#334f96',label='H onset - N control')
    axes[1].fill_between(x,ci[0],ci[1],color='#334f96',alpha=.15)
    for ax in axes:
        ax.axvline(0,color='grey',ls='--'); ax.axhline(0,color='grey',lw=.5)
        ax.set_xlabel('Token offset from onset / matched control')
    fig.suptitle(f'{metric}: L{layer}H{head}; contributing sources {count.min()}..{count.max()}')
    destination=Path(path).with_name(Path(path).stem+f'.{metric}.L{layer}H{head}.onset.png')
    fig.savefig(destination,dpi=140); plt.close(fig)
    return destination


def write_gallery(output, entries, report):
    links = []
    for group,c in report['groups'].items():
        base = Path(c['statistics']).with_suffix('')
        links.append('<li>'+html.escape(group)+': '+ ' | '.join(
            f'<a href="{base}.{kind}.png">{kind}</a>' for kind in ('reads','chains','writes'))+'</li>')
    rows = []
    for e in entries:
        figure = f'<a href="{html.escape(e["figure"])}">figure</a>' if 'figure' in e else 'use notebook for any head'
        rows.append(f'<tr data-case="{e["answer_class"]}"><td>{html.escape(e["split"]+"/"+e["task_type"])}</td>'
                    f'<td>{html.escape(e["sample_id"])}</td><td>{e["answer_class"]}</td>'
                    f'<td>{e["normal_tokens"]}</td><td>{e["hallucinated_tokens"]}</td>'
                    f'<td><a href="{html.escape(e["text"])}">all labeled tokens</a></td><td>{figure}</td></tr>')
    content = ('<!doctype html><meta charset="utf-8"><style>body{margin:30px;font:15px/1.6 sans-serif}'
               'table{border-collapse:collapse}td,th{padding:5px 15px;border-bottom:1px solid #ddd}</style>'
               '<h1>Normal / hallucinated attention audit</h1><p>Every captured sample is listed. Positive answer = contains H; '
               'negative answer = all ordinary response tokens have known N labels.</p><ul>'+''.join(links)+'</ul>'
               '<p><a href="summary.md">Summary</a> | <a href="view_attention_audit.ipynb">Notebook: any sample, layer, head, metric</a></p>'
               '<label>Answer class <select onchange="document.querySelectorAll(\'tr[data-case]\').forEach(r=>r.hidden=this.value && r.dataset.case!==this.value)">'
               '<option value="">all</option><option>negative</option><option>positive</option><option>unknown</option></select></label>'
               '<table><thead><tr><th>Group</th><th>Sample</th><th>Answer class</th><th>N</th><th>H</th><th>Text</th><th>Head view</th></tr></thead><tbody>'
               +''.join(rows)+'</tbody></table>')
    (Path(output)/'gallery.html').write_text(content,encoding='utf-8')
