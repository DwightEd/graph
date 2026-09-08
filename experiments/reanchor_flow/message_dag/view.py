"""Portable target graphs with origin/head controls and N/H token labels."""
from html import escape
import json
from pathlib import Path

import numpy as np


def render_target(folder, target):
    folder = Path(folder)
    with np.load(folder/'meta.npz') as f: meta=dict(f)
    with np.load(folder/f'target_{target}.npz') as f: graph=dict(f)
    start=int(meta['response_start'])
    labels=np.full(len(meta['token_ids'])-start,-1,int)
    if (folder/'labels.npz').exists():
        with np.load(folder/'labels.npz') as f: labels=f['labels']
    parts=[]
    for i,token in enumerate(meta['token_text']):
        y=labels[i-start] if i>=start else -1
        color='#ddd' if meta['special_mask'][i] else '#ffd5d5' if y==1 else '#d9efdf' if y==0 else 'transparent'
        parts.append(f'<span style="background:{color}" title="token={i}; label={y}">{escape(str(token))}</span>')
    qrow=int(graph['query'])-int(meta['row_position'][0])
    payload=dict(tokens=meta['token_text'].tolist(),start=start,labels=labels.tolist(),special=meta['special_mask'].tolist(),
                 target=int(target),layers=int(meta['layers']),heads=int(meta['heads']),
                 source_names=graph['source_names'].tolist(),source_kinds=graph['source_kinds'].tolist(),
                 edges=graph['edge_index'].tolist(),credit=graph['edge_source_credit'].tolist(),
                 attention=graph['edge_attention'].tolist(),value_norm=graph['edge_value_norm'].tolist(),
                 source_output=graph['source_output'].tolist(),root_credit=graph['root_credit'].tolist(),
                 eligible=graph['eligible_absolute_by_source'].tolist(),
                 trajectory=graph['node_input'][:,:,qrow].tolist())
    data=json.dumps(payload,ensure_ascii=True).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>Message DAG</title>
<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;line-height:1.5}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}svg{background:#fafafa;width:100%;min-width:900px}select,input{margin:8px}#canvas{overflow:auto}.note{color:#555}</style>
<h1>来源 → 消息变换／聚合 → 同一输出目标</h1>
<p id="identity"></p><p>全文：红=幻觉 token，绿=正常，灰=特殊。图节点是输入后状态；query q 预测 token q+1。</p>
<pre>TEXT</pre>
<p>蓝边=支持当前 observed-vs-runner 判断，橙边=反对；不是事实真假。prompt 菱形为各层边界 V，灰线仅表示原生残差／MLP 通道存在，不表示选中来源实际经过它。</p>
<label>来源 <select id="origin"></select></label><label>head <select id="head"></select></label>
<label>显示边数 <input id="budget" type="number" min="1" max="200" value="50"></label>
<p id="coverage" class="note"></p><div id="canvas"></div>
<h2>来源／出口闭合</h2><table><thead><tr><th>来源</th><th>源边界贡献</th><th>目标出口贡献</th></tr></thead><tbody id="roots"></tbody></table>
<h2>预测位置跨层净贡献</h2><p>同一目标、同一来源；包含所有后续路径。符号不代表事实正确性。</p><div id="trajectory"></div>
<h2>保存的显示边</h2><table><thead><tr><th>层／head</th><th>来源</th><th>接收状态</th><th>attention</th><th>原生 V 范数</th><th>同目标贡献</th></tr></thead><tbody id="edges"></tbody></table>
<script>
const D=PAYLOAD,NS='http://www.w3.org/2000/svg';
function el(tag,attrs={},text){const x=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);if(text!==undefined)x.textContent=text;return x;}
function cellrow(body,values){const r=document.createElement('tr');for(const v of values){const c=document.createElement('td');c.textContent=v;r.appendChild(c);}body.appendChild(r);}
const origin=document.getElementById('origin'),head=document.getElementById('head');
for(const[value,label]of [['all','全部来源净贡献'],['material','全部材料单位'],...D.source_names.map((s,i)=>[String(i),s])]){const o=document.createElement('option');o.value=value;o.textContent=label;origin.appendChild(o);}
for(const value of ['all',...Array.from({length:D.heads},(_,i)=>String(i))]){const o=document.createElement('option');o.value=value;o.textContent=value==='all'?'全部物理 head':'H'+value;head.appendChild(o);}
document.getElementById('identity').textContent='目标 token '+D.target+' ('+D.tokens[D.target]+'); predictor q='+(D.target-1)+'. 分析不使用标签，所有边参与计算。';
const roots=document.getElementById('roots');D.source_names.forEach((s,i)=>cellrow(roots,[s,D.root_credit[i].toPrecision(5),D.source_output[i].toPrecision(5)]));
function show(){
 const gs=D.source_names.map((_,i)=>i).filter(i=>origin.value==='all'||(origin.value==='material'?D.source_kinds[i]==='material':i===Number(origin.value)));
 const sum=(v)=>gs.reduce((a,i)=>a+v[i],0),gross=(v)=>gs.reduce((a,i)=>a+Math.abs(v[i]),0);
 const candidates=D.edges.map((e,i)=>({e,c:sum(D.credit[i]),gross:gross(D.credit[i]),a:D.attention[i],v:D.value_norm[i]})).filter(x=>head.value==='all'||x.e[1]===Number(head.value));
 candidates.sort((a,b)=>b.gross-a.gross);const edges=candidates.slice(0,Math.max(1,Math.min(200,Number(document.getElementById('budget').value))));
 const denominator=gs.reduce((s,i)=>s+D.eligible[i],0),display=edges.reduce((s,x)=>s+x.gross,0);
 document.getElementById('coverage').textContent='显示 '+edges.length+' 条保存的候选边；覆盖该来源全部普通严格过去边绝对贡献的 '+(denominator?(100*display/denominator).toFixed(1)+'%':'NA')+'。显示缺席不表示没有路径；截断不影响数值传播。';
 const columns=new Map();for(const {e:[l,h,s,q]}of edges){columns.set((s<D.start?'b':'r')+s,s);columns.set('r'+q,q);}columns.set('r'+(D.target-1),D.target-1);
 const keys=[...columns.keys()].sort((a,b)=>columns.get(a)-columns.get(b)||a.localeCompare(b));
 const X=k=>65+keys.indexOf(k)*Math.max(40,1200/Math.max(1,keys.length-1)),Y=stage=>50+stage*25;
 const width=Math.max(1400,keys.length*40+150),height=Y(2*D.layers)+160,svg=el('svg',{viewBox:`0 0 ${width} ${height}`});
 const defs=el('defs'),marker=el('marker',{id:'arrow',viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:6,markerHeight:6,markerUnits:'userSpaceOnUse',orient:'auto'});marker.appendChild(el('path',{d:'M0,0 L8,4 L0,8 Z',fill:'#555'}));defs.appendChild(marker);svg.appendChild(defs);
 for(let l=0;l<=D.layers;l++){svg.appendChild(el('line',{x1:45,x2:width-10,y1:Y(2*l),y2:Y(2*l),stroke:'#e6e6e6'}));svg.appendChild(el('text',{x:2,y:Y(2*l)+4,'font-size':11},'L'+l));if(l<D.layers)svg.appendChild(el('text',{x:2,y:Y(2*l+1)+4,'font-size':10,fill:'#888'},'A'+l));}
 for(const key of keys){const pos=columns.get(key),x=X(key);if(key[0]==='r')svg.appendChild(el('line',{x1:x,x2:x,y1:Y(0),y2:Y(2*D.layers),stroke:'#ccc','stroke-dasharray':'2,3'}));
  const color=D.special[pos]?'#777':pos>=D.start?(D.labels[pos-D.start]===1?'#b83b3b':D.labels[pos-D.start]===0?'#287449':'#555'):'#555';
  svg.appendChild(el('text',{x,y:height-120,fill:color,'font-size':11,transform:`rotate(55 ${x} ${height-120})`},(key[0]==='b'?'boundary ':'')+pos+': '+D.tokens[pos].slice(0,16)));}
 const maximum=Math.max(1e-12,...edges.map(x=>x.gross));
 for(const {e:[l,h,s,q],c,gross:g}of edges){const sx=X((s<D.start?'b':'r')+s),tx=X('r'+q),sy=Y(2*l),ty=Y(2*l+1),color=c>=0?'#267db3':'#d17a21';
  const line=el('path',{d:`M ${sx} ${sy} Q ${(sx+tx)/2} ${sy-16} ${tx} ${ty}`,fill:'none',stroke:color,'stroke-width':.5+4*g/maximum,opacity:.75,'marker-end':'url(#arrow)'});line.appendChild(el('title',{},`L${l}H${h}: ${s} -> ${q}; contribution=${c}`));svg.appendChild(line);
  svg.appendChild(el(s<D.start?'rect':'circle',s<D.start?{x:sx-3,y:sy-3,width:6,height:6,transform:`rotate(45 ${sx} ${sy})`,fill:color}:{cx:sx,cy:sy,r:2.5,fill:color}));svg.appendChild(el('circle',{cx:tx,cy:ty,r:2.5,fill:color}));}
 document.getElementById('canvas').replaceChildren(svg);
 const body=document.getElementById('edges');body.replaceChildren();for(const {e:[l,h,s,q],c,a,v}of edges)cellrow(body,[`L${l}H${h}`,s+': '+D.tokens[s],q+': '+D.tokens[q],a.toPrecision(4),v.toPrecision(4),c.toPrecision(5)]);
 const trajectory=Array.from({length:D.layers+1},(_,l)=>gs.reduce((s,g)=>s+D.trajectory[g][l],0)),scale=Math.max(1e-9,...trajectory.map(Math.abs));
 const chart=el('svg',{viewBox:'0 0 1000 210'});chart.appendChild(el('line',{x1:30,x2:970,y1:100,y2:100,stroke:'#bbb'}));
 chart.appendChild(el('polyline',{points:trajectory.map((v,l)=>`${30+940*l/D.layers},${100-80*v/scale}`).join(' '),fill:'none',stroke:'#267db3','stroke-width':2}));
 chart.appendChild(el('text',{x:30,y:200,'font-size':14},'L0 → L'+D.layers+'; scale ±'+scale.toPrecision(3)));document.getElementById('trajectory').replaceChildren(chart);
}
origin.addEventListener('change',show);head.addEventListener('change',show);document.getElementById('budget').addEventListener('input',show);show();
</script>'''.replace('TEXT',''.join(parts)).replace('PAYLOAD',data)
    path=folder/f'target_{target}.html';path.write_text(html,encoding='utf-8')
    return path


def render_sample(folder, targets):
    folder=Path(folder)
    with np.load(folder/'meta.npz') as f: meta=dict(f)
    start=int(meta['response_start']);tokens=meta['token_text']
    from .selection import content_positions, uniform
    content=set(content_positions(meta).tolist())
    labels=np.full(len(tokens)-start,-1,int)
    if (folder/'labels.npz').exists():
        with np.load(folder/'labels.npz') as f: labels=f['labels']
    # Small comparisons expose every target. Large full-token runs keep a
    # bounded HTML subset; all other saved targets remain viewable in notebook.
    chosen=list(targets) if len(targets)<=24 else uniform(targets,22)
    preferred=sorted(targets,key=lambda t:(t not in content,labels[t-start]!=1,t==start,t))
    chosen=sorted(set(chosen+preferred[:2])) if len(targets)>24 else chosen
    for t in chosen: render_target(folder,t)
    first=next(t for t in preferred if t in chosen)
    options=''.join(f'<option value="target_{t}.html"'+(' selected' if t==first else '')+f'>token {t}: {escape(repr(str(tokens[t])))} ({labels[t-start]})</option>' for t in chosen)
    rows=''.join(f'<tr><td>{t}</td><td>{escape(repr(str(tokens[t])))}</td><td>{labels[t-start]}</td><td>'
                 +(f'<a href="target_{t}.html">图</a> · ' if t in chosen else 'notebook 可查看 · ')
                 +f'<a href="target_{t}.npz">npz</a></td></tr>' for t in targets)
    page=folder/'index.html'
    page.write_text('<!doctype html><meta charset="utf-8"><h1>消息 DAG</h1><p>标签：1=H，0=N，-1=未知。默认选择内容候选目标；不代表该词已被确认为事实。'
        '小批的所有目标均可打开；超过24个目标时限制预生成页面，其余可用 notebook 即时查看。</p>'
        +'<select onchange="document.getElementById(\'frame\').src=this.value">'+options+'</select>'
        +f'<iframe id="frame" src="target_{first}.html" style="width:100%;height:1100px;border:0"></iframe>'
        +'<table><tr><th>token</th><th>text</th><th>label</th><th>graph</th></tr>'+rows+'</table>',encoding='utf-8')
    return page
