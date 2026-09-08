"""Portable event viewer: actual tokens, native read sites and message paths."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..attention_rhythm_report import jsonable


def compact(value):
    if isinstance(value,np.ndarray): return compact(value.tolist())
    if isinstance(value,(list,tuple)): return [compact(v) for v in value]
    if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
    if isinstance(value,(float,np.floating)):
        return float(f'{value:.5g}') if np.isfinite(value) else None
    return jsonable(value)


def render_event(folder,event,scan,labels,destination):
    start = int(scan['response_start']);b = int(event['event_position'])
    sites = event['event_sites'];weights = event['root_attention_sum']
    top = np.argsort(-weights)[:min(20,np.count_nonzero(weights))]
    records = []
    for layer,head,row in sites:
        records.append(dict(layer=int(layer),head=int(head),row=int(row),
                            local=scan['local_mass'][layer,head],remote=scan['remote_mass'][layer,head],
                            tv=scan['time_tv'][layer,head],gain=float(scan['remote_gain'][layer,head,row])))
    data = dict(sample='/'.join(Path(folder).parts[-3:]),position=b,start=start,rows=scan['row_position'],targets=event['target_position'],
                text=scan['token_text'],special=scan['special_mask'],labels=labels,sites=records,
                response=event['margin_response'],baseline=event['baseline_margin'],positive=event['positive_id'],negative=event['negative_id'],
                explicit=event['explicit_contrast'],norm=event['state_response_norm'],
                mlp=event['mlp_alignment'],energy=event['attention_head_energy'],
                cancellation=event['injection_cancellation'],injection=event['injection_norm'],
                roots=[dict(position=int(i),weight=float(weights[i])) for i in top],
                root_coverage=float(weights[top].sum()/weights.sum()) if weights.sum() else None)
    payload = json.dumps(compact(data),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    Path(destination).write_text(PAGE.replace('__DATA__',payload),encoding='utf-8')
    return Path(destination)


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>内部回看事件 · 消息传播</title><style>
body{font:15px/1.65 system-ui;color:#202a38;background:#f6f7f9;margin:0}main{max-width:1120px;margin:28px auto;padding:0 20px}
h1{font-size:25px}h2{font-size:19px;margin-top:28px}p{max-width:1000px}.muted{color:#556477}.card{background:white;border:1px solid #dce2ea;border-radius:8px;padding:18px;margin:14px 0}
canvas{display:block;width:100%;height:250px}.heat{height:310px}select{font:inherit;padding:5px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #e5e8ee;padding:4px 9px}
.tokens{max-height:220px;overflow:auto;background:#fff;padding:8px}.token{display:inline-block;padding:2px 3px;margin:2px;border-radius:3px;white-space:pre-wrap}.H{background:#ffd9dd}.N{background:#e6f0fc}.unknown{background:#eee}.seed{outline:2px solid #9251b5}.special{opacity:.4}#detail{position:sticky;bottom:0;background:#202a38;color:white;padding:7px 14px;font:13px/1.5 monospace;min-height:20px}
</style><main><h1 id="title"></h1><p>这里的一页是一个模型内部回看位置 b；可包含多个物理 layer/head 读取点。它不是“标注幻觉起点”的定义。原始计算使用所有远处普通来源，包括旧回答位置。</p>
<div class="card"><h2>位置与原始文本</h2><p class="muted">蓝色 N，红色 H，灰色未知；紫框是回看载体 b。标签仅用于事后查看。悬停显示绝对位置。</p><div id="tokens" class="tokens"></div></div>
<div class="card"><h2>1. 哪个 head 从近处转向远处？</h2><select id="site"></select><p id="siteinfo"></p><canvas id="attention"></canvas><p class="muted">蓝：近邻普通 attention 份额；橙：远处份额；紫：同一 head 在相邻 query token 间的 TV。横轴是 query 位置。增量比较使用同一组远处源坐标。</p>
<h2>实际远处来源</h2><p id="rootnote"></p><table><thead><tr><th>源位置</th><th>类型</th><th>原始 token</th><th>读取点 attention 之和</th></tr></thead><tbody id="roots"></tbody></table></div>
<div class="card"><h2>2. 同一次读取如何作用于后续输出？</h2><canvas id="baseline"></canvas><p class="muted">原参考计算的候选 logit 差 M。只有明确给出正确／错误候选，M&lt;0 才表示正确候选仍落后。</p><canvas id="hops"></canvas><p class="muted">蓝：0 跳；橙：1 跳；紫：2+ 跳；黑：三者之和 C。M&lt;0 且 C&gt;0 表示名义正候选仍落后，但定向增强该读取束有正向局部增益。跳数只计事件之后严格跨位置的 attention 传输。</p>
<canvas id="variants"></canvas><p class="muted">黑：完整响应；蓝：固定 Q/K 的路径；橙：不经过 MLP 分支的路径。三者使用相同原前向状态。曲线差表示路径家族作用，两个差值可能重叠。</p>
<p>默认候选是实际 token 与原 runner：正值不代表正确，负值不代表幻觉。明确提供的候选对会在悬停信息中标明。不同 token 的读出方向不同，不能把曲线下降直接叫作“事实丢失”。</p></div>
<div class="card"><h2>3. 哪些层和位置发生了转换？</h2><canvas class="heat" id="norm"></canvas><p class="muted">完整切向状态范数，颜色按 log10 缩放；白色为零。纵轴是层边界（0 为第0层输入），横轴是 query 位置。响应范数不是信息比特或正确事实的保留率。</p>
<canvas class="heat" id="mlp"></canvas><p class="muted">当前消息响应与 MLP 响应的余弦：红色同向，蓝色相反，白色无支持。方向相反需要结合最终候选作用判断，不能自动称作 MLP 覆盖证据。</p>
<table><thead><tr><th>注入层</th><th>原生消息合并范数</th><th>多头向量抵消率</th></tr></thead><tbody id="injections"></tbody></table>
<canvas class="heat" id="energy"></canvas><p class="muted">各物理 layer/head 的 WO 消息响应平方范数之和，按 log10 缩放。保留头身份；能量不等于最终读出贡献。</p></div>
<p>这是 teacher-forcing 参考计算上的局部响应审计；不能仅由这张图判断自由生成已进入自强化闭环。数值文件包含所有事件与所有实际后续目标，此页只展示一个事件。</p><div id="detail">悬停图或 token 查看坐标与数值。</div></main>
<script>const D=__DATA__;const $=id=>document.getElementById(id),colors=['#2878b8','#dd8126','#9552b2','#202a38'];
const fmt=x=>x==null?'缺失':Number(x).toPrecision(4);const detail=s=>$('detail').textContent=s;
$('title').textContent=`${D.sample} · 内部回看 b=${D.position} · ${D.sites.length} 个原生读取点`;
for(let p=D.start;p<D.text.length;p++){let span=document.createElement('span');const y=D.labels[p-D.start];span.className='token '+(y==1?'H':y==0?'N':'unknown')+(p==D.position?' seed':'')+(D.special[p]?' special':'');span.textContent=D.text[p]||'∅';span.title=`绝对位置 ${p} · 标签 ${y}`;span.onmouseenter=()=>detail(span.title);$('tokens').append(span)}
function setup(id){const c=$(id),w=c.clientWidth,h=c.clientHeight,ratio=devicePixelRatio||1;c.width=w*ratio;c.height=h*ratio;const ctx=c.getContext('2d');ctx.scale(ratio,ratio);ctx.font='11px system-ui';return {c,ctx,w,h,left:68,top:23,right:20,bottom:37}}
function chart(id,x,series,names,title,info){const o=setup(id),{c,ctx,w,h,left,top,right,bottom}=o;const good=series.flat().filter(Number.isFinite);let lo=Math.min(0,...good),hi=Math.max(0,...good);if(hi==lo)hi=lo+1;const min=x[0],max=x[x.length-1],X=t=>left+(t-min)/(max-min||1)*(w-left-right),Y=v=>h-bottom-(v-lo)/(hi-lo)*(h-top-bottom);
ctx.fillStyle='#243247';ctx.fillText(title,left,14);ctx.strokeStyle='#d8dfe8';ctx.beginPath();ctx.moveTo(left,Y(0));ctx.lineTo(w-right,Y(0));ctx.stroke();ctx.fillText(fmt(hi),4,top+8);ctx.fillText(fmt(lo),4,h-bottom);for(let k=0;k<=4;k++){let v=min+(max-min)*k/4;ctx.fillText(Math.round(v),X(v)-10,h-13)}
series.forEach((s,j)=>{ctx.strokeStyle=colors[j];ctx.lineWidth=j==3?2:1.5;ctx.beginPath();let active=false;s.forEach((v,i)=>{if(v==null||!Number.isFinite(v)){active=false;return}if(active)ctx.lineTo(X(x[i]),Y(v));else ctx.moveTo(X(x[i]),Y(v));active=true});ctx.stroke()});
c.onmousemove=e=>{const p=Math.max(0,Math.min(x.length-1,Math.round((e.offsetX-left)/(w-left-right)*(x.length-1))));detail(`${title} · 位置 ${x[p]} · ${names.map((n,i)=>n+': '+fmt(series[i][p])).join(' | ')} ${info?info(p):''}`)}}
function heat(id,a,title,diverging=false,head=false){const {c,ctx,w,h,left,top,right,bottom}=setup(id),ny=a.length,nx=a[0].length,dw=(w-left-right)/nx,dh=(h-top-bottom)/ny;const vals=a.flat().filter(v=>v!=null&&(diverging||v>0)).map(v=>diverging?v:Math.log10(v));const lo=diverging?-1:vals.reduce((u,v)=>Math.min(u,v),Infinity),hi=diverging?1:vals.reduce((u,v)=>Math.max(u,v),-Infinity);ctx.fillStyle='#243247';ctx.fillText(title,left,14);
for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const v=a[y][x];if(v==null||!diverging&&v<=0)continue;const z=diverging?v:2*(Math.log10(v)-lo)/(hi-lo||1)-1,t=Math.min(1,Math.abs(z));ctx.fillStyle=z<0?`rgb(${255-200*t},${255-130*t},255)`:`rgb(255,${255-170*t},${255-185*t})`;ctx.fillRect(left+x*dw,top+y*dh,dw+.3,dh+.3)}
ctx.fillStyle='#34435a';for(let k=0;k<=4;k++){const i=Math.round((nx-1)*k/4);ctx.fillText(head?i:D.rows[i],left+i*dw,h-13)}for(let y=0;y<ny;y+=Math.max(1,Math.floor(ny/7)))ctx.fillText(y,30,top+(y+.5)*dh);
c.onmousemove=e=>{const x=Math.max(0,Math.min(nx-1,Math.floor((e.offsetX-left)/dw))),y=Math.max(0,Math.min(ny-1,Math.floor((e.offsetY-top)/dh)));detail(`${title} · 层/边界 ${y} · ${head?'head '+x:'位置 '+D.rows[x]+' '+JSON.stringify(D.text[D.rows[x]])} · ${fmt(a[y][x])}`)}}
D.sites.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`L${s.layer} H${s.head} · 远处增量 ${fmt(s.gain)}`;$('site').add(o)});
function drawSite(){const s=D.sites[+$('site').value];$('siteinfo').textContent=`L${s.layer} H${s.head} 在 b=${D.position} 发生回看；这里比较同一个物理 head。`;chart('attention',D.rows,[s.local,s.remote,s.tv],['local','remote','相邻token TV'],'原生 attention 的时间变化',i=>JSON.stringify(D.text[D.rows[i]]))}$('site').onchange=drawSite;
function tableRow(id,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}$(id).append(tr)}
for(const r of D.roots)tableRow('roots',[r.position,r.position<D.start?'prompt':'旧回答',JSON.stringify(D.text[r.position]),fmt(r.weight)]);
$('rootnote').textContent=`展示权重最大的 ${D.roots.length} 个来源，覆盖该求和质量的 ${D.root_coverage==null?'未知':(100*D.root_coverage).toFixed(1)}%；计算使用完整来源束。这里的跨读取点求和不是概率或事实归因。`;
for(let l=0;l<D.injection.length;l++)if(D.injection[l]>0||D.cancellation[l]!=null)tableRow('injections',[l,fmt(D.injection[l]),fmt(D.cancellation[l])]);
function draw(){drawSite();const begin=D.targets.findIndex(t=>t>=D.position+1),x=D.targets.slice(begin),sum=(a)=>a[0].map((_,i)=>a.reduce((v,s)=>v+s[i],0));const total=D.response.map(sum),info=i=>{const j=i+begin,t=D.targets[j];return `${JSON.stringify(D.text[t])} · 标签 ${D.labels[t-D.start]} · ${D.explicit[j]?'明确候选':'observed/runner'} IDs ${D.positive[j]} / ${D.negative[j]}`};
chart('baseline',x,[D.baseline.slice(begin)],['M'],'原参考计算的候选差 M',info);
chart('hops',x,[...D.response[0],total[0]].map(a=>a.slice(begin)),['0跳','1跳','2+跳','总和'],'后续目标的候选 logit 差响应',info);
chart('variants',x,[total[1],total[2],Array(x.length).fill(null),total[0]].map((a,i)=>i==2?a:a.slice(begin)),['固定QK','无MLP路径','空','完整'],'路径家族的作用',info);
heat('norm',D.norm,'状态响应范数 · log10');heat('mlp',D.mlp,'MLP 响应方向 · cosine',true);heat('energy',D.energy,'物理 head 响应能量 · log10',false,true)}draw();window.addEventListener('resize',draw);
</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,required=True,help='OUTPUT/samples/split/task/id')
    p.add_argument('--position',type=int,required=True,help='absolute query/carrier position b, not response-row index')
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    from .event_report import labels_for
    with np.load(args.sample/'scan.npz',allow_pickle=False) as data: scan = dict(data)
    with np.load(args.sample/f'event_{args.position}.npz',allow_pickle=False) as data: event = dict(data)
    destination = args.output or args.sample/f'event_{args.position}.html'
    print(render_event(args.sample,event,scan,labels_for(args.sample,scan),destination))


if __name__=='__main__': main()
