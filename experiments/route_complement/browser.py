"""Self-contained selection of physical heads, token curves and prompt addresses."""

import argparse
import base64
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output / 'diagnostics'
    values = json.loads((directory / 'head_browser_data.json').read_text())
    compact = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()
    payload = base64.b64encode(gzip.compress(compact)).decode()
    page = '''<!doctype html><meta charset="utf-8"><title>完整物理头局部对照</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1200px;padding:20px;color:#243044}select,input{font:inherit;margin:8px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:8px}svg{width:100%;height:180px}article{margin:25px 0}.note{background:#edf1f6;padding:15px}</style>
<h1>自然采样：逐头读取、消息与使用</h1>
<p class="note">只评价已有四处局部声明。supported 不表示整答正确。所有1024头均可查看；选择头只是展示，不参与检测。每对片段起点相同，但文字、长度、前缀不同，不把差异称为因果证明。来源地址无需人工实体或证据标注。</p>
<label>题目<select id="case"></select></label><label>层<input id="layer" type="number" min="0" max="31" value="16"></label><label>头<input id="head" type="number" min="0" max="31" value="0"></label><label>观测<select id="field"></select></label>
<div id="content"></div><script type="module">
const packed=Uint8Array.from(atob('PAYLOAD'),c=>c.charCodeAt(0));
const decoded=await new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream('gzip'))).text();const data=JSON.parse(decoded);
const byId=id=>document.getElementById(id);
for(const c of [...new Set(data.rows.map(r=>r.case))]){const o=document.createElement('option');o.value=c;o.textContent=c;byId('case').appendChild(o)}
data.fields.forEach((name,i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;byId('field').appendChild(o)});
function render(){const root=byId('content');root.replaceChildren();const l=+byId('layer').value,h=+byId('head').value,f=+byId('field').value;if(l<0||l>31||h<0||h>31)return;
const selected=data.rows.filter(r=>r.case===byId('case').value);const together=selected.flatMap(r=>r.values[l][h][f]);const min=Math.min(...together),max=Math.max(...together),range=Math.max(max-min,.01);
for(const row of selected){const a=document.createElement('article');const title=document.createElement('h2');title.textContent=row.side+' · '+row.key+` · 共同纵轴 ${min.toFixed(3)} 至 ${max.toFixed(3)}`;a.appendChild(title);const y=row.values[l][h][f];
const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 1000 180');const poly=document.createElementNS(svg.namespaceURI,'polyline');poly.setAttribute('points',y.map((v,i)=>`${20+i*960/Math.max(y.length-1,1)},${150-120*(v-min)/range}`).join(' '));poly.setAttribute('fill','none');poly.setAttribute('stroke',row.side==='supported'?'#247a6a':'#b1454a');poly.setAttribute('stroke-width','3');svg.appendChild(poly);a.appendChild(svg);
const table=document.createElement('table');for(let i=0;i<y.length;i++){const tr=document.createElement('tr');for(const v of [row.positions[i],row.text[i],y[i].toFixed(5)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}table.appendChild(tr)}a.appendChild(table);
for(const kind of ['read','use']){const p=document.createElement('p');p.textContent=(kind==='read'?'来源条件 attention 地址':'来源条件 |门导数| 地址')+'：';const weights=row[kind][l][h];const indices=weights.map((v,i)=>i).sort((i,j)=>weights[j]-weights[i]).slice(0,8);for(const i of indices){const s=document.createElement('span');s.textContent=` [${i}] ${row.prompt.slice(Math.max(0,i-3),i+4).join('')} (${weights[i].toFixed(3)}) `;p.appendChild(s)}a.appendChild(p)}root.appendChild(a)}}
for(const id of ['case','layer','head','field'])byId(id).onchange=render;render();</script>'''.replace('PAYLOAD', payload)
    (directory / 'HEAD_BROWSER.html').write_text(page)


if __name__ == '__main__':
    main()
