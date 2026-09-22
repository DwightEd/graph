"""Small self-contained token report; no inferred truth labels or tuned threshold."""

from html import escape

import numpy as np


def write_report(path, rows):
    content = [
        '<!doctype html><meta charset="utf-8"><title>Native support</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px}',
        'table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}',
        'th{position:sticky;top:0;background:white}td{font-variant-numeric:tabular-nums}',
        'svg{width:100%;height:180px}.token{white-space:pre-wrap}p{line-height:1.6}</style>',
        '<h1>原生来源支持检测</h1>',
        '<p>分数越高，prompt 与此前来源支持的净值越低。分数不是幻觉概率；',
        'prompt 来源不等于语义适用。橙线是局部项，蓝线是包含历史复用的总分。',
        '熵、回看与 FFN 仅为解释字段。没有读取正确答案、角色表或自然幻觉标签。</p>',
    ]
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        content.append(f'<h2>{escape(identity)}</h2>')
        content.append(curves(selected))
        content.append('<table><tr><th>t</th><th>token</th><th>risk</th><th>局部</th>'
                       '<th>历史</th><th>熵</th><th>惊讶度</th><th>回看头数</th></tr>')
        for row in selected:
            numbers = ("risk", "direct_risk", "inherited_risk", "entropy", "surprisal")
            cells = ''.join(f'<td>{row[name]:.4f}</td>' for name in numbers)
            content.append(f'<tr><td>{row["target"]}</td><td class="token">'
                           f'{escape(row["token"])}</td>{cells}<td>{row["read_event_heads"]}</td></tr>')
        content.append('</table>')
    path.write_text('\n'.join(content), encoding='utf-8')


def curves(rows):
    values = np.asarray([[row["risk"], row["direct_risk"]] for row in rows])
    scale = max(float(np.abs(values).max()), 0.01)
    horizontal = np.linspace(15, 1185, len(rows))
    lines = ['<svg viewBox="0 0 1200 180" role="img" aria-label="Token risk trajectory">',
             '<path d="M15 90H1185" stroke="#ddd"/>']
    for index, color in enumerate(("#2266bb", "#cf7925")):
        points = ' '.join(f'{x:.1f},{90 - 75 * y / scale:.1f}' for x, y in zip(horizontal, values[:, index]))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    lines.append(f'<text x="15" y="15" font-size="12">±{scale:.3g}</text></svg>')
    return ''.join(lines)
