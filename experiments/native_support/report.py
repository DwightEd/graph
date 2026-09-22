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


def curves(rows, names=("risk", "direct_risk")):
    values = np.asarray([[row[name] for name in names] for row in rows])
    scale = max(float(np.abs(values).max()), 0.01)
    horizontal = np.linspace(15, 1185, len(rows))
    lines = ['<svg viewBox="0 0 1200 180" role="img" aria-label="Token risk trajectory">',
             '<path d="M15 90H1185" stroke="#ddd"/>']
    for index, color in enumerate(("#2266bb", "#cf7925")):
        points = ' '.join(f'{x:.1f},{90 - 75 * y / scale:.1f}' for x, y in zip(horizontal, values[:, index]))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    lines.append(f'<text x="15" y="15" font-size="12">±{scale:.3g}</text></svg>')
    return ''.join(lines)


def write_comparison_report(path, rows, primary, evaluation):
    content = [
        '<!doctype html><meta charset="utf-8"><title>Route comparison</title>',
        '<style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 16px}',
        'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}',
        'th{position:sticky;top:0;background:white}svg{width:100%;height:180px}.token{white-space:pre-wrap}</style>',
        '<h1>读取结构与输出作用：同样本比较</h1>',
        f'<p>固定主基线：{escape(primary)}。历史传播保留为 v1 对照；各列独立评分，没有标签调参或组合。</p>',
        '<p>functional collapse 使用普通 prompt 和因果位置校正；offline 列使用历史完整 prompt/整段长度协议。',
        '来源块只是输入分区，不是已核验适用事实。没有校准告警阈值。完整加权指标见 evaluation.json。</p>',
        '<p><a href="onsets.csv">起点逐词排序</a> · <a href="high_risk_normals.csv">高风险正常词</a></p>',
    ]
    if evaluation["status"] == "evaluated":
        content.append(metric_table(evaluation["methods"]))
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        names = (primary, "attention_displacement" if primary == "routing_imbalance" else "prompt_attention_displacement")
        content.extend((f'<h2>{escape(identity)}</h2><p>蓝：功能消息路由；橙：attention 路由。</p>', curves(selected, names)))
        content.append('<table><tr><th>t</th><th>token</th><th>路由主分数</th><th>prompt读取</th>'
                       '<th>峰窗口正写入</th><th>峰窗口负写入</th><th>FFN负写入</th><th>熵</th><th>v1总分</th></tr>')
        for row in selected:
            fields = (primary, "prompt_read", "focus_positive_write", "focus_negative_write", "ffn_negative", "entropy", "risk")
            cells = ''.join(f'<td>{row[name]:.4f}</td>' for name in fields)
            content.append(f'<tr><td>{row["target"]}</td><td class="token">{escape(row["token"])}</td>{cells}</tr>')
        content.append('</table>')
    path.write_text('\n'.join(content), encoding='utf-8')


def metric_table(methods):
    phases = ("all_error", "span_onset_vs_normal", "first_error_vs_normal", "continuation_vs_normal")
    result = [('<h2>AUROC（相同标注；缺测保留为空）</h2><table><tr><th>方法</th><th>全错误</th>'
               '<th>span起点</th><th>每答首错</th><th>延续</th><th>AP</th><th>同答AUROC</th></tr>')]
    for name, metrics in methods.items():
        values = [metrics[phase]["auroc"] for phase in phases]
        values.extend((metrics["all_error"]["ap"], metrics["all_error"]["within_answer"]["pair_weighted_auroc"]))
        cells = ''.join('<td>缺测</td>' if value is None else f'<td>{value:.4f}</td>' for value in values)
        result.append(f'<tr><td>{escape(name)}</td>{cells}</tr>')
    return ''.join(result) + '</table>'
