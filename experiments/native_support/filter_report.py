"""A focused ranking report: baseline, temporal gain, head-state gain and recovery."""

from html import escape

from .report import curves, metric_table


def comparison_table(comparisons):
    rows = [('<h2>同样本增量</h2><table><tr><th>候选 − 对照</th><th>阶段</th>'
             '<th>ΔAUROC</th><th>ΔAP</th><th>Δ同答AUROC</th></tr>')]
    for item in comparisons.get("comparisons", []):
        values = (item["delta_auroc"], item["delta_ap"], item["delta_within_auroc"])
        cells = ''.join('<td>缺测</td>' if value is None else f'<td>{value:+.4f}</td>' for value in values)
        rows.append(f'<tr><td>{escape(item["candidate"])} − {escape(item["control"])}</td>'
                    f'<td>{escape(item["phase"])}</td>{cells}</tr>')
    return ''.join(rows) + '</table>'


def answer_table(rows, baseline):
    content = [('<table><tr><th>t</th><th>token</th><th>当前路由</th><th>普通均值</th>'
                '<th>状态滤波</th><th>当前权重</th><th>有效词数</th><th>熵</th></tr>')]
    fields = (baseline, "route_mean", "route_state_filter", "filter_current_weight", "filter_effective_tokens", "entropy")
    for row in rows:
        cells = ''.join(f'<td>{row[name]:.4f}</td>' for name in fields)
        content.append(f'<tr><td>{row["target"]}</td><td class="token">{escape(row["token"])}</td>{cells}</tr>')
    return ''.join(content) + '</table>'


def write_filter_report(path, rows, summary, evaluation, comparisons):
    baseline = summary["primary_baseline"]
    content = [
        '<!doctype html><meta charset="utf-8"><title>Native routing filter</title>',
        '<style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 16px}',
        'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}',
        'th{position:sticky;top:0;background:white}svg{width:100%;height:180px}.token{white-space:pre-wrap}</style>',
        '<h1>原生路由：当前分数与因果状态滤波</h1>',
        f'<p>基线 {escape(baseline)}；候选 route_state_filter；窗口 {summary["window"]}。',
        '全部规则在评价前固定，不根据本次结果自动选择方法。状态相似性尚不是幻觉机制证据。</p>',
        '<p>先看候选是否超过当前路由，再看是否超过普通均值；同时检查首错、同答排序和恢复位置。',
        '四答诊断不能确认稳定提升。阴性结果保留，不翻转方向。</p>',
        '<p><a href="evaluation.json">完整评价</a> · <a href="comparisons.csv">增量</a> · ',
        '<a href="onsets.csv">起点排名</a> · <a href="recovery.csv">错误后的正常词</a> · ',
        '<a href="high_risk_normals.csv">高风险正常词</a></p>',
    ]
    if evaluation["status"] == "evaluated":
        content.extend((metric_table(evaluation["methods"]), comparison_table(comparisons)))
    else:
        content.append('<p>缺少真实标注，分数已保存，本次没有 AUROC/AP。</p>')
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        content.extend((f'<h2>{escape(identity)}</h2><p>蓝：状态滤波；橙：当前路由。</p>',
                        curves(selected, ("route_state_filter", baseline)), answer_table(selected, baseline)))
    path.write_text('\n'.join(content), encoding='utf-8')
