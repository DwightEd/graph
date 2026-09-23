"""Show ranking gains together with the cost of adding direct risk observations."""

from html import escape

from .filter_report import comparison_table
from .report import curves, metric_table


def alarm_table(audit):
    content = [('<h2>固定参考阈值：补检与误报</h2><table><tr><th>对照</th><th>阶段</th>'
                '<th>补回错误</th><th>丢失错误</th><th>新增正常告警</th><th>移除正常告警</th></tr>')]
    fields = ("recovered_error_tokens", "lost_error_tokens", "added_normal_alarms", "removed_normal_alarms")
    for control, phases in audit.get("comparisons", {}).items():
        for phase, item in phases.items():
            cells = ''.join(f'<td>{item[name]}</td>' for name in fields)
            content.append(f'<tr><td>{escape(control)}</td><td>{escape(phase)}</td>{cells}</tr>')
    return ''.join(content) + '</table>'


def write_fusion_report(path, rows, summary, evaluation, comparisons, audit):
    content = [
        '<!doctype html><meta charset="utf-8"><title>Native risk fusion</title>',
        '<style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 16px}',
        'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}',
        'th{position:sticky;top:0;background:white}svg{width:100%;height:180px}.token{white-space:pre-wrap}</style>',
        '<h1>持续路由与当前观测的直接风险融合</h1>',
        '<p>冻结外部参考分位 → max(路由状态分位, 当前注意力分位, 当前熵分位)。',
        '风险方向固定，不读取金标边界，不以 AUROC 选择权重。分位不是幻觉概率，也不是显著性 p 值。</p>',
        '<p>主对照为仅路由状态和原始路由；instant_envelope 使用当前路由代替状态，检验时序增量。',
        '高熵正常词可能增加误报，不能只看首错提升。饱和于参考极值会出现并列排名。</p>',
        '<p>阈值为各方法在无标签参考集上的来源等权 95% 分位，严格大于才告警。',
        '这不是正常词 FPR=5% 的保证；参考集未筛除错误，参考先验和分位使用同一批来源。</p>',
        '<p><a href="evaluation.json">完整排名评价</a> · <a href="comparisons.csv">排名增量</a> · ',
        '<a href="complementarity.json">补检与误报统计</a> · <a href="alarm_changes.csv">全部告警变化与起点</a> · ',
        '<a href="onsets.csv">起点排名</a> · <a href="high_risk_normals.csv">高风险正常词</a> · ',
        '<a href="recovery.csv">错误后的正常词</a></p>',
    ]
    if evaluation["status"] == "evaluated":
        content.extend((metric_table(evaluation["methods"]), comparison_table(comparisons), alarm_table(audit)))
    else:
        content.append('<p>缺少标注；无标签分数已保存，本次没有 AUROC/AP。</p>')
    fields = ("percentile_route_state", "percentile_attention", "percentile_entropy", "risk_envelope")
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        content.extend((f'<h2>{escape(identity)}</h2><p>蓝：融合分数；橙：路由状态分位。</p>',
                        curves(selected, ("risk_envelope", "percentile_route_state")),
                        ('<table><tr><th>t</th><th>token</th><th>路由状态分位</th><th>注意力分位</th>'
                         '<th>熵分位</th><th>融合</th><th>最大分位来源（含并列）</th></tr>')))
        for row in selected:
            cells = ''.join(f'<td>{row[name]:.4f}</td>' for name in fields)
            content.append(f'<tr><td>{row["target"]}</td><td class="token">{escape(row["token"])}</td>'
                           f'{cells}<td>{escape(row["dominant_views"])}</td></tr>')
        content.append('</table>')
    path.write_text('\n'.join(content), encoding='utf-8')
