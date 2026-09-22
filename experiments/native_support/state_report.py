"""Report the latent memory decision alongside ranking gains and recovery."""

from html import escape

from .filter_report import comparison_table
from .report import curves, metric_table


def state_table(rows, baseline, candidate):
    fields = (baseline, "route_mean", candidate, "reset_probability", "expected_run_length")
    headers = ("当前路由", "普通均值", "候选分数", "切换概率", "期望段长")
    if candidate == "joint_observed":
        fields += ("joint_state", "prior_pull", "observed_current_weight")
        headers += ("原后验均值", "直接先验拉动", "当前观测权重")
    else:
        fields += ("reset_contribution", "continuation_contribution", "state_route_sd")
        headers += ("新段贡献", "延续贡献", "状态标准差")
    content = ['<table><tr><th>t</th><th>token</th>' + ''.join(f'<th>{name}</th>' for name in headers) + '</tr>']
    for row in rows:
        cells = ''.join(f'<td>{row[name]:.4f}</td>' for name in fields)
        content.append(f'<tr><td>{row["target"]}</td><td class="token">{escape(row["token"])}</td>{cells}</tr>')
    return ''.join(content) + '</table>'


def write_state_report(path, rows, summary, evaluation, comparisons):
    candidate = summary["candidate_method"]
    content = [
        '<!doctype html><meta charset="utf-8"><title>Joint switching state</title>',
        '<style>body{font:15px system-ui;max-width:1450px;margin:30px auto;padding:0 16px}',
        'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}',
        'th{position:sticky;top:0;background:white}svg{width:100%;height:180px}.token{white-space:pre-wrap}</style>',
        '<h1>多观测切换状态模型</h1>',
        '<p>功能路由给出风险方向，读取路由与熵参与旧状态/新状态的联合概率比较。',
        f'当前候选 {escape(candidate)}；分数不是幻觉概率，统计切换不等于语义重锚。</p>',
        f'<p>参考方式：{escape(summary["reference_mode"])}；先验期望段长 {summary["window"]}，不截断实际段长。',
        '首行切换概率为 1 是初始化。先验的同 source 排除名单保存在 scoring_protocol.json。</p>',
        '<p><a href="evaluation.json">完整评价</a> · <a href="comparisons.csv">增量</a> · ',
        '<a href="onsets.csv">起点排名</a> · <a href="recovery.csv">错误后的正常词</a> · ',
        '<a href="high_risk_normals.csv">高风险正常词</a></p>',
    ]
    if candidate == "joint_observed":
        content.append('<p>候选固定原状态长度后验，仅去掉最终读出的直接先验收缩。'
                       '参考统计仍影响分段。先验拉动=原后验均值−候选；其正负不是错误标签。</p>')
    if evaluation["status"] == "evaluated":
        content.extend((metric_table(evaluation["methods"]), comparison_table(comparisons)))
    else:
        content.append('<p>本次缺少真实标注，没有新 AUROC/AP；旧有效评价文件保留。</p>')
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        content.extend((f'<h2>{escape(identity)}</h2><p>蓝：当前候选；橙：普通均值。</p>',
                        curves(selected, (candidate, "route_mean")), state_table(selected, summary["raw_baseline"], candidate)))
    path.write_text('\n'.join(content), encoding='utf-8')
