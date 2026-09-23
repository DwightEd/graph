"""One report links detection metrics to saved head-level native observations."""

from html import escape

import numpy as np
from state_audit.storage import read_arrays, read_json

from .report import curves, metric_table


def heatmap(values, title):
    rows, columns = values.shape
    scale = max(float(np.max(np.abs(values))), 1e-12)
    cells = []
    for layer in range(rows):
        for head in range(columns):
            value = float(values[layer, head])
            opacity = abs(value) / scale
            color = "#2055bf" if value >= 0 else "#ba352e"
            cells.append(f'<rect x="{head * 10}" y="{layer * 10}" width="9" height="9" '
                         f'fill="{color}" opacity="{opacity:.3f}"><title>L{layer} H{head}: {value:.5g}</title></rect>')
    return f'<figure><figcaption>{escape(title)}（纵轴层、横轴头）</figcaption><svg viewBox="0 0 {columns * 10} {rows * 10}" style="max-width:360px">{"".join(cells)}</svg></figure>'


def write_report(output, rows, protocol, evaluation):
    destination = output / "state_dynamics"
    content = ['<!doctype html><meta charset="utf-8"><title>Offline state dynamics</title>',
               '<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}svg{width:100%;max-height:340px}figure{display:inline-block;width:30%;vertical-align:top}</style>',
               '<h1>离线原生响应状态检测</h1><p>分数为历史主导统计模式的后验，不是校准后的幻觉概率。',
               '完整回答参与评分；头间结构先保留，再用训练来源的固定低维表示拟合。高分本身不证明错误。</p>',
               '<p><a href="evaluation.json">AUROC / AP</a> · <a href="comparisons.csv">同样本增益</a> · ',
               '<a href="tokens.csv">全部 token</a> · <a href="onsets.csv">起点</a> · ',
               '<a href="high_risk_normals.csv">高分正常词</a> · <a href="recovery.csv">恢复位置</a> · ',
               '<a href="observations.csv">回看/不确定性/复用轨迹</a> · ',
               '<a href="projection_error.json">留出来源投影误差</a></p>']
    content.append(metric_table(evaluation["methods"]) if evaluation["status"] == "evaluated" else '<p>无标注，未计算 AUROC。</p>')
    settings = read_json(output / "settings.json")
    baseline = list(protocol["methods"])[2]
    for index, response in enumerate(settings["responses"]):
        selected = [row for row in rows if row["response_id"] == response["id"]]
        observation = read_arrays(destination / "capture" / f"{index:04d}" / "observations.npz")
        peak = int(np.argmax([row["state_dynamics"] for row in selected]))
        content.extend((f'<h2>{escape(response["id"])}</h2>', curves(selected, ("state_dynamics", baseline)),
                        f'<p>自动展示本答最高分位置 t={peak}，仅用于定位；完整层头轨迹在 observations.npz。</p>'))
        channels = list(observation["profile_channels"])
        for name in ("source_read", "source_response_entropy", "source_direction_variance"):
            content.append(heatmap(observation["profile"][peak, ..., channels.index(name)], name))
    (destination / "report.html").write_text("\n".join(content), encoding="utf-8")
