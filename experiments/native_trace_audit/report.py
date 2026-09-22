"""Rebuild an inspectable report from cached NPZs without loading a language model."""

import html
import tarfile
from io import BytesIO

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from .tables import (
    dependency_rows,
    head_rows,
    identity,
    ledger_rows,
    query_row,
    source_edge_rows,
)
from .timeline import reanchor_rows

TABLE_NAMES = (
    "queries",
    "ledger",
    "head_sources",
    "dependencies",
    "source_edges",
    "reanchor",
)


def table_html(rows, fields):
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            display = f"{value:.5g}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(display)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='scroll'><table><tr>"
        + header
        + "</tr>"
        + "".join(body)
        + "</table></div>"
    )


def panel_html(manifest, summary, relative, edges, dependencies):
    context, panel = manifest["context"], manifest["panel"]
    title = f"{context['case_id']} / {context['side']} / {panel['name']}"
    text = html.escape("".join(context["token_text"][context["prompt_length"] :]))
    suffix = html.escape(panel["suffix"])
    candidates = " / ".join(html.escape(value) for value in panel["candidate_texts"])
    details = table_html(
        [summary],
        ("logit_gap", "entropy", "surprisal", "ledger_error", "numeric_absolute_sum"),
    )
    source_table = table_html(
        edges, ("layer", "head", "group", "source", "token", "attention", "logit_write")
    )
    dependence = dependency_html(dependencies)
    timeline = (
        f'<img src="{relative}/timeline.png" alt="native token timeline">'
        if panel["natural"]
        else ""
    )
    return (
        f"<section><h2>{html.escape(title)}</h2><p>{html.escape(panel['meaning'])}</p>"
        f"<p>历史状态：{html.escape(context['history_status'])}；只核验指定局部 claim。</p>"
        f"<details><summary>原生前缀与候选</summary><pre>{text}</pre>"
        f"<p>额外强制前缀：<code>{suffix}</code></p><p>候选：{candidates}</p></details>"
        f"{details}<img src='{relative}/ledger.png' alt='residual ledger'>"
        f"<img src='{relative}/heads.png' alt='all physical layer head measurements'>{timeline}"
        f"<details><summary>来源词与实际写入：按绝对值展示，保留正负号</summary>{source_table}</details>"
        f"{dependence}</section>"
    )


def dependency_html(dependencies):
    dependency_tables = []
    for kind in ("route_mass_difference", "mlp_logit_write"):
        selected = [row for row in dependencies if row["receiver_kind"] == kind]
        selected.sort(key=lambda row: -abs(row["local_sensitivity"]))
        dependency_tables.append(
            f"<h3>{kind}</h3>"
            + table_html(
                selected[:20],
                (
                    "receiver_layer",
                    "receiver_head",
                    "sender_kind",
                    "sender_layer",
                    "sender_head",
                    "source_group",
                    "local_sensitivity",
                ),
            )
        )
    return (
        "<details><summary>后续读取与 FFN 更新的局部依赖</summary>"
        "<p>按同类接收量分别展示；梯度不等于消息流量，不能相乘或跨层相加。</p>"
        + "".join(dependency_tables)
        + "</details>"
    )


def report_panel(directory, output, settings):
    from .plots import plot_heads, plot_ledger, plot_timeline

    manifest = read_json(directory / "plan.json")
    traces = [
        read_arrays(directory / f"query_{plan['query']:06d}.npz")
        for plan in manifest["plans"]
        if (directory / f"query_{plan['query']:06d}.npz").exists()
    ]
    queries = [query_row(manifest, trace) for trace in traces]
    tables = {name: [] for name in TABLE_NAMES}
    tables.update(queries=queries, html="")
    if not traces or int(traces[-1]["decision_offset"]) != 0:
        return tables
    decision = traces[-1]
    edges = source_edge_rows(manifest, decision)
    plot_ledger(decision, directory / "ledger.png")
    plot_heads(decision, directory / "heads.png")
    events = []
    if manifest["panel"]["natural"]:
        plot_timeline(traces, directory / "timeline.png")
        events = reanchor_rows(
            traces,
            manifest["context"]["prompt_length"],
            manifest["special_ids"],
            manifest["panel"]["token_ids"],
            recent=settings["recent"],
        )
        events = [dict(**identity(manifest), **event) for event in events]
    dependencies = dependency_rows(manifest, decision)
    markup = panel_html(
        manifest,
        queries[-1],
        directory.relative_to(output).as_posix(),
        edges,
        dependencies,
    )
    ledgers = [row for trace in traces for row in ledger_rows(manifest, trace)]
    tables.update(
        ledger=ledgers,
        head_sources=head_rows(manifest, decision),
        dependencies=dependencies,
        source_edges=edges,
        reanchor=events,
        html=markup,
    )
    return tables


def write_html(output, summary, sections):
    introduction = """<h1>原始轨迹审计：读取、写入与后续使用</h1>
<p>没有删除信息、替换消息或训练检测器。正值表示支持候选 1 相对候选 2。
自然措辞对比与新增的 teacher-forced 约束问题分别报告；首个分歧 token 的分数不是完整命题的支持概率。</p>
<p>残差曲线是用本次最终 RMSNorm 尺度得到的同一输出方向投影，不是各中间层的预测。
局部梯度反映当前查询内依赖，不等于实际写入量，不跨层相加，也不证明跨 token 的因果中继。</p>
<p>Attention 大不等于理解约束，FFN 负写入不等于信息被删除；此报告不自动给样本命名幻觉机制。</p>"""
    css = "body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#17243b}"
    css += "img{width:100%}section{border-top:2px solid #ddd;margin-top:40px}td,th{padding:8px;border-bottom:1px solid #ddd}"
    css += ".scroll{overflow:auto}pre{white-space:pre-wrap}details{margin:18px 0}"
    counts = f"<p>已保存 {summary['completed_queries']} / {summary['expected_queries']} 个查询；仅记录已采集数据。</p>"
    (output / "report.html").write_text(
        "<!doctype html><html lang='zh'><meta charset='utf-8'>"
        f"<title>原始轨迹审计</title><style>{css}</style><body>{introduction}{counts}"
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )


def write_report(output):
    settings = read_json(output / "settings.json")
    tables = {name: [] for name in TABLE_NAMES}
    sections = []
    expected = sum(
        min(settings["window"], len(c["prefix_ids"]) - c["prompt_length"]) + 2
        for c in settings["inputs"]["contexts"]
    )
    for plan_path in sorted((output / "cases").glob("*/*/*/plan.json")):
        result = report_panel(plan_path.parent, output, settings)
        for name in TABLE_NAMES:
            tables[name].extend(result[name])
        sections.append(result["html"])
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows, list(rows[0]) if rows else ["case_id"])
    comparisons = paired_decisions(tables["queries"])
    write_csv(
        output / "paired_decisions.csv",
        comparisons,
        list(comparisons[0]) if comparisons else ["case_id"],
    )
    summary = {
        "purpose": settings["purpose"],
        "model": settings["model"],
        "model_run": bool(tables["queries"]),
        "expected_queries": expected,
        "completed_queries": len(tables["queries"]),
        "decision_panels": sum(bool(s) for s in sections),
        "max_abs_ledger_error": max(
            (abs(row["ledger_error"]) for row in tables["queries"]), default=None
        ),
        "capture_seconds": sum(row["capture_seconds"] for row in tables["queries"]),
        "dependency_csv": "Top 32 absolute head/source derivatives per receiver; all FFN senders; full arrays in NPZ",
        "reanchor_scope": "Full native attention; first 3 captured rows have no assessed prior window",
        "detector_evaluation": False,
        "interventions": False,
    }
    write_json(output / "report.json", summary)
    write_html(output, summary, sections)
    package_report(output)
    return summary


def paired_decisions(queries):
    grouped = {}
    for row in queries:
        if row["decision_offset"] == 0:
            grouped.setdefault((row["case_id"], row["panel"]), {})[row["side"]] = row
    rows = []
    for (case_id, panel), sides in grouped.items():
        if set(sides) != {"supported", "unsupported"}:
            continue
        rows.append(
            {
                "case_id": case_id,
                "panel": panel,
                "supported_prefix_gap": sides["supported"]["logit_gap"],
                "unsupported_prefix_gap": sides["unsupported"]["logit_gap"],
                "difference": sides["unsupported"]["logit_gap"]
                - sides["supported"]["logit_gap"],
                "comparison": "descriptive; natural prefixes and previous errors are not matched",
            }
        )
    return rows


def package_report(output):
    """Review archive includes compact dependencies/ledger as well as figures and tables."""
    with tarfile.open(output / "review.tar.gz", "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".png", ".html"}:
                archive.add(path, arcname=path.relative_to(output))
        for path in sorted(output.glob("cases/*/*/*/query_*.npz")):
            # Full native vectors stay on the server; retain exact arrays used by the review.
            arrays = read_arrays(path)
            omitted = {
                "head_writes",
                "group_readouts",
                "residual_before",
                "residual_mid",
                "residual_after",
                "attention_write",
                "mlp_write",
                "mlp_activation",
                "head_readout",
            }
            selected = {
                name: value for name, value in arrays.items() if name not in omitted
            }
            buffer = BytesIO()
            np.savez_compressed(buffer, **selected)
            info = tarfile.TarInfo(str(path.relative_to(output)))
            info.size = buffer.tell()
            buffer.seek(0)
            archive.addfile(info, buffer)
