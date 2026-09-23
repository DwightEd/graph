"""Compact offline comparison; all source and token text is escaped."""

from html import escape


def write_report(destination, rows, evaluation):
    columns = ("root_route", "direct_choice_route", "raw_route", "root_route_mean")
    metrics = []
    if evaluation["status"] == "evaluated":
        for method, phases in evaluation["methods"].items():
            result = phases["all_error"]
            metrics.append(f"<tr><td>{escape(method)}</td><td>{result['auroc']}</td><td>{result['ap']}</td></tr>")
    tokens = []
    for row in rows:
        values = "".join(f"<td>{row[name]:.5f}</td>" for name in columns)
        tokens.append(f"<tr><td>{escape(row['response_id'])}</td><td>{row['target']}</td>"
                      f"<td>{escape(row['token'])}</td>{values}</tr>")
    header = "".join(f"<th>{name}</th>" for name in columns)
    content = """<!doctype html><meta charset="utf-8"><title>Source transport</title>
<style>body{font:15px system-ui;margin:2em}td,th{padding:.3em .8em;text-align:left}
table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>
<h1>Signed source provenance through native value paths</h1>
<p>Automatic source blocks; no manual evidence types or label fitting.
Attention patterns and RMS scales are fixed for attribution; SwiGLU uses secant/product rules.
The forward is native. Scores are candidates, not probabilities of factual correctness.</p>
<p><a href="source_choices.csv">Source / candidate actions</a> ·
<a href="high_risk_normals.csv">High-ranked normal tokens</a> ·
<a href="onsets.csv">Onsets</a> · <a href="comparisons.csv">AUROC/AP changes</a></p>
<table><tr><th>Method</th><th>AUROC</th><th>AP</th></tr>"""
    content += "".join(metrics) + "</table><h2>Tokens</h2><table><tr>"
    content += "<th>Response</th><th>Target</th><th>Token</th>" + header + "</tr>"
    content += "".join(tokens) + "</table>"
    (destination / "report.html").write_text(content, encoding="utf-8")
