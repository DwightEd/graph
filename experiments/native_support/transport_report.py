"""Compact offline comparison; all source and token text is escaped."""

from html import escape


def write_report(destination, rows, evaluation):
    columns = ("transport_route", "raw_route", "route_offline_mean")
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
<h1>Source response budget states</h1>
<p>Automatic source blocks; no manual evidence types or label fitting.
Routing scores are empirical candidates, not probabilities of factual correctness.</p>
<table><tr><th>Method</th><th>AUROC</th><th>AP</th></tr>"""
    content += "".join(metrics) + "</table><h2>Tokens</h2><table><tr>"
    content += "<th>Response</th><th>Target</th><th>Token</th>" + header + "</tr>"
    content += "".join(tokens) + "</table>"
    (destination / "report.html").write_text(content, encoding="utf-8")
