"""Every token, three fixed observations, one evaluation. No state fitting or ablation."""

from html import escape

import numpy as np
from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)
from tqdm import tqdm

from .comparison import evaluation_headlines
from .comparison_evaluation import evaluate_comparison
from .report import curves, metric_table
from .source_regions import compile_regions, region_mask
from .token_features import cached_features

DIRECTORY = "token_detection"
POSITION_FIELDS = ("target", "query", "token_id", "ledger_error")


def source_regions(output, settings):
    directories = (output / "route_comparison_v2", output / "route_filter_v3", output / DIRECTORY)
    directory = next((path for path in directories if (path / "source_regions.json").is_file()), directories[-1])
    return compile_regions(settings, directory)


def token_rows(response, scores, methods):
    rows = []
    for target in range(len(scores["token_id"])):
        values = {name: float(scores[name][target]) for name in (*methods, "risk", "ledger_error")}
        rows.append({"response_id": response["id"], "source_id": response["source_id"],
                     "target": target, "query": int(scores["query"][target]),
                     "token_id": int(scores["token_id"][target]),
                     "token": response["token_text"][response["prompt_length"] + target], **values})
    return rows


def score_tokens(output, settings, regions, protocol):
    destination = output / DIRECTORY
    methods = protocol["methods"]
    fields = (*methods, *POSITION_FIELDS)
    rows = []
    for index, response in enumerate(tqdm(settings["responses"], desc="token detection")):
        old_cache = output / "route_filter_v3" / "features" / f"{index:04d}.npz"
        cache = old_cache if old_cache.is_file() else destination / "features" / f"{index:04d}.npz"
        features = cached_features(response, output / "responses" / f"{index:04d}", cache,
                                   region_mask(regions, response), fields=fields)
        count = len(response["token_ids"]) - response["prompt_length"]
        expected_queries = np.arange(response["prompt_length"] - 1, len(response["token_ids"]) - 1)
        if not np.array_equal(features["target"], np.arange(count)) or not np.array_equal(features["query"], expected_queries):
            raise ValueError(f"{response['id']}: cached prediction positions differ from input")
        scores = {name: features[name] for name in fields}
        scores["risk"] = scores[protocol["primary_method"]]
        current = token_rows(response, scores, methods)
        directory = destination / "responses" / f"{index:04d}"
        write_arrays(directory / "scores.npz", **scores)
        rows.extend(current)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    return rows


def write_report(destination, rows, protocol, evaluation):
    content = ['<!doctype html><meta charset="utf-8"><title>Token detection</title>',
               '<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 15px}',
               'table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}',
               'svg{width:100%;height:180px}</style><h1>逐 token 原生检测</h1>',
               f'<p>主分数：{escape(protocol["primary_method"])}。读取路由、消息范数路由、输出熵独立报告。',
               '不做消融、分段、参考拟合或融合；高分不是已校准的幻觉概率。</p>',
               '<p><a href="tokens.csv">全部token分数</a> · <a href="evaluation.json">完整AUROC/AP</a> · ',
               '<a href="onsets.csv">首错与片段起点</a> · <a href="high_risk_normals.csv">高分正常词</a></p>']
    if evaluation["status"] == "evaluated":
        content.append(metric_table(evaluation["methods"]))
    else:
        content.append('<p>缺少自然标签，分数已保存，没有AUROC/AP。</p>')
    methods = tuple(protocol["methods"])
    for identity in dict.fromkeys(row["response_id"] for row in rows):
        selected = [row for row in rows if row["response_id"] == identity]
        content.extend((f'<h2>{escape(identity)}</h2><p>蓝：消息范数路由；橙：注意力路由。</p>', curves(selected, methods[:2])))
    (destination / "report.html").write_text('\n'.join(content), encoding="utf-8")


def finish(output, protocol, rows, annotations):
    destination = output / DIRECTORY
    evaluation = evaluate_comparison(output, destination, annotations or output / "annotations.json", protocol["methods"], rows)
    summary = {**protocol, "responses": len({r["response_id"] for r in rows}), "scored_tokens": len(rows),
               "evaluation": evaluation_headlines(evaluation),
               "evaluation_performed_by_this_stage": evaluation["status"] == "evaluated"}
    write_json(destination / "summary.json", summary)
    write_report(destination, rows, protocol, evaluation)
    return summary


def run_detection(output, settings, annotations=None):
    regions = source_regions(output, settings)
    prefix = "" if regions["status"] == "available" else "prompt_"
    methods = (prefix + "routing_imbalance", prefix + "attention_displacement", "entropy")
    protocol = {"purpose": "native_token_detection", "primary_method": methods[0],
                "methods": {name: name for name in methods}, "source_regions": regions["status"],
                "cohort": settings.get("cohort", {"selection": "input_manifest"}),
                "prediction_alignment": "query=P+t-1; current target is not a routing input",
                "labels_used_for_scoring": False, "gradients": False, "interventions": False,
                "model_forward_during_scoring": False, "reference_required": False,
                "state_fitting": False, "score_fusion": False, "automatic_method_selection": False,
                "risk_definition": "unchanged_historical_message_norm_routing_imbalance",
                "mechanism_claim": "none; source norm allocation is not semantic support"}
    write_json(output / DIRECTORY / "scoring_protocol.json", protocol)
    rows = score_tokens(output, settings, regions, protocol)
    return finish(output, protocol, rows, annotations)


def evaluate_detection(output, annotations=None):
    from .evaluate import unavailable_evaluation

    destination = output / DIRECTORY
    annotations = annotations or output / "annotations.json"
    if not annotations.is_file():
        return unavailable_evaluation(destination, annotations)
    protocol = read_json(destination / "scoring_protocol.json")
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(token_rows(response, saved, protocol["methods"]))
    return finish(output, protocol, rows, annotations)["evaluation"]
