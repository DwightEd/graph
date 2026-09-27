"""Render exposed regression cases only after all requested task predictions freeze.

Run ``python -m experiments.probabilistic_detection.cases --run RUN``. Add
``--include-fit`` to score known training examples with frozen models; those
results are explicitly in-sample diagnostics and never held-out evidence.
"""

import argparse
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "teaching/state_audit/src"))

import numpy as np
from state_audit.storage import read_json, write_json

from experiments.native_support.ragtruth_benchmark.data import annotations


# Verified against binding_pilot_20260927/{RESULTS.md,CASE_BROWSER.html}.
# These descriptions select reporting cases only; they never enter a detector.
KNOWN_CASES = {
    "15604": "私人养老金误套5%税率",
    "9022": "营业时间、intimate及WiFi错误",
    "7305": "不同日期营业时间写成每天相同",
    "219": "more than four days写成for four days",
    "12045": "旧回归记录中的两个官方错误片段",
    "12219": "否认Passage3提供折被说明",
    "11907": "旧记录官方标签为空的正常对照",
    "12015": "旧记录官方标签为空的正常对照",
}
DEFAULT_METHODS = ("source_first", "source_refine", "full_s0.5_t0.25",
                   "selected_gaussian", "selected_readout", "selected_detector", "linear", "interactions",
                   "conditioned", "trees")
PARTITION_NOTES = {
    "fit": "训练内诊断（in-sample）；不得计入留出性能或泛化证据。",
    "dev": "开发来源：用于选模和定阈值，不是独立最终测试。",
    "test": "官方测试来源；历史研究已暴露，属于探索性复验。",
}


def require_frozen(run, tasks):
    """Check every requested task before any annotation loader is called."""
    for task in tasks:
        directory = run / task
        required = ["models.joblib", "selection.json", "development_scores.npz",
                    "test_scores.npz", "test_frozen.json"]
        if not all((directory / name).is_file() for name in required):
            raise ValueError(f"{task}: freeze models and all test predictions before case evaluation")
        if read_json(directory / "test_frozen.json").get("status") != "all_predictions_frozen":
            raise ValueError(f"{task}: test prediction freeze is incomplete")
        if not all((run / "packs" / f"{task}_{split}.{suffix}").is_file()
                   for split in ("train", "test") for suffix in ("json", "npz")):
            raise ValueError(f"{task}: original packed prediction inputs are missing")
        if (directory / "readout_selection.json").exists():
            for name in ("readouts.joblib", "readout_development_scores.npz"):
                if not (directory / name).is_file():
                    raise ValueError(f"{task}: selected readout artifacts are incomplete")


def read_selected_scores(path, methods):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in methods if name in saved.files}


def task_thresholds(directory):
    result = dict(read_json(directory / "selection.json")["thresholds"])
    path = directory / "readout_selection.json"
    if path.exists():
        result.update(read_json(path)["thresholds"])
    path = directory / "detector_selection.json"
    if path.exists():
        result["selected_detector"] = read_json(path)["threshold"]
    return result


def frozen_fit_scores(directory, pack, methods):
    """No fitting: optional predictions for known fit examples are in-sample."""
    import joblib
    from .model import ConditionalGaussian
    from .iteration import score_readouts
    from .run import append_gaussian_scores, existing_scores, feature_views

    models = joblib.load(directory / "models.joblib")
    selection = read_json(directory / "selection.json")
    scores = existing_scores(pack)
    for name, model in models.items():
        if isinstance(model, ConditionalGaussian):
            parts = model.predict_components(pack["context"], pack["observations"])
            append_gaussian_scores(scores, name, parts)
    raw, matched = feature_views(pack, models["full_s0.5"])
    scores.update(logistic=models["logistic"].decision_function(models["scaler"].transform(raw)),
        logistic_matched=models["logistic_matched"].decision_function(matched),
        trees=models["trees"].decision_function(raw))
    scores["selected_gaussian"] = scores[selection["selected"]].copy()
    if (directory / "readout_selection.json").exists():
        scores.update(score_readouts(directory, pack))
    path = directory / "detector_selection.json"
    if path.exists():
        scores["selected_detector"] = scores[read_json(path)["selected"]].copy()
    return {name: scores[name] for name in methods if name in scores}


def response_scores(pack, record, directory, methods, frozen, include_fit):
    region = np.arange(record["packed_start"], record["packed_stop"])
    partition = record["partition"]
    if partition == "fit":
        if not include_fit:
            return None
        selected = {name: value[region] for name, value in pack.items()}
        return frozen_fit_scores(directory, selected, methods)
    if partition == "dev":
        if not pack["development"][region].all():
            raise ValueError("Development record disagrees with packed row membership")
        positions = np.cumsum(pack["development"]) - 1
        return {name: values[positions[region]] for name, values in frozen.items()}
    return {name: values[region] for name, values in frozen.items()}


def span_summary(indices, neighbors, scores, thresholds):
    result = {}
    for method, values in scores.items():
        threshold = thresholds[method]
        if threshold is None:
            result[method] = dict(status="threshold_unavailable", threshold=None)
            continue
        alarm = values > threshold
        hits = int(alarm[indices].sum())
        status = "entire_span" if hits == len(indices) else "partial_span" if hits else "missed"
        result[method] = dict(status=status, threshold=threshold, detected_tokens=hits,
            gold_tokens=len(indices), token_recall=hits / len(indices),
            neighboring_normal_tokens=len(neighbors),
            neighboring_false_alarms=int(alarm[neighbors].sum()),
            neighboring_normal_fpr=float(alarm[neighbors].mean()) if len(neighbors) else None)
    return result


def token_rows(indices, targets, offsets, text, labels, scores, thresholds):
    result = []
    for index in indices:
        target = int(targets[index])
        start, stop = map(int, offsets[target])
        values = {name: dict(score=float(score[index]),
            alarm=bool(score[index] > thresholds[name]) if thresholds[name] is not None else None)
            for name, score in scores.items()}
        result.append(dict(token_index=target, char_start=start, char_stop=stop,
                           text=text[start:stop], gold=int(labels[index]), methods=values))
    return result


def build_case(record, response, annotation, targets, token_ids, scores, thresholds, radius):
    targets = np.asarray(targets)
    offsets = np.asarray(response["offsets"])
    if annotation["response_text"] != response["text"]:
        raise ValueError("Case annotation text differs from cached response")
    if not np.array_equal(np.asarray(response["answer_ids"])[targets], token_ids):
        raise ValueError("Case prediction token IDs differ from cached offsets")
    if not np.array_equal(np.asarray(annotation["token_ids"]), response["answer_ids"]):
        raise ValueError("Case annotations refer to different token IDs")
    if not np.array_equal(np.flatnonzero(annotation["valid_tokens"]), targets):
        raise ValueError("Case predictions do not cover every valid original token")
    labels = np.asarray(annotation["labels"])[targets]
    if any(values.shape != targets.shape or not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Case scores are incomplete or nonfinite")
    spans = []
    for span in annotation["character_spans"]:
        overlap = (offsets[targets, 0] < span["end"]) & (offsets[targets, 1] > span["start"])
        selected = np.flatnonzero(overlap)
        if not len(selected):
            raise ValueError("Official case span has no valid scored token")
        neighbors = np.flatnonzero((targets >= targets[selected[0]] - radius)
            & (targets <= targets[selected[-1]] + radius) & (labels == 0))
        shown = np.sort(np.r_[selected, neighbors])
        spans.append(dict(character_start=span["start"], character_stop=span["end"],
            text=response["text"][span["start"]:span["end"]],
            methods=span_summary(selected, neighbors, scores, thresholds),
            tokens=token_rows(shown, targets, offsets, response["text"], labels, scores, thresholds)))
    return dict(id=record["id"], source_id=record["source_id"], task=record["task"],
        generator=record["generator"], official_split=record["split"], partition=record["partition"],
        partition_note=PARTITION_NOTES[record["partition"]], status="scored",
        in_sample=record["partition"] == "fit", description=KNOWN_CASES.get(record["id"], "指定样本"),
        response_text=response["text"], spans=spans,
        normal_tokens=token_rows(np.flatnonzero(labels == 0), targets, offsets,
            response["text"], labels, scores, thresholds) if not spans else [],
        methods_available=list(scores), annotation_origin=annotation["annotation_origin"])


def partition_predictions(directory, split, methods):
    detector = directory / "detector_selection.json"
    selected = read_json(detector)["selected"] if detector.exists() else None
    requested = list(methods)
    if split == "train" and "selected_detector" in methods and selected is not None:
        requested.append(selected)
    name = "test_scores.npz" if split == "test" else "development_scores.npz"
    result = read_selected_scores(directory / name, requested)
    if split == "train" and (directory / "readout_development_scores.npz").exists():
        result.update(read_selected_scores(directory / "readout_development_scores.npz", requested))
    if split == "train" and "selected_detector" in methods and selected is not None:
        result["selected_detector"] = result[selected].copy()
    return {name: result[name] for name in methods if name in result}


def collect_partition(run, task, split, identities, methods, include_fit, radius):
    from .data import load_pack

    metadata = read_json(run / "packs" / f"{task}_{split}.json")
    chosen = [record for record in metadata["records"] if record["id"] in identities]
    if not chosen:
        return []
    pack, _ = load_pack(run / "packs", task, split)
    directory = run / task
    frozen = partition_predictions(directory, split, methods)
    expected = int(pack["development"].sum()) if split == "train" else len(pack["target"])
    if any(len(values) != expected for values in frozen.values()):
        raise ValueError("Frozen case scores do not match their packed partition")
    thresholds = task_thresholds(directory)
    root = Path(metadata["source_cache"])
    scored = [row for row in chosen if row["partition"] != "fit" or include_fit]
    truth = annotations(root, read_json(root / "manifest.json"), scored) if scored else {}
    result = []
    for record in chosen:
        scores = response_scores(pack, record, directory, methods, frozen, include_fit)
        if scores is None:
            result.append(dict(id=record["id"], task=task, source_id=record["source_id"],
                official_split=split, partition="fit", in_sample=True, status="fit_not_scored",
                partition_note=PARTITION_NOTES["fit"],
                reason="No held-out prediction exists; --include-fit enables an explicitly in-sample diagnostic."))
            continue
        response = read_json(root / record["directory"] / "response.json")
        region = slice(record["packed_start"], record["packed_stop"])
        case = build_case(record, response, truth[record["id"]], pack["target"][region],
            pack["token_id"][region], scores, thresholds, radius)
        case["methods_unavailable"] = [name for name in methods if name not in scores]
        result.append(case)
    return result


def number(value):
    return "—" if value is None else f"{value:.4f}"


def token_table(tokens, methods):
    headers = "".join(f"<th>{escape(name)}</th>" for name in methods)
    rows = []
    for token in tokens:
        cells = []
        for method in methods:
            value = token["methods"][method]
            alarm = "报警" if value["alarm"] else "正常" if value["alarm"] is not None else "阈值缺失"
            cells.append(f"<td>{number(value['score'])} / {alarm}</td>")
        role = "错误" if token["gold"] else "邻近正常"
        rows.append(f"<tr class='gold{token['gold']}'><td>{token['token_index']}</td>"
            f"<td>{escape(token['text'])}</td><td>{role}</td>{''.join(cells)}</tr>")
    return f"<table><tr><th>token</th><th>原文</th><th>标注</th>{headers}</tr>{''.join(rows)}</table>"


def span_html(span, methods):
    names = {"entire_span": "整个片段检出", "partial_span": "部分token检出", "missed": "未检出",
             "threshold_unavailable": "开发阈值不可用"}
    rows = []
    for method, result in span["methods"].items():
        hits = f"{result['detected_tokens']}/{result['gold_tokens']}" if "detected_tokens" in result else "—"
        rows.append(f"<tr><td>{escape(method)}</td><td>{names[result['status']]}</td>"
            f"<td>{hits}</td><td>{number(result.get('neighboring_normal_fpr'))}</td>"
            f"<td>{number(result['threshold'])}</td></tr>")
    return (f"<h3>官方错误片段：{escape(span['text'])}</h3><table><tr><th>方法</th><th>状态</th>"
        f"<th>错误token报警</th><th>邻近正常FPR</th><th>开发阈值</th></tr>{''.join(rows)}</table>"
        + token_table(span["tokens"], methods))


def render_html(report):
    cards = []
    for case in report["cases"]:
        title = f"{case['id']} · {case.get('task', '未找到')} · {case.get('partition', 'unknown')}"
        note = case.get("partition_note", case.get("reason", ""))
        body = f"<h2>{escape(title)}</h2><p class='warning'>{escape(note)}</p>"
        if case["status"] == "scored":
            body += f"<p>{escape(case['description'])}</p><details><summary>完整原回答</summary><pre>{escape(case['response_text'])}</pre></details>"
            body += "".join(span_html(span, case["methods_available"]) for span in case["spans"])
            if not case["spans"]:
                body += "<p>官方标签为空；保留全部正常token检查误报。</p>"
                body += token_table(case["normal_tokens"], case["methods_available"])
        else:
            body += f"<p>{escape(case.get('reason', case['status']))}</p>"
        cards.append(f"<section>{body}</section>")
    return ("<!doctype html><html lang='zh'><meta charset='utf-8'><title>概率检测逐例回归</title>"
        "<style>body{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#20242a}"
        "section{margin:28px 0;border-top:2px solid #ccd5df}table{border-collapse:collapse;display:block;overflow:auto;margin:14px 0}"
        "td,th{padding:7px;border:1px solid #d8dee5;white-space:pre-wrap}th{background:#e9eff6}"
        ".gold1{background:#fff0ee}.warning{font-weight:bold;color:#8b3424}pre{white-space:pre-wrap}</style>"
        "<h1>概率检测逐例回归</h1><p>全部模型与预测冻结后读取官方标注；阈值固定为各方法开发集正常token的95%分位数，严格 score &gt; threshold。"
        "已知错例仅作暴露后的回归检查；训练内、开发、测试结果分别标识，不能替代独立泛化证据。</p>"
        + "".join(cards) + "</html>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--ids", nargs="+", default=list(KNOWN_CASES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--neighbor-tokens", type=int, default=8)
    parser.add_argument("--include-fit", action="store_true")
    args = parser.parse_args(argv)
    prepared = sorted(path.name.removesuffix("_train.json") for path in (args.run / "packs").glob("*_train.json"))
    tasks = args.tasks or prepared
    if not tasks or args.neighbor_tokens < 0:
        raise ValueError("Require prepared tasks and a nonnegative neighbor-token radius")
    require_frozen(args.run, sorted(set(prepared) | set(tasks)))
    output = (args.output or args.run) / "cases"
    if (output / "report.json").exists() or (output / "index.html").exists():
        raise FileExistsError("Case report already exists; choose a new --output")
    cases = [case for task in tasks for split in ("train", "test")
             for case in collect_partition(args.run, task, split, set(args.ids), args.methods,
                                           args.include_fit, args.neighbor_tokens)]
    lookup = {case["id"]: case for case in cases}
    if len(lookup) != len(cases):
        raise ValueError("Known response appears in multiple packed partitions")
    ordered = [lookup.get(identity, dict(id=identity, status="not_in_packs",
        reason="This response is absent from the requested frozen run; no score was fabricated.")) for identity in args.ids]
    report = dict(run=str(args.run.resolve()), tasks=tasks, requested_methods=args.methods,
        neighbor_tokens=args.neighbor_tokens, include_fit=args.include_fit,
        threshold_rule="frozen_development_95_percent_normal_quantile_strict_gt",
        known_case_origin="binding_pilot_20260927/RESULTS.md and CASE_BROWSER.html",
        exposed_regression=True, cases=ordered)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    (output / "index.html").write_text(render_html(report), encoding="utf-8")
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
