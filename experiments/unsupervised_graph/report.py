"""Reuse the existing official-span and positive/negative case renderer."""

from pathlib import Path
import numpy as np
from state_audit.storage import read_arrays, read_json, write_json

from experiments.native_support.ragtruth_benchmark.data import annotations
from experiments.probabilistic_detection.cases import KNOWN_CASES, build_case, render_html
from .data import load_inputs
from .model import fusion_scores


def cases(args):
    result = []
    for task in args.tasks:
        directory = args.output / task
        selection_path = directory / "selection.json"
        selection = read_json(selection_path if selection_path.exists() else directory / "pilot.json")
        for split in (("train", "test") if args.include_test else ("train",)):
            pack, metadata = load_inputs(args.packs, task, split)
            records = [row for row in metadata["records"] if row["id"] in KNOWN_CASES]
            if not records:
                continue
            root = Path(metadata["source_cache"])
            truth = annotations(root, read_json(root / "manifest.json"), records)
            for record in records:
                partition = record["partition"]
                filename = {"fit": "fit_scores", "dev": "development_scores", "test": "test_scores"}[partition]
                available = fusion_scores(read_arrays(directory / f"{filename}.npz"))
                chosen = selection["selected"]
                available["selected_dev"] = available[chosen]
                region = np.arange(record["packed_start"], record["packed_stop"])
                if split == "train":
                    mask = pack["development"] if partition == "dev" else ~pack["development"]
                    positions = (np.cumsum(mask) - 1)[region]
                else:
                    positions = region
                methods = ("fixed_unsupervised", "graph", "isolation", "selected_dev")
                scores = {name: available[name][positions] for name in methods}
                limits = dict(selection["mixed_thresholds"])
                limits["selected_dev"] = selection["thresholds"][chosen]
                response = read_json(root / record["directory"] / "response.json")
                result.append(build_case(record, response, truth[record["id"]], pack["target"][region],
                    pack["token_id"][region], scores, limits, radius=8))
    report = dict(cases=result, exposed_regression=True, test_included=args.include_test,
        threshold_rule="fixed/graph/isolation: unlabelled dev mixture95; selected_dev: labelled dev normal95")
    name = "cases_final" if args.include_test else ("cases_selection" if selection_path.exists() else "cases_pilot")
    output = args.output / name
    output.mkdir(exist_ok=True)
    write_json(output / "report.json", report)
    html = render_html(report).replace("阈值固定为各方法开发集正常token的95%分位数",
        "fixed/graph/isolation阈值为无标签开发混合分布95%分位；selected_dev阈值为开发正常token95%分位")
    html = html.replace("全部模型与预测冻结后读取官方标注", "模型不使用自然标签拟合；开发标注用于选型，测试标注仅在三任务预测冻结后读取")
    (output / "index.html").write_text(html)
    print(output / "index.html", flush=True)
