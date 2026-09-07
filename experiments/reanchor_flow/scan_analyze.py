"""Fit, score and audit existing full scans, without loading a language model."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import save_json as _write_json
from .detection_metrics import detection_report, render_detection_report
from .routing_transition import RoutingSequence, RoutingTransitionModel
from .scan_dataset import ScanDataset, ScanLabelStore

SCORES = ("routing_joint", "routing_static", "routing_transition")


def partition_sources(train, test, seed=2026):
    """Hold out calibration sources; remove test-source overlap using IDs only."""
    test_sources = {r.source_id for r in test.records}
    eligible = train.select(
        r.sample_id for r in train.records if r.source_id not in test_sources
    )
    source_tasks = defaultdict(set)
    for record in eligible.records:
        source_tasks[record.source_id].add(record.task_type)
    strata = defaultdict(list)
    for source, tasks in source_tasks.items():
        strata[tuple(sorted(tasks))].append(source)
    calibration_sources = set()
    for tasks, sources in strata.items():
        if len(sources) < 2:
            raise ValueError(
                f"need at least two independent training sources for {tasks}"
            )
        sources.sort(
            key=lambda source: hashlib.sha256(f"{seed}:{source}".encode()).digest()
        )
        calibration_sources.update(sources[: max(1, round(0.2 * len(sources)))])
    fit = eligible.select(
        r.sample_id for r in eligible.records if r.source_id not in calibration_sources
    )
    calibration = eligible.select(
        r.sample_id for r in eligible.records if r.source_id in calibration_sources
    )
    return (
        fit,
        calibration,
        {
            "seed": seed,
            "fit_source_ids": sorted({r.source_id for r in fit.records}),
            "calibration_source_ids": sorted(calibration_sources),
            "excluded_train_sample_ids": sorted(
                set(train.sample_ids) - set(eligible.sample_ids)
            ),
            "test_source_ids": sorted(test_sources),
            "labels_used": False,
        },
    )


def _sequences(scans, description):
    weights = scans.sample_weights()
    for record in tqdm(scans.records, desc=description, leave=False):
        scan = scans.load(record.sample_id, fields=("reanchor_bucket_transport",))
        yield RoutingSequence.from_scan(scan, sample_weight=weights[record.sample_id])


def _raw_scores(scores):
    return np.column_stack(
        (
            scores.joint,
            scores.current,
            np.where(scores.has_previous, scores.innovation, scores.current),
        )
    )


class ScoreCalibration:
    """Source-balanced location/scale on held-out unlabeled training sources.

    First-token marginal and later joint energies have different dimensions.
    Calibrate them separately without clipping the tail or using test labels.
    Scores are standardized anomaly energies, never correctness probabilities.
    """

    def fit(self, model, scans):
        weights = np.zeros(2)
        sums = np.zeros((2, 3))
        squares = np.zeros((2, 3))
        for sequence in _sequences(scans, "calibrate sources"):
            result = model.score(sequence)
            values = _raw_scores(result)
            for group in (0, 1):
                rows = np.flatnonzero(result.has_previous == group)
                if not len(rows):
                    continue
                rows = rows[
                    np.linspace(0, len(rows) - 1, min(128, len(rows)), dtype=int)
                ]
                weight = sequence.sample_weight / len(rows)
                weights[group] += sequence.sample_weight
                sums[group] += weight * values[rows].sum(axis=0)
                squares[group] += weight * (values[rows] ** 2).sum(axis=0)
        if np.any(weights == 0):
            raise ValueError(
                "calibration requires first tokens and adjacent continuation tokens"
            )
        self.mean = sums / weights[:, None]
        self.scale = np.sqrt(
            np.maximum(squares / weights[:, None] - self.mean**2, 1e-6)
        )
        return self

    def transform(self, result):
        group = result.has_previous.astype(int)
        values = (_raw_scores(result) - self.mean[group]) / self.scale[group]
        return dict(zip(SCORES, values.T, strict=True))

    def save(self, path):
        np.savez_compressed(path, schema=1, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            instance = cls()
            instance.mean, instance.scale = stored["mean"], stored["scale"]
        return instance


def _compatible(train, test):
    if train.split != "train" or test.split != "test":
        raise ValueError("--scans must contain train/ and test/ capture manifests")
    for key in ("model", "model_dtype", "tokenizer", "flow_signal", "local_window"):
        if train.config.get(key) != test.config.get(key):
            raise ValueError(f"train/test capture configuration differs: {key}")
    if {r.task_type for r in test.records} - {r.task_type for r in train.records}:
        raise ValueError("test includes a task without training scans")


def _task_view(scans, task):
    return scans.select(r.sample_id for r in scans.records if r.task_type == task)


def _score_test(test, models, calibrations, output):
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    index = []
    for number, record in enumerate(tqdm(test.records, desc="score all test tokens")):
        scan = test.load(record.sample_id, fields=("reanchor_bucket_transport",))
        sequence = RoutingSequence.from_scan(scan)
        model = models[record.task_type]
        result = model.score(sequence)
        calibrated = calibrations[record.task_type].transform(result)
        excess = model.head_excess(result).reshape(scan.rows, -1)
        top = np.argsort(-excess, axis=1)[:, : min(3, excess.shape[1])]
        heads = result.per_head_joint.shape[-1]
        path = predictions / f"{number:06d}.npz"
        np.savez_compressed(
            path,
            schema=1,
            sample_id=record.sample_id,
            source_id=record.source_id,
            task_type=record.task_type,
            query_position=scan.row_position,
            prediction_position=scan.row_position + 1,
            response_index=scan.response_index,
            has_previous=result.has_previous,
            **calibrated,
            position=scan.response_index.astype(float),
            raw_joint=result.joint,
            raw_static=result.current,
            raw_previous=result.previous,
            raw_innovation=result.innovation,
            top_layer=top // heads,
            top_head=top % heads,
            top_head_excess=np.take_along_axis(excess, top, axis=1),
        )
        index.append(
            {
                "sample_id": record.sample_id,
                "source_id": record.source_id,
                "task_type": record.task_type,
                "path": str(path.relative_to(output)),
                "scored_tokens": scan.rows,
                "full_response_tokens": scan.metadata["full_response_tokens"],
            }
        )
    return index


def _probe(fit, test, output, index, cache):
    from .routing_probe import SupervisedRoutingProbe

    labels = ScanLabelStore(fit, dataset_root=cache / "train" if cache else None)
    for task in sorted({r.task_type for r in test.records}):
        training = _task_view(fit, task)
        weights = training.sample_weights()

        def factory(training=training, task=task, weights=weights):
            for record in tqdm(
                training.records, desc=f"{task} supervised diagnostic", leave=False
            ):
                scan = training.load(
                    record.sample_id, fields=("reanchor_bucket_transport",)
                )
                yield (
                    RoutingSequence.from_scan(
                        scan, sample_weight=weights[record.sample_id]
                    ),
                    labels.load(scan),
                )

        probe = SupervisedRoutingProbe().fit(factory)
        probe.save(output / "models" / f"{task}_supervised_probe.npz")
        for item in tqdm(
            [x for x in index if x["task_type"] == task],
            desc=f"{task} probe test",
            leave=False,
        ):
            scan = test.load(item["sample_id"], fields=("reanchor_bucket_transport",))
            path = output / item["path"]
            with np.load(path, allow_pickle=False) as stored:
                payload = dict(stored)
            payload.update(probe.score(RoutingSequence.from_scan(scan)))
            np.savez_compressed(path, **payload)


def _evaluate(test, output, index, cache, bootstrap, seed, supervised):
    store = ScanLabelStore(test, dataset_root=cache / "test" if cache else None)
    names = (*SCORES, "position")
    if supervised:
        names += ("supervised_routing", "supervised_position")
    columns = {name: [] for name in (*names, "labels", "source_ids", "task_types")}
    for item in tqdm(index, desc="join test labels after freeze"):
        scan = test.load(item["sample_id"], fields=())
        labels = store.load(scan)
        path = output / item["path"]
        with np.load(path, allow_pickle=False) as saved:
            payload = dict(saved)
        payload["labels"] = labels
        np.savez_compressed(path, **payload)
        for name in names:
            columns[name].append(payload[name])
        columns["labels"].append(labels)
        columns["source_ids"].append(np.repeat(item["source_id"], len(labels)))
        columns["task_types"].append(np.repeat(item["task_type"], len(labels)))
    values = {key: np.concatenate(parts) for key, parts in columns.items()}
    scores = {key: values[key] for key in (*SCORES, "position")}
    report = detection_report(
        values["labels"],
        scores,
        values["source_ids"],
        values["task_types"],
        bootstrap=bootstrap,
        seed=seed,
    )
    if supervised:
        report["supervised_diagnostic"] = detection_report(
            values["labels"],
            {key: values[key] for key in ("supervised_routing", "supervised_position")},
            values["source_ids"],
            values["task_types"],
            primary="supervised_routing",
            bootstrap=bootstrap,
            seed=seed,
        )
    return report, values, scores


def run_analysis(
    scans,
    output,
    *,
    cache=None,
    bootstrap=200,
    seed=2026,
    supervised_probe=False,
    plot=True,
    events=True,
):
    scans, output = Path(scans), Path(output)
    cache = Path(cache) if cache else None
    train, test = ScanDataset(scans / "train"), ScanDataset(scans / "test")
    _compatible(train, test)
    fit, calibration, partition = partition_sources(train, test, seed)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    _write_json(output / "source_partition.json", partition)
    models, calibrations = {}, {}
    for task in sorted({r.task_type for r in test.records}):
        training = _task_view(fit, task)
        model = RoutingTransitionModel().fit(
            lambda training=training, task=task: _sequences(
                training, f"{task} fit scans"
            )
        )
        model.save(output / "models" / f"{task}_routing.npz")
        normalizer = ScoreCalibration().fit(model, _task_view(calibration, task))
        normalizer.save(output / "models" / f"{task}_calibration.npz")
        models[task], calibrations[task] = model, normalizer
    index = _score_test(test, models, calibrations, output)
    frozen = {
        "schema": 1,
        "analysis": "head_resolved_conditional_routing_density",
        "capture_root": str(scans.resolve()),
        "density_uses_labels": False,
        "score_direction": "higher_is_more_hallucinated",
        "seed": seed,
        "calibration": "source-balanced unlabeled mean/std; separate first/continuation",
        "first_token_rule": "current-state fallback, separately calibrated",
        "cross_head_model": "product of per-head densities, no averaged head state",
        "test_samples": len(index),
        "scored_tokens": sum(x["scored_tokens"] for x in index),
        "full_response_tokens": sum(x["full_response_tokens"] for x in index),
        "first_token_fallbacks": len(index),
        "predictions": index,
        "limitations": [
            "observational routing anomaly is not causal factual evidence",
            "test cohort aggregates informed this exploratory design",
            "sum of edge-message norms is not net residual-write magnitude",
        ],
    }
    _write_json(output / "frozen_detector.json", frozen)
    # This is the first point at which any label reader can be constructed.
    if supervised_probe:
        _probe(fit, test, output, index, cache)
    report, values, scores = _evaluate(
        test, output, index, cache, bootstrap, seed, supervised_probe
    )
    report["coverage"] = {
        key: frozen[key]
        for key in (
            "test_samples",
            "scored_tokens",
            "full_response_tokens",
            "first_token_fallbacks",
        )
    }
    _write_json(output / "detection_report.json", report)
    _save_summary(report, output)
    if plot:
        render_detection_report(
            output / "detection_curves.png",
            values["labels"],
            scores,
            values["task_types"],
        )
        if supervised_probe:
            render_detection_report(
                output / "supervised_diagnostic_curves.png",
                values["labels"],
                {
                    key: values[key]
                    for key in ("supervised_routing", "supervised_position")
                },
                values["task_types"],
                primary="supervised_routing",
            )
    if events:
        from .routing_events import RoutingEventAudit

        eligible = train.select((*fit.sample_ids, *calibration.sample_ids))
        train_labels = ScanLabelStore(
            eligible, dataset_root=cache / "train" if cache else None
        )
        test_labels = ScanLabelStore(
            test, dataset_root=cache / "test" if cache else None
        )
        audit = RoutingEventAudit().discover(eligible, train_labels)
        audit.run(test, test_labels, output / "events", plot=plot)
    _print_report(report, output)
    return report


def _save_summary(report, output):
    lines = [
        "# 路由检测结果（探索性评估）",
        "",
        "主方法是无监督逐 head 时序密度；分数越高越异常，不是幻觉概率。",
        "AUPRC 使用 average precision。95% 区间按 source 成簇重采样，未知标签不计入指标。",
        "",
        "| 任务 | 方法 | AUROC [95% CI] | AUPRC [95% CI] | 幻觉比例 |",
        "|---|---|---|---|---|",
    ]

    def value(number, interval):
        if number is None:
            return "NA"
        suffix = f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else ""
        return f"{number:.4f}{suffix}"

    sections = [("无监督主结果", report)]
    if "supervised_diagnostic" in report:
        sections.append(
            (
                "独立监督诊断：使用 train 标签，不属于无监督结果",
                report["supervised_diagnostic"],
            )
        )
    for title, section in sections:
        if section is not report:
            lines.extend(["", f"## {title}", "", lines[5], lines[6]])
        for task, group in section["groups"].items():
            for name, score in group["scores"].items():
                prevalence = group["prevalence"]
                baseline = f"{prevalence:.4f}" if prevalence is not None else "NA"
                lines.append(
                    f"| {task} | {name} | {value(score['auroc'], score['auroc_ci95'])} | {value(score['auprc'], score['auprc_ci95'])} | {baseline} |"
                )
        lines.extend(["", "与对照的配对差值（主方法 − 对照）：", ""])
        for task, group in section["groups"].items():
            for name, difference in group["paired_differences"].items():
                lines.append(
                    f"- {task} / {name}: ΔAUROC {value(difference['auroc'], difference['auroc_ci95'])}; ΔAUPRC {value(difference['auprc'], difference['auprc_ci95'])}。"
                )
    lines.extend(
        [
            "",
            "## 如何判读",
            "",
            "先检查 routing_joint 是否同时超过 position 和 routing_static；差值区间跨 0 时，新增时序建模的优势尚不确定。",
            "routing_transition 是条件转换异常对照；稳定但错误的状态未必有大的转换异常。",
            "如果监督路由诊断超过监督位置基线、无监督结果却不理想，说明扫描含可学习的判别信号，但当前无监督密度假设不够有效。",
            "事件图用于定位训练集选出的 head 及其在 test 的轨迹差异，不能单凭图证明证据被 MLP 接纳、置信度编码或因果贡献。",
            "此设计已参考 test 的 cohort 汇总，当前 test 指标属于探索性结果；正式确认需新的未参与设计的数据。",
            "",
        ]
    )
    (output / "detection_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _print_report(report, output):
    for task, group in report["groups"].items():
        print(
            f"\n{task}: tokens={group['known_tokens']} prevalence={group['prevalence']}"
        )
        for name, score in group["scores"].items():
            print(f"  {name:22s} AUROC={score['auroc']} AUPRC={score['auprc']}")
    if "supervised_diagnostic" in report:
        print(
            "\nSupervised readout is a separate diagnostic (see report), not the unsupervised detector."
        )
    print(f"\nResults: {output / 'detection_report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scans",
        type=Path,
        required=True,
        help="completed capture root containing train/ and test/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        help="optional moved label-cache root containing train/ and test/",
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--supervised-probe",
        action="store_true",
        help="also fit a separately reported label-trained linear diagnostic",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--no-events", action="store_true", help="only evaluate detection"
    )
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    run_analysis(
        args.scans,
        args.output,
        cache=args.cache,
        bootstrap=args.bootstrap,
        seed=args.seed,
        supervised_probe=args.supervised_probe,
        plot=not args.no_plot,
        events=not args.no_events,
    )


if __name__ == "__main__":
    main()
