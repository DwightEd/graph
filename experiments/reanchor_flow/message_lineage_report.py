"""Post-hoc N/H comparisons and fixed-score evaluation of message lineage."""
from collections import defaultdict
from html import escape
from pathlib import Path
import json

import numpy as np
from tqdm.auto import tqdm

from .attention_audit_stats import Moments, bracket_positions, matched_difference, mean, token_classes
from .attention_rhythm_report import save_json
from .detection_metrics import detection_report, render_detection_report

SCORES = ("material_deficit", "direct_deficit", "negative_logprob", "position")
METRICS = ("direct_material", "inherited_material", "history_remainder")


def evaluate(audit, output, manifest, *, bootstrap=200, match_window=32, seed=2026):
    """Same eligible targets for all scorers; source-balanced head comparisons."""
    audit, output = Path(audit), Path(output)
    report = {"primary": SCORES[0], "labels_used_for_lineage": False,
              "labels_used_for_evaluation": True, "trained_detector": False,
              "score_direction_selected_with_labels": False,
              "mechanism_status": "conditional attribution; empirical validation pending",
              "coverage": manifest["analysis_coverage"], "cohorts": {}, "detection": {}}
    grouped = defaultdict(lambda: defaultdict(list))
    for e in manifest["samples"]:
        grouped[e["split"] + "/" + e["task_type"]][e["source_id"]].append(e)
    tokens = defaultdict(lambda: defaultdict(list))
    for group, sources in grouped.items():
        moments = {"raw": Moments(), "matched": Moments()}
        coverage = dict(samples=0, normal=0, hallucinated=0, excluded=0, matched=0)
        for source, entries in tqdm(sources.items(), desc=f"lineage N/H {group}", unit="source"):
            acc = {"raw": Moments(), "matched": Moments()}
            for e in entries:
                with np.load(audit / e["path"]) as data:
                    trace = dict(data)
                with np.load(output / e["path"]) as data:
                    lineage = dict(data)
                with np.load((audit / e["path"]).with_suffix(".labels.npz")) as data:
                    labels = data["labels"]
                start = int(trace["response_start"])
                if len(labels) != len(trace["token_ids"]) - start or not np.isin(labels, (-1, 0, 1)).all():
                    raise ValueError("labels must cover the captured response")
                rows = trace["row_position"][:-1]
                score = np.stack([lineage["score_" + name][:-1] for name in SCORES])
                valid = (np.isin(labels, (0, 1)) & ~trace["special_mask"][rows + 1]
                         & ~trace["special_mask"][rows] & np.isfinite(score).all(0))
                pairs = bracket_positions(labels, token_classes(trace["token_text"][start:]), valid, match_window)
                values = np.stack((lineage["source_margin"][..., :-1, 1],
                                   lineage["material_history_margin"][..., :-1],
                                   lineage["remainder_history_margin"][..., :-1]))
                acc["raw"].add(np.stack([mean(values[..., valid & (labels == y)], -1) for y in (0, 1)]))
                acc["matched"].add(mean(matched_difference(values, pairs), -1))
                coverage["samples"] += 1
                coverage["normal"] += int(((labels == 0) & valid).sum())
                coverage["hallucinated"] += int(((labels == 1) & valid).sum())
                coverage["excluded"] += int((~valid).sum())
                coverage["matched"] += len(pairs)
                target = tokens[e["split"]]
                for name, value in (("labels", labels[valid]), ("query_position", rows[valid]),
                                    ("token_position", rows[valid] + 1)):
                    target[name].append(value)
                for name, value in (("source_id", source), ("sample_id", e["sample_id"]), ("task_type", e["task_type"])):
                    target[name].append(np.repeat(value, valid.sum()))
                for name, value in zip(SCORES, score):
                    target[name].append(value[valid])
            for key in moments:
                moments[key].add(acc[key].finish()["mean"])
        statistics = {"metric_names": np.array(METRICS)}
        for key, moment in moments.items():
            statistics.update({key + "_" + name: value for name, value in moment.finish(inference=True).items()})
        file = output / "cohorts" / (group.replace("/", "_") + ".npz")
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file, **statistics)
        plot_cohort(statistics, file.with_suffix(".png"), group)
        report["cohorts"][group] = {**coverage, "sources": len(sources), "statistics": str(file.relative_to(output))}
    for split, lists in tokens.items():
        data = {key: np.concatenate(value) for key, value in lists.items()}
        np.savez_compressed(output / f"{split}_tokens.npz", **data)
        scores = {key: data[key] for key in SCORES}
        report["detection"][split] = detection_report(data["labels"], scores, data["source_id"], data["task_type"],
                                                     primary=SCORES[0], bootstrap=bootstrap, seed=seed)
        render_detection_report(output / f"{split}_detection.png", data["labels"], scores, data["task_type"], primary=SCORES[0])
    report["replication"] = {}
    for task in sorted({e["task_type"] for e in manifest["samples"]}):
        files = [output / "cohorts" / f"{split}_{task}.npz" for split in ("train", "test")]
        if all(p.exists() for p in files) and all(f"{s}/{task}" in report["cohorts"] for s in ("train", "test")):
            with np.load(files[0]) as train, np.load(files[1]) as test:
                same = train["matched_mean"] * test["matched_mean"] > 0
                supported = same & (train["matched_q_by"] < .05) & (test["matched_q_by"] < .05)
                report["replication"][task] = {name: int(x.sum()) for name, x in zip(METRICS, supported)}
    save_json(output / "summary.json", report)
    lines = ["# 材料消息来源分解与检测评估", "", "这是冻结 attention／归一化和指定 MLP 分配规则下的 value 路径归因。",
             "材料位置是边界输入；尚未验证事实相关性或因果必要性。主分数方向固定，不训练、不按标签挑 head。", "",
             f"已评估 {len(manifest['samples'])}/{manifest['analysis_coverage']['planned_samples']} 个计划样本；部分结果={manifest['analysis_coverage']['partial']}。", "",
             "| split/task | scorer | AUROC | AP | tokens | positives |", "|---|---|---:|---:|---:|---:|"]
    for split, detection in report["detection"].items():
        for task, row in detection["groups"].items():
            for name, score in row["scores"].items():
                number = lambda x: "NA" if x is None else f"{x:.4f}"
                lines.append(f"| {split}/{task} | {name} | {number(score['auroc'])} | {number(score['auprc'])} | {row['known_tokens']} | {row['positives']} |")
            main = row["scores"][SCORES[0]]
            print(f"{split}/{task}: tokens={row['known_tokens']} positives={row['positives']} "
                  f"material_deficit AUROC={main['auroc']} AP={main['auprc']}", flush=True)
    lines += ["", "summary.json 包含 source bootstrap 区间，以及主分数减各对照的配对区间。",
              "所有分数在完全相同的已知、普通 query/target、有限分数 token 上比较；排除数在 cohorts 内。",
              "cohorts/*.npz 保留全部物理 layer/head 的 N/H 均值、位置插值差异、source 区间和 BY 校正。",
              "replication 只数固定相同 head 在 train/test 均通过 BY 且方向一致的条目；不是训练出的检测器。",
              "*_tokens.npz 包含 sample_id/source_id/task_type/query_position/token_position，支持逐 token 核查。",
              "未完成分组不表示零效应；没有材料或有效分数的样本不能算正常。"]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def plot_cohort(data, path, group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(len(METRICS), 3, figsize=(13, 10), squeeze=False)
    for i, name in enumerate(METRICS):
        raw = data["raw_mean"][:, i]
        finite = np.abs(raw[np.isfinite(raw)])
        scale = max(float(finite.max()) if len(finite) else 0, 1e-8)
        for j, title in enumerate(("Normal", "Hallucinated", "H - interpolated N")):
            value = raw[j] if j < 2 else data["matched_mean"][i]
            ax = axes[i, j]
            if j == 2:
                finite = np.abs(value[np.isfinite(value)])
                scale = max(float(finite.max()) if len(finite) else 0, 1e-8)
            m = ax.imshow(value, origin="lower", aspect="auto", cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#eeeeee"), vmin=-scale, vmax=scale)
            if j == 2:
                layer, head = np.where(data["matched_q_by"][i] < .05)
                ax.scatter(head, layer, s=5, c="black")
            ax.set(title=f"{name}: {title}", xlabel="Physical head", ylabel="Layer")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if np.isfinite(value).any():
                fig.colorbar(m, ax=ax, shrink=.75)
            else:
                ax.text(.5, .5, "No matched support" if j == 2 else "No eligible tokens",
                        transform=ax.transAxes, ha="center", va="center")
    fig.suptitle(group + " | signed direct readout; dots: source-level BY q < .05")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def render_sample(audit_path, lineage_path, layer=None, head=None):
    """Self-contained target selector for a physical reader head, all N/H text."""
    with np.load(audit_path) as file:
        trace = dict(file)
    with np.load(lineage_path) as file:
        flow = dict(file)
    with np.load(Path(audit_path).with_suffix(".labels.npz")) as file:
        labels = file["labels"]
    l, h, r = flow["material_history_margin"].shape
    if layer is None or head is None:
        # Label-blind display choice only; all head statistics remain in NPZ.
        layer, head = np.unravel_index(np.abs(flow["material_history_margin"][..., :-1]).sum(-1).argmax(), (l, h))
    layer, head = int(layer), int(head)
    start = int(trace["response_start"])
    pieces = []
    for pos, token in enumerate(trace["token_text"]):
        label = int(labels[pos-start]) if pos >= start else -1
        color = "#ddd" if trace["special_mask"][pos] else "#ffd1d1" if label == 1 else "#d8f2df" if label == 0 else "transparent"
        pieces.append(f'<span title="position={pos}; label={label}" style="background:{color}">{escape(str(token))}</span>')
    options = ''.join(f'<option value="{i}">token {int(q)+1}: {escape(str(trace["token_text"][q+1]))} [{"H" if labels[i]==1 else "N" if labels[i]==0 else "?"}]</option>'
                      for i, q in enumerate(trace["row_position"][:-1]))
    payload = {"tokens": trace["token_text"].tolist(), "queries": trace["row_position"][:-1].tolist()}
    for kind in ("direct", "relay"):
        for name in ("position", "native", "material"):
            payload[kind + "_" + name] = flow[f"top_{kind}_{name}"][layer, head, :-1].tolist()
    # Do not join an incoming edge selected for b+1 to an outgoing edge for q+1:
    # these are different output contrasts. The table shows only one target q.
    script = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    html = f'''<!doctype html><meta charset="utf-8"><title>Message lineage</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.65}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap}}table{{border-collapse:collapse;width:100%}}</style>
<h1>{escape(str(trace['sample_id']))}: reader L{layer}H{head}</h1>
<p>红：幻觉；绿：正常；灰：特殊 token。q 的消息对应预测 token q+1。当前 head 仅供展示；统计保留全部 head。</p>
<p>材料来源分量经过指定的 value／MLP 分解传播，不等于事实相关性或因果效应。表中仅显示保存的最大绝对读出边，缺席不代表没有路径；没有回看峰筛选。</p>
<h2>完整输入与回答</h2><pre>{''.join(pieces)}</pre>
<label>预测目标 <select id="target">{options}</select></label>
<h2>该目标的来源消息</h2><table><thead><tr><th>边类型</th><th>来源位置／文本</th><th>原生消息读出</th><th>材料分量读出</th><th>剩余分量</th></tr></thead><tbody id="edges"></tbody></table>
<script>const data={script}; const select=document.getElementById('target');
function show(){{const q=Number(select.value), body=document.getElementById('edges');body.replaceChildren();
for(const kind of ['direct','relay']){{for(let k=0;k<data[kind+'_position'][q].length;k++){{
const s=data[kind+'_position'][q][k];if(s<0)continue;
const native=data[kind+'_native'][q][k], material=data[kind+'_material'][q][k];
const row=document.createElement('tr');for(const value of [kind==='direct'?'直接材料读取':'历史载体读取',s+': '+data.tokens[s],native.toFixed(5),material.toFixed(5),(native-material).toFixed(5)]){{const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);}}body.appendChild(row);}}}}}}
select.addEventListener('change',show);show();</script>'''
    target = Path(lineage_path).with_suffix(".html")
    target.write_text(html, encoding="utf-8")
    return target
