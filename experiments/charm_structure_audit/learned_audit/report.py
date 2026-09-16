"""Straightforward role-based reports; no test-fitted threshold or span expansion."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


ROLE_NAMES = ("first_error", "later_onset", "continuation", "normal")


def token_roles(sample):
    error = sample["gold"].astype(bool)
    onset = sample["onset"].astype(bool)
    first = np.zeros(len(error), bool)
    positions = np.flatnonzero(error)
    if len(positions):
        first[positions[0]] = True
    return dict(first_error=first, later_onset=onset & ~first,
                continuation=error & ~onset, normal=~error)


def binary_metrics(labels, logits, threshold, scores=None):
    labels = np.asarray(labels, bool)
    scores = expit(logits) if scores is None else scores
    alarm = scores > threshold
    positive, negative = int(labels.sum()), int((~labels).sum())
    return dict(tokens=len(labels), positives=positive, negatives=negative,
        auroc=float(roc_auc_score(labels, scores)) if positive and negative else None,
        ap=float(average_precision_score(labels, scores)) if positive else None,
        recall=float(alarm[labels].mean()) if positive else None,
        fpr=float(alarm[~labels].mean()) if negative else None)


def source_interval(values, sources, bootstrap):
    """First average within source; confidence interval is for mean logit change."""
    unique = np.unique(sources)
    means = np.array([values[sources == source].mean() for source in unique])
    if not len(means):
        return dict(sources=0, mean=None, ci95=None)
    rng = np.random.default_rng(0)
    draws = [rng.choice(means, len(means), replace=True).mean() for _ in range(bootstrap)]
    return dict(sources=len(means), mean=float(means.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist() if draws and len(means) > 1 else None)


def summarize_interventions(samples, threshold, bootstrap):
    base = np.concatenate([sample["logits"] for sample in samples])
    base_scores = np.concatenate([sample["score"] for sample in samples])
    sources = np.concatenate([np.repeat(sample["source_id"], len(sample["gold"])) for sample in samples])
    masks = {name: np.concatenate([token_roles(sample)[name] for sample in samples]) for name in ROLE_NAMES}
    names = sorted(samples[0]["controls"])
    if any(set(sample["controls"]) != set(names) for sample in samples):
        raise ValueError("completed samples contain different control experiments")
    result = {}
    for name in names:
        changed = np.concatenate([sample["controls"][name] for sample in samples])
        difference = changed - base
        result[name] = {}
        for role, mask in masks.items():
            selected = mask | masks["normal"]
            labels = mask[selected] if role != "normal" else np.zeros(selected.sum(), bool)
            original = binary_metrics(labels, base[selected], threshold, base_scores[selected])
            altered = binary_metrics(labels, changed[selected], threshold)
            result[name][role] = dict(baseline=original, control=altered,
                delta_auroc=altered["auroc"] - original["auroc"] if original["auroc"] is not None else None,
                delta_ap=altered["ap"] - original["ap"] if original["ap"] is not None else None,
                mean_absolute_logit_change=float(abs(difference[mask]).mean()) if mask.any() else None,
                source_mean_logit_change=source_interval(difference[mask], sources[mask], bootstrap))
    return result


def paired_control_contrasts(samples, bootstrap):
    """Direct paired contrast at fixed RNG seed or GNN layer, not two unrelated CIs."""
    names = set.intersection(*[set(sample["controls"]) for sample in samples])
    pairs = []
    for name in sorted(names):
        if name.startswith("head_s"):
            pairs.append((name.replace("head_s", "layer_s"), name))
        if name.startswith("layer_s"):
            pairs.append((name.replace("layer_s", "vector_s"), name))
        if name.startswith("other_label_g"):
            pairs.append((name.replace("other_label_g", "same_label_g"), name))
    result = {}
    for left, right in pairs:
        rows = {}
        for role in ROLE_NAMES:
            values, sources = [], []
            for sample in samples:
                mask = token_roles(sample)[role]
                values.extend((sample["controls"][right] - sample["controls"][left])[mask])
                sources.extend([sample["source_id"]] * int(mask.sum()))
            selected_logits_left, selected_logits_right, labels = [], [], []
            for sample in samples:
                role_masks = token_roles(sample)
                chosen = role_masks[role] | role_masks["normal"]
                label = role_masks[role] if role != "normal" else np.zeros(len(chosen), bool)
                selected_logits_left.extend(sample["controls"][left][chosen])
                selected_logits_right.extend(sample["controls"][right][chosen])
                labels.extend(label[chosen])
            left_metrics = binary_metrics(labels, np.asarray(selected_logits_left), .5)
            right_metrics = binary_metrics(labels, np.asarray(selected_logits_right), .5)
            rows[role] = dict(source_mean_logit_change=source_interval(np.asarray(values), np.asarray(sources), bootstrap),
                delta_auroc=right_metrics["auroc"] - left_metrics["auroc"] if left_metrics["auroc"] is not None else None,
                delta_ap=right_metrics["ap"] - left_metrics["ap"] if left_metrics["ap"] is not None else None)
        result[right + "_minus_" + left] = rows
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(samples, output, threshold, bootstrap):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = summarize_interventions(samples, threshold, bootstrap)
    contrasts = paired_control_contrasts(samples, bootstrap)
    labels = np.concatenate([sample["gold"] for sample in samples])
    logits = np.concatenate([sample["logits"] for sample in samples])
    original_scores = np.concatenate([sample["score"] for sample in samples])
    result = dict(responses=len(samples), threshold=threshold, baseline=binary_metrics(labels, logits, threshold, original_scores), interventions=metrics,
        paired_contrasts=contrasts, diagnostics={sample["id"]: sample["diagnostics"] for sample in samples},
        interpretation="Frozen-detector sensitivity, not new detection performance or LLM causal mechanism.")
    write_json(output / "mechanisms.json", result)
    token_rows = []
    for sample in samples:
        for role, mask in token_roles(sample).items():
            for token in np.flatnonzero(mask):
                start, end = sample["offsets"][token]
                row = dict(id=sample["id"], source_id=sample["source_id"], token=int(token), role=role,
                    text=sample["response"][start:end], score=float(sample["score"][token]))
                for name in sorted(sample["controls"]):
                    row[name + "_delta_logit"] = float(sample["controls"][name][token] - sample["logits"][token])
                token_rows.append(row)
    write_csv(output / "token_effects.csv", token_rows)
    write_readable_report(result, output)
    return result


def write_readable_report(result, output):
    baseline = result["baseline"]
    lines = ["# CHARM：改变哪类信息后，判断发生变化？", "",
        f"原分数复核：{baseline['tokens']} tokens，AUROC={baseline['auroc']}，AP={baseline['ap']}。", "",
        "AUROC变化是控制减原模型；负数表示在这一扰动下变差。",
        "分数变化不能独自证明机制。重排后分布可能不自然；先检查实际改动和匹配覆盖。", "",
        "| 控制 | 首错AUROC变化 | 续写AUROC变化 |", "|---|---:|---:|"]
    def display(value):
        return "缺测" if value is None else f"{value:+.5f}"
    for name, rows in result["interventions"].items():
        lines.append(f"| {name} | {display(rows['first_error']['delta_auroc'])} | {display(rows['continuation']['delta_auroc'])} |")
    lines.extend(["", "## 怎么读", "",
        "vector打乱整条多头向量与邻居的配对；layer进一步打乱不同层在同一邻居上的配对；head进一步打乱同层各头的配对。",
        "head比layer更伤排序，且熵保持不变，才支持模型使用熵以外的同层多头关系。不能只比较分数绝对改变量。",
        "positive_head只打乱正权重，非零位置不动，可辅助检查效果是否只是稀疏模式变化。",
        "same_label/other_label只替换来源状态，两个实验动完全相同的边。coverage很低时不能据无变化下结论。",
        "这些来源标签只用于解释实验，不进入原检测器。g0/g1/g2是图网络层，不是LLM层。",
        "head_names改变物理head编号，说明通道身份敏感性，不证明头间合作。",
        "原模型和校准阈值没有修改；首错低召回与高整体排序仍须分别报告。"])
    (Path(output) / "README_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
