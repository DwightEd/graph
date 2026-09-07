"""Describe frozen, label-free computation modes and audit their associations.

Labels enter only the association report. Signed margins explain support for
the observed token against a fixed alternative, not truth or causal necessity.
Every mode is reported; no head or mode is selected by hallucination labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .routing_events import position_adjusted_gap


def _finite_list(value):
    array = np.asarray(value)
    result = array.astype(object)
    result[~np.isfinite(array)] = None
    return result.tolist()


def _bh_adjust(pvalues):
    """BH adjustment includes unsupported modes as p=1 in the family size."""

    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.minimum.accumulate(
        (pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def _source_inference(values, *, seed, bootstrap):
    """Whole-source bootstrap and two-sided paired source sign-flip test.

    Sign flips assume exchangeable signs of the source-level matched gaps;
    this observational assumption does not establish a causal effect.
    """

    values = np.asarray(values, dtype=float)
    n_sources, n_patterns = values.shape
    empty = np.full(n_patterns, np.nan)
    if not n_sources:
        return empty, empty.copy(), empty.copy(), empty.copy()
    mean = values.mean(0)
    if n_sources < 2:
        return mean, empty.copy(), empty.copy(), empty.copy()
    rng = np.random.default_rng(seed)
    draws = np.stack([
        values[rng.integers(n_sources, size=n_sources)].mean(0)
        for _ in range(bootstrap)
    ])
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    exceed = np.zeros(n_patterns, dtype=int)
    exact = n_sources <= 14
    repeats = 2 ** n_sources if exact else max(9999, bootstrap)
    for start in range(0, repeats, 256):
        count = min(256, repeats - start)
        if exact:
            masks = np.arange(start, start + count, dtype=np.uint64)[:, None]
            signs = 2.0 * ((masks >> np.arange(n_sources, dtype=np.uint64)) & 1) - 1
        else:
            signs = 2.0 * rng.integers(2, size=(count, n_sources)) - 1
        permuted = signs @ values / n_sources
        exceed += (np.abs(permuted) >= np.abs(mean) - 1e-12).sum(0)
    pvalue = exceed / repeats if exact else (exceed + 1) / (repeats + 1)
    return mean, lower, upper, pvalue


@dataclass
class _Source:
    sample_count: int = 0
    paired_samples: int = 0
    gap_sum: np.ndarray | None = None
    mode_samples: np.ndarray | None = None
    signed_sum: dict = field(default_factory=dict)
    transition_sum: np.ndarray | None = None
    transition_samples: int = 0

    def merge(self, other):
        """Pool the same source across tasks before any source-level inference."""

        self.sample_count += other.sample_count
        self.paired_samples += other.paired_samples
        if other.gap_sum is not None:
            self.gap_sum = other.gap_sum.copy() if self.gap_sum is None else self.gap_sum + other.gap_sum
        self.mode_samples = other.mode_samples.copy() if self.mode_samples is None else self.mode_samples + other.mode_samples
        for name, values in other.signed_sum.items():
            self.signed_sum[name] = self.signed_sum.get(name, 0) + values
        if other.transition_sum is not None:
            self.transition_sum = (other.transition_sum.copy() if self.transition_sum is None
                                   else self.transition_sum + other.transition_sum)
        self.transition_samples += other.transition_samples

    def add(self, gap, modes, trace, transition_counts):
        self.sample_count += 1
        if gap is not None:
            self.paired_samples += 1
            self.gap_sum = gap.copy() if self.gap_sum is None else self.gap_sum + gap
        if self.mode_samples is None:
            self.mode_samples = np.zeros(len(modes), dtype=int)
        self.mode_samples += modes.any(-1)
        if transition_counts.sum():
            frequency = transition_counts / transition_counts.sum()
            self.transition_sum = frequency if self.transition_sum is None else self.transition_sum + frequency
            self.transition_samples += 1
        for name in ("head_margin", "mlp_margin", "stage_margin"):
            values = np.asarray(trace[name], dtype=float)
            sample_mean = np.zeros((len(modes), *values.shape[:-1]))
            for mode, mask in enumerate(modes):
                if mask.any():
                    sample_mean[mode] = values[..., mask].mean(-1)
            if name not in self.signed_sum:
                self.signed_sum[name] = sample_mean
            else:
                self.signed_sum[name] += sample_mean


def _closure(trace):
    """Numerical identities check scalar bookkeeping, not semantic fidelity."""

    head = np.asarray(trace["head_margin"], dtype=float)
    attention = np.asarray(trace["attention_margin"], dtype=float)
    stage = np.asarray(trace["stage_margin"], dtype=float)
    post = np.asarray(trace["post_attention_margin"], dtype=float)
    errors = {
        "head_sum_to_attention": np.max(np.abs(head.sum(1) - attention)),
        "attention_residual_update": np.max(np.abs(stage[:-1] + attention - post)),
        "mlp_residual_update": np.max(np.abs(post + trace["mlp_margin"] - stage[1:])),
        "final_stage_to_margin": np.max(np.abs(stage[-1] - trace["final_margin"])),
    }
    corrected = {
        "head_sum_to_attention": np.max(np.abs(
            head.sum(1) + trace.get("head_remainder_margin", 0) - attention
        )),
        "attention_residual_update": np.max(np.abs(
            stage[:-1] + attention + trace.get("attention_add_remainder_margin", 0) - post
        )),
        "mlp_residual_update": np.max(np.abs(
            post + trace["mlp_margin"] + trace.get("mlp_add_remainder_margin", 0) - stage[1:]
        )),
        "final_stage_to_margin": np.max(np.abs(
            stage[-1] + trace.get("readout_bias", 0) + trace.get("readout_remainder", 0) - trace["final_margin"]
        )),
    }
    if "edge_margin" in trace and "omitted_margin" in trace:
        errors["stored_plus_omitted_edges_to_head"] = np.max(np.abs(
            np.asarray(trace["edge_margin"]).sum(-1) + trace["omitted_margin"] - head
        ))
        corrected["stored_plus_omitted_edges_to_head"] = errors["stored_plus_omitted_edges_to_head"]
    if "edge_sum_margin" in trace:
        errors["all_edges_to_head"] = np.max(np.abs(trace["edge_sum_margin"] - head))
        corrected["all_edges_to_head"] = np.max(np.abs(
            trace["edge_sum_margin"] + trace["edge_rounding_remainder"] - head
        ))
    return tuple({name: float(value) for name, value in items.items()} for items in (errors, corrected))


class NativeCohort:
    """Streaming per-source sufficient summaries; never retains sample traces."""

    def __init__(self, n_patterns, seed=2026, bootstrap=200):
        if n_patterns < 1 or bootstrap < 2:
            raise ValueError("n_patterns must be positive and bootstrap at least two")
        self.n_patterns = n_patterns
        self.seed = seed
        self.bootstrap = bootstrap
        self.sources = defaultdict(dict)
        self.counts = defaultdict(lambda: np.zeros((n_patterns, 3), dtype=int))
        self.transitions = defaultdict(lambda: np.zeros((n_patterns, n_patterns), dtype=int))
        self.closure_max = {}
        self.corrected_closure_max = {}
        self.closure_samples = 0

    def add(self, sample_id, source_id, task_type, response_index, labels, prediction, trace):
        """Add one sample once; labels -1 are descriptive-only unknown tokens."""

        del sample_id  # Identity stays in per-sample artifacts, not this accumulator.
        pattern = np.asarray(prediction["pattern_id"], dtype=int)
        labels = np.asarray(labels)
        if pattern.shape != labels.shape or len(pattern) != len(response_index):
            raise ValueError("predictions, response indices and labels must align")
        if np.any((pattern < 0) | (pattern >= self.n_patterns)):
            raise ValueError("pattern_id is outside the frozen model's mode range")
        modes = np.arange(self.n_patterns)[:, None] == pattern
        gap = position_adjusted_gap(modes.astype(float), labels, response_index)
        previous = np.asarray(prediction.get("has_previous", np.r_[False, np.diff(trace["row_position"]) == 1]))
        rows = np.flatnonzero(previous & (np.arange(len(pattern)) > 0))
        transitions = np.zeros((self.n_patterns, self.n_patterns), dtype=int)
        np.add.at(transitions, (pattern[rows - 1], pattern[rows]), 1)
        self.transitions[task_type] += transitions
        source = self.sources[task_type].setdefault(str(source_id), _Source())
        source.add(gap, modes, trace, transitions)
        for group, mask in enumerate((labels == 0, labels == 1, ~np.isin(labels, (0, 1)))):
            self.counts[task_type][:, group] += modes[:, mask].sum(-1)
        raw, corrected = _closure(trace)
        for destination, errors in ((self.closure_max, raw), (self.corrected_closure_max, corrected)):
            for name, value in errors.items():
                destination[name] = max(destination.get(name, 0), value)
        self.closure_samples += 1

    def _task_report(self, sources, counts, transitions):
        paired = [s.gap_sum / s.paired_samples for s in sources if s.paired_samples]
        gaps = np.asarray(paired).reshape(-1, self.n_patterns)
        mean, lower, upper, pvalues = _source_inference(
            gaps, seed=self.seed, bootstrap=self.bootstrap,
        )
        qvalues = _bh_adjust(np.where(np.isfinite(pvalues), pvalues, 1))
        descriptions = {}
        mode_sources = np.zeros(self.n_patterns, dtype=int)
        mode_samples = np.zeros(self.n_patterns, dtype=int)
        for source in sources:
            present = source.mode_samples > 0
            mode_sources += present
            mode_samples += source.mode_samples
            for name, sums in source.signed_sum.items():
                divisor = source.mode_samples.reshape((-1,) + (1,) * (sums.ndim - 1))
                source_mean = np.divide(sums, divisor, out=np.zeros_like(sums), where=divisor > 0)
                descriptions[name] = descriptions.get(name, 0) + source_mean
        modes = []
        for mode in range(self.n_patterns):
            modes.append({
                "pattern_id": mode,
                "tokens_nonhallucinated": int(counts[mode, 0]),
                "tokens_hallucinated": int(counts[mode, 1]),
                "tokens_unknown": int(counts[mode, 2]),
                "containing_samples": int(mode_samples[mode]),
                "containing_sources": int(mode_sources[mode]),
                "paired_sources": len(gaps),
                "position_matched_h_minus_n": _finite_list(mean[mode]),
                "ci_lower": _finite_list(lower[mode]),
                "ci_upper": _finite_list(upper[mode]),
                "signflip_p": _finite_list(pvalues[mode]),
                "bh_q_within_task": float(qvalues[mode]) if np.isfinite(pvalues[mode]) else None,
                "positive_gap_sources": int((gaps[:, mode] > 0).sum()),
                "negative_gap_sources": int((gaps[:, mode] < 0).sum()),
                "zero_gap_sources": int((gaps[:, mode] == 0).sum()),
                "signed_means": {
                    name: _finite_list(total[mode] / mode_sources[mode])
                    if mode_sources[mode] else None
                    for name, total in descriptions.items()
                },
            })
        return {
            "samples": sum(s.sample_count for s in sources),
            "sources": len(sources), "paired_sources": len(gaps),
            "paired_samples": sum(s.paired_samples for s in sources),
            "transition_counts": transitions.tolist(),
            "transition_pairs": int(transitions.sum()),
            "transition_sources": sum(s.transition_samples > 0 for s in sources),
            "source_balanced_transition_frequency": _finite_list(np.mean([
                s.transition_sum / s.transition_samples for s in sources if s.transition_samples
            ], axis=0)) if any(s.transition_samples for s in sources) else None,
            "patterns": modes,
        }

    def report(self):
        """Return JSON-safe all-mode statistics without ranking or filtering."""

        tasks = {
            task: self._task_report(list(sources.values()), self.counts[task], self.transitions[task])
            for task, sources in sorted(self.sources.items())
        }
        if tasks:
            pooled = defaultdict(_Source)
            for sources in self.sources.values():
                for source_id, source in sources.items():
                    pooled[source_id].merge(source)
            tasks["ALL"] = self._task_report(
                list(pooled.values()),
                sum(self.counts.values()),
                sum(self.transitions.values()),
            )
        family = [mode for task in tasks.values() for mode in task["patterns"]]
        qvalues = _bh_adjust([1 if mode["signflip_p"] is None else mode["signflip_p"] for mode in family])
        for mode, qvalue in zip(family, qvalues):
            mode["bh_q"] = float(qvalue) if mode["signflip_p"] is not None else None
        return {
            "schema": 1, "status": "observational_native_computation_audit",
            "n_patterns": self.n_patterns, "tasks": tasks,
            "numerical_closure": {
                "samples": self.closure_samples, "raw_max_abs": self.closure_max,
                "remainder_corrected_max_abs": self.corrected_closure_max,
                "meaning": "Raw discrepancies include output bias and native dtype rounding; corrected checks include recorded remainders. Bookkeeping closure is not an independent semantic test.",
            },
            "interpretation": {
                "mode_discovery": "Frozen before labels; every frozen mode is reported, including absent modes.",
                "signed_margin": "Support for the observed token against the frozen native alternative; neither factual support nor causal necessity.",
                "association": "H-N mode prevalence within sample log2(response_index+1) bins containing both labels; equal bins, samples within source, then sources.",
                "uncertainty": f"{self.bootstrap} whole-source bootstrap draws; pointwise 95% intervals, conditional on the frozen model.",
                "multiplicity": "Two-sided source sign-flip p; bh_q uses Benjamini-Hochberg across the full task-by-mode family including ALL and absent modes. bh_q_within_task is supplementary. Assumes exchangeable source-gap signs and BH dependence conditions.",
                "source_pooling": "ALL pools every sample with the same source_id across tasks before equal-source inference.",
                "signed_means": "Equal samples within each source containing a mode, then equal such sources; all labels included; individual head axes retained.",
                "transitions": "Frozen mode before→after, only has_previous rows; diagonal counts are persistence. Raw counts plus equal-sample-then-source frequencies; labels do not define transitions.",
                "limits": "Coarse position matching leaves residual confounding; modes need not represent factual mechanisms. No four failure classes are assigned.",
            },
        }

    def plot(self, path, report=None):
        render_native_cohort(path, self.report() if report is None else report)


def render_native_cohort(path, report):
    """All frozen modes with effect sizes and source counts, no cherry-picking."""

    import matplotlib.pyplot as plt

    tasks = list(report["tasks"])
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        rows = report["tasks"][task]["patterns"]
        for row in rows:
            value, low, high = (row[name] for name in (
                "position_matched_h_minus_n", "ci_lower", "ci_upper",
            ))
            y = row["pattern_id"]
            if value is not None:
                ax.plot(value * 100, y, "o", color="#365f8d")
            if low is not None:
                ax.plot([low * 100, high * 100], [y, y], color="#365f8d")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(range(len(rows)), [f"Mode {r['pattern_id']}" for r in rows])
        ax.set_title(f"{task}\n{report['tasks'][task]['paired_sources']} paired sources")
        ax.set_xlabel("Within-position H − N prevalence (pp)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Frozen computation modes: observational association, all modes")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_native_mode_writes(path, report, task="ALL"):
    """All modes' conditional signed readout means, not vector centroids.

    The final contrast varies across tokens. This view explains observed-token
    support inside each mode; it is not a shared semantic direction or proof of
    information preservation. The underlying head dimension remains explicit.
    """

    import matplotlib.pyplot as plt

    modes = report["tasks"][task]["patterns"]
    observed = [mode for mode in modes if mode["signed_means"].get("head_margin") is not None]
    if not observed:
        return
    limits = {
        name: max(max(float(np.max(np.abs(mode["signed_means"][name]))) for mode in observed), 1e-8)
        for name in ("head_margin", "mlp_margin", "stage_margin")
    }
    fig, axes = plt.subplots(len(modes), 3, figsize=(14, 3.3 * len(modes)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.5, 1, 1]})
    for mode, row in zip(modes, axes):
        means = mode["signed_means"]
        if means.get("head_margin") is None:
            for ax in row:
                ax.text(0.5, 0.5, f"Mode {mode['pattern_id']}: not observed", ha="center", va="center")
                ax.set_axis_off()
            continue
        head = np.asarray(means["head_margin"])
        layers, heads = head.shape
        view = row[0].imshow(head, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                            vmin=-limits["head_margin"], vmax=limits["head_margin"])
        row[0].set_title(f"Mode {mode['pattern_id']}: individual head mean writes\n"
                         f"{mode['containing_samples']} samples / {mode['containing_sources']} sources")
        row[0].set_xticks(np.arange(0, heads, max(1, heads // 8)))
        row[0].set_yticks(np.arange(0, layers, max(1, layers // 8)))
        row[0].set_xlabel("Head (zero-based)")
        row[0].set_ylabel("Layer (zero-based)")
        fig.colorbar(view, ax=row[0], fraction=0.025, pad=0.02, label="mean signed margin")
        mlp = np.asarray(means["mlp_margin"])
        row[1].barh(np.arange(layers), mlp, color=np.where(mlp >= 0, "#b04458", "#376b9e"))
        row[1].set_xlim(-limits["mlp_margin"], limits["mlp_margin"])
        row[1].set_ylim(layers - 0.5, -0.5)
        row[1].set_title("MLP mean writes")
        row[1].set_ylabel("Layer")
        row[2].plot(means["stage_margin"], np.arange(layers + 1), "o-", markersize=3, color="#414b58")
        row[2].set_xlim(-limits["stage_margin"], limits["stage_margin"])
        row[2].set_ylim(layers + 0.5, -0.5)
        row[2].set_title("Residual stage mean support")
        row[2].set_ylabel("Stage (0 = embedding)")
        for ax in row[1:]:
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_xlabel("Mean signed margin")
            ax.grid(axis="x", alpha=0.2)
            ax.locator_params(axis="x", nbins=3)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    fig.suptitle(f"{task}: conditional signed readout means for every frozen mode\n"
                 "Observed/runner tokens vary across rows; equal-source means, not vector centroids or factual attribution",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def native_transition_examples(trace, prediction, maximum=8, separation=3):
    """Largest projected changes for inspection, not thresholded discoveries.

    Ranking uses the frozen model's consecutive-row change. Head/MLP identities
    are located by sketch-vector change; the norms rank the display only, while
    the corresponding signed vectors are retained for interpretation.
    """

    positions = np.asarray(trace["row_position"])
    scores = np.asarray(prediction["transition"])
    previous = np.asarray(prediction.get("has_previous", np.r_[False, np.diff(positions) == 1]))
    eligible = (scores > 0) & previous & (np.arange(len(positions)) > 0)
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(-scores[ranked], kind="stable")]
    selected, result = [], []
    for row in ranked:
        if any(abs(positions[row] - positions[other]) < separation for other in selected):
            continue
        selected.append(int(row))
        event = {
            "predictor_position": int(positions[row]),
            "response_index": int(positions[row] + 1 - int(trace["response_start"])),
            "before_pattern_id": int(prediction["pattern_id"][row - 1]),
            "after_pattern_id": int(prediction["pattern_id"][row]),
            "projected_transition": float(scores[row]),
        }
        if "head_sketch" in trace:
            change = np.asarray(trace["head_sketch"])[:, :, row] - trace["head_sketch"][:, :, row - 1]
            lengths = np.linalg.norm(change, axis=-1)
            layer, head = np.unravel_index(np.argmax(lengths), lengths.shape)
            event["largest_head_sketch_change"] = {
                "layer": int(layer), "head": int(head), "norm": float(lengths[layer, head]),
                "signed_delta": change[layer, head].tolist(),
            }
        if "mlp_sketch" in trace:
            change = np.asarray(trace["mlp_sketch"])[:, row] - trace["mlp_sketch"][:, row - 1]
            lengths = np.linalg.norm(change, axis=-1)
            layer = int(np.argmax(lengths))
            event["largest_mlp_sketch_change"] = {
                "layer": layer, "norm": float(lengths[layer]), "signed_delta": change[layer].tolist(),
            }
        result.append(event)
        if len(result) == maximum:
            break
    return result


def render_native_sample(path, trace, prediction, tokenizer=None, labels=None):
    """Full trajectory plus a readable local computation view at its largest change.

    The display's head subset and focus use no hallucination labels. All heads
    remain in capture/model/report. Positive/negative edges are signed observed
    token support, not evidence provenance. Returns the plotted edge metadata.
    """

    import matplotlib.pyplot as plt

    def token_text(position):
        if "token_text" in trace:
            return str(trace["token_text"][position])
        token_id = int(trace["token_ids"][position])
        return tokenizer.decode([token_id]) if tokenizer is not None else str(token_id)

    positions = np.asarray(trace["row_position"]) + 1 - int(trace["response_start"])
    coords = np.asarray(prediction["coords"])
    modes = np.asarray(prediction["pattern_id"])
    transition = np.asarray(prediction["transition"])
    focus = int(np.argmax(transition))
    window = slice(max(0, focus - 32), min(len(positions), focus + 33))
    x = positions[window]
    head = np.asarray(trace["head_margin"])
    flat = head.reshape(-1, head.shape[-1])
    displayed = np.argsort(-np.max(np.abs(flat[:, window]), axis=1), kind="stable")[:12]
    n_heads = head.shape[1]
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [2, 3, 3, 3]})
    for component in range(min(2, coords.shape[1])):
        axes[0].plot(positions, coords[:, component], lw=0.8, label=f"coordinate {component + 1}")
    axes[0].scatter(positions, np.full(len(positions), axes[0].get_ylim()[0]),
                    c=plt.get_cmap("tab20")(modes % 20), s=9)
    axes[0].axvline(positions[focus], color="black", linestyle="--", label="largest transition")
    if labels is not None:
        for position in positions[np.asarray(labels) == 1]:
            axes[0].axvspan(position - 0.5, position + 0.5, color="crimson", alpha=0.09)
    axes[0].set_title("Full-response frozen coordinates; bottom colors = mode; red shading = annotation only")
    axes[0].legend(loc="upper left", ncols=3)
    axes[0].set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    views = [flat[displayed, window], np.asarray(trace["stage_margin"])[:, window],
             np.asarray(trace["mlp_margin"])[:, window]]
    titles = [f"Individual head signed writes ({len(displayed)} displayed by absolute activity; no head averaging)",
              "Residual support (each column fixes its observed-token / runner contrast)", "MLP signed writes"]
    for index, (ax, values, title) in enumerate(zip(axes[1:], views, titles)):
        limit = max(float(np.max(np.abs(values))), 1e-8)
        view = ax.imshow(values, origin="upper", aspect="auto", cmap="RdBu_r",
                         vmin=-limit, vmax=limit, interpolation="nearest",
                         extent=(x[0] - 0.5, x[-1] + 0.5, len(values) - 0.5, -0.5))
        ax.axvline(positions[focus], color="black", linestyle="--", lw=0.8)
        ax.set_title(title)
        if index == 0:
            ax.set_yticks(range(len(displayed)), [f"L{i // n_heads} H{i % n_heads}" for i in displayed])
        else:
            ticks = np.linspace(0, len(values) - 1, min(9, len(values)), dtype=int)
            ax.set_yticks(ticks)
            ax.set_ylabel("Residual stage" if index == 1 else "Layer")
        fig.colorbar(view, ax=ax, fraction=0.02, pad=0.01, label="signed margin")
    for ax in axes:
        ax.set_xlabel("Predicted response token index (zero-based)")
    edges = []
    if "edge_margin" in trace:
        values = np.asarray(trace["edge_margin"])[:, :, focus]
        source = np.asarray(trace["edge_source_position"])[:, :, focus]
        valid = source >= 0
        for sign in (1, -1):
            candidates = np.flatnonzero(valid & (sign * values > 0))
            ranked = candidates[np.argsort(-sign * values.ravel()[candidates], kind="stable")[:2]]
            for entry in ranked:
                layer, h, edge = np.unravel_index(entry, values.shape)
                position = int(source[layer, h, edge])
                edges.append({"layer": int(layer), "head": int(h), "source_position": position,
                              "source_token": token_text(position), "signed_margin": float(values[layer, h, edge])})
    edge_text = "; ".join(
        f"L{e['layer']}H{e['head']} ← {e['source_position']}:{e['source_token']!r} ({e['signed_margin']:+.3g})"
        for e in edges
    )
    omitted = float(np.asarray(trace.get("omitted_margin", np.zeros_like(head)))[:, :, focus].sum())
    target = token_text(int(trace["row_position"][focus]) + 1)
    fig.text(0.05, 0.018, f"Focus q={int(trace['row_position'][focus])} → {target!r}, mode {int(modes[focus])}; omitted signed sum={omitted:+.3g} (can cancel). Stored ± edges: {edge_text}",
             fontsize=8, wrap=True, parse_math=False)
    fig.suptitle("Observed-token support and computation changes — not causal or factual attribution", fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"focus_response_index": int(positions[focus]), "focus_predictor_position": int(trace["row_position"][focus]),
            "displayed_layer_head": [[int(i // n_heads), int(i % n_heads)] for i in displayed],
            "omitted_signed_sum": omitted, "focus_token_text": target, "edges": edges,
            "transition_examples": native_transition_examples(trace, prediction)}
