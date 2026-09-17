"""Figures for independent node signals versus full-model behavior."""

from pathlib import Path

import numpy as np

from .visualize import finish


def plot_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def comparison_plots(positions, counts, matched, output):
    plt = plot_module()
    output = Path(output)
    regions = ['early_third', 'middle_third', 'late_third']
    figure, ax = plt.subplots(figsize=(8, 4.8))
    for model, group in positions.groupby('model', sort=False):
        selected = group.set_index('region').loc[regions]
        label = model + ' | hits: ' + ', '.join(f'{r.hits}/{r.tokens}' for r in selected.itertuples())
        ax.plot(range(3), selected.recall, marker='o', label=label)
    ax.set_xticks(range(3), ['Early third', 'Middle third', 'Late third'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Error-token recall at each model\'s original threshold')
    ax.set_title('Node-only versus full CHARM: where are errors detected?')
    ax.legend(fontsize=8)
    finish(figure, output/'node_vs_full_position_recall')
    decision_plot(counts, output/'decision_overlap')
    if not matched.empty:
        matched_score_plot(matched, output/'matched_position_scores')


def decision_plot(counts, path):
    plt = plot_module()
    groups = ['both_correct', 'node_only_correct', 'full_only_correct', 'both_wrong']
    figure, ax = plt.subplots(figsize=(9, 5))
    for index, population in enumerate(('error', 'normal')):
        selected = counts[(counts.population == population) & (counts.region == 'all')].set_index('group').loc[groups]
        bars = ax.bar(np.arange(4)+(index-.5)*.36, selected.fraction, width=.36, label=population)
        for bar, row in zip(bars, selected.itertuples()):
            ax.annotate(f'{row.tokens}/{row.denominator}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(range(4), ['Both correct', 'Node only correct', 'Full only correct', 'Both wrong'])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Fraction within error / normal population')
    ax.set_title('Two separately trained models: decision agreement, NOT causal graph gains')
    ax.legend()
    finish(figure, path)


def matched_score_plot(matched, path):
    plt = plot_module()
    regions = ['early_third', 'middle_third', 'late_third']
    figure, ax = plt.subplots(figsize=(8, 5))
    for model, frame in matched.groupby('model', sort=False):
        group = frame.set_index('region').loc[regions]
        ax.plot(range(3), group.error_score, marker='o', label=model+': error')
        ax.plot(range(3), group.normal_score, marker='s', linestyle='--', label=model+': matched normal')
    ax.set_xticks(range(3), ['Early third', 'Middle third', 'Late third'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Token-weighted mean probability (not a span alarm)')
    ax.set_title('Fixed equal-length pairs; each model has its own score scale')
    ax.legend()
    finish(figure, path)


def node_heatmap(frame, field, title, path, centered=True):
    plt = plot_module()
    matrix = frame.pivot(index='llm_layer', columns='llm_head', values=field)
    values = matrix.to_numpy()
    figure, ax = plt.subplots(figsize=(8, 5.5))
    scale = max(float(np.nanmax(abs(values))), 1e-9) if centered else 1.
    image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', vmin=-scale if centered else 0, vmax=scale)
    ax.set_xlabel('Original LLM head index')
    ax.set_ylabel('Original LLM layer index')
    ax.set_title(title)
    figure.colorbar(image, ax=ax, label=field)
    finish(figure, path)


def node_plots(stats, effects, output):
    output = Path(output)
    for region in ('first', 'front_half', 'back_half'):
        selected = stats[(stats.region == region) & (stats.cohort == 'all')]
        if len(selected):
            node_heatmap(selected, 'source_delta', region+': paired self-attention difference (not importance)',
                         output/(region+'_self_attention_difference'))
    back = stats[stats.region == 'back_half']
    for cohort in ('node_hit', 'node_miss'):
        selected = back[back.cohort == cohort]
        if len(selected):
            node_heatmap(selected, 'source_delta', 'Back half: '+cohort+' minus paired normal (selection-based)',
                         output/(cohort+'_self_attention_difference'))
    effect_plot(effects, output/'node_channel_exchange_effects')


def effect_plot(frame, path):
    plt = plot_module()
    selected = frame[(frame.region == 'all') & (frame.llm_layer >= 0)]
    if selected.empty:
        return
    figure, ax = plt.subplots(figsize=(10, 4.8))
    for operation, group in selected.groupby('operation', sort=False):
        labels = [f'L{r.llm_layer}' + (f'H{r.llm_head}' if r.llm_head >= 0 else '') for r in group.itertuples()]
        ax.plot(labels, group['mean'], marker='.', label=operation)
        ax.fill_between(labels, group.low, group.high, alpha=.2)
    ax.axhline(0, linestyle='--', linewidth=.8)
    ax.set_ylabel('Source-balanced paired AUROC change')
    ax.set_title('Frozen NODE-ONLY model: changed input minus original; no graph messages')
    ax.tick_params(axis='x', rotation=65, labelsize=8)
    ax.legend()
    finish(figure, path)
