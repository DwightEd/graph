"""Descriptive score-group plots: denominators and no selected best heads."""

from pathlib import Path

import numpy as np


def save_figure(figure, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'), dpi=160)
    figure.savefig(path.with_suffix('.svg'))
    plt.close(figure)


def highlow_plots(groups, bins, channels, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    figure, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(bins.answer_score_bin+1, bins.error_fraction, marker='o')
    for row in bins.itertuples():
        ax.annotate(f'{int(row.errors)}/{int(row.tokens)}', (row.answer_score_bin+1, row.error_fraction),
                    xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xlabel('Within-answer score rank bin (low to high)')
    ax.set_ylabel('Fraction of tokens labelled as errors')
    ax.set_title('Observed error frequency, not calibrated probability')
    ax.set_ylim(0, min(1.15, max(.15, bins.error_fraction.max()*1.4)))
    save_figure(figure, output/'error_fraction_by_score_rank')

    selected = groups[groups['tail'].isin(['high', 'low'])]
    figure, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(selected))
    ax.bar(x, selected.tokens)
    ax.set_xticks(x, selected.label+' / '+selected['tail'], rotation=15)
    for position, row in enumerate(selected.itertuples()):
        ax.annotate(str(row.tokens), (position, row.tokens), ha='center', va='bottom')
    ax.set_ylabel('Tokens (both gold labels use the same answer-level quantiles)')
    ax.set_title('High/low score does not mean error/normal')
    ax.margins(y=.15)
    save_figure(figure, output/'four_score_groups')

    if channels.empty:
        return
    for contrast, frame in channels.groupby('contrast'):
        pivot = frame.pivot(index='llm_layer', columns='llm_head', values='source_mean_delta')
        scale = max(float(np.max(abs(pivot.to_numpy()))), 1e-8)
        figure, ax = plt.subplots(figsize=(8, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect='auto', vmin=-scale, vmax=scale)
        ax.set_xlabel('Original LLM head')
        ax.set_ylabel('Original LLM layer')
        ax.set_title(contrast+'\nRaw node-value contrast, NOT model importance')
        figure.colorbar(image, ax=ax, label='Equal-source mean difference')
        save_figure(figure, output/contrast)
