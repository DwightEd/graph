"""One figure per question, with explicit denominators; no score-selected gallery."""

import html
from pathlib import Path

import numpy as np


def finish(figure, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'), dpi=160)
    figure.savefig(path.with_suffix('.svg'))
    plt.close(figure)


def localization_plots(counts, positions, grid, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    for population in counts.population:
        group = positions[(positions.population == population) & (positions.axis == 'interval_third')]
        group = group.set_index('position').reindex(range(3))
        positive = population in ('gold_error', 'matched_error')
        field, rate = ('TP', 'recall') if positive else ('FP', 'fpr')
        for metric, title in ((field, 'Alarmed tokens / available tokens'), (rate, 'Token recall' if positive else 'Normal-token false alarm rate')):
            figure, ax = plt.subplots(figsize=(6.5, 4))
            ax.bar(range(3), group[metric])
            ax.set_xticks(range(3), ['Early third', 'Middle third', 'Late third'])
            ax.set_title(population + ': ' + title)
            for position, row in group.iterrows():
                if np.isfinite(row[metric]):
                    ax.annotate(f'{int(row[field])}/{int(row.tokens)}', (position, row[metric]),
                                xytext=(0, 4), textcoords='offset points', ha='center')
            if metric == rate:
                ax.set_ylim(0, 1.12)
            else:
                ax.margins(y=.2)
            finish(figure, directory / f'{population}_{metric}')
        subset = grid[grid.population == population]
        heatmap_counts(subset, directory / (population + '_span_by_sentence'), population)
    matched_score_plots(positions, directory)
    figure, ax = plt.subplots(figsize=(8, 4.5))
    rows = counts.set_index('population')
    names = list(rows.index)
    total, success = rows.spans.to_numpy(), rows.success_count.to_numpy()
    ax.bar(np.arange(len(names)) - .18, total, width=.36, label='Total intervals')
    ax.bar(np.arange(len(names)) + .18, success, width=.36, label='Error: any hit; normal: zero alarms')
    ax.set_xticks(range(len(names)), names, rotation=15)
    ax.set_ylabel('Number of intervals')
    ax.legend(fontsize=8)
    ax.set_title('Separate populations; any-hit is NOT full-span coverage')
    for x, count in enumerate(success):
        ax.annotate(str(count), (x + .18, count), ha='center', va='bottom')
    finish(figure, directory / 'span_counts')



def matched_score_plots(positions, directory):
    import matplotlib.pyplot as plt

    for axis in ('interval_third', 'interval_offset'):
        figure, ax = plt.subplots(figsize=(7.5, 4.5))
        plotted = False
        for name in ('matched_error', 'matched_normal'):
            group = positions[(positions.population == name) & (positions.axis == axis)]
            group = group[(group.position >= 0) & (group.position < 20)].sort_values('position')
            if len(group):
                plotted = True
                ax.plot(group.position, group.mean_score, marker='.', label=name)
        if not plotted:
            plt.close(figure)
            continue
        ax.set_xlabel('Relative third' if axis == 'interval_third' else 'Token offset from interval start (first 20 shown)')
        ax.set_ylabel('Mean score (token-weighted; not an alarm rate)')
        ax.set_ylim(0, 1)
        ax.set_title('Same matched windows: score difference by position')
        ax.legend()
        finish(figure, Path(directory) / ('matched_scores_' + axis))


def heatmap_counts(frame, path, title):
    import matplotlib.pyplot as plt

    rates = np.full((3, 3), np.nan)
    labels = {}
    for row in frame.itertuples():
        i, j = int(row.interval_third), int(row.sentence_third)
        rates[i, j] = row.alarm_rate
        labels[i, j] = f'{int(row.alarms)}/{int(row.tokens)}\n{row.alarm_rate:.1%}'
    figure, ax = plt.subplots(figsize=(6.2, 4.7))
    image = ax.imshow(np.ma.masked_invalid(rates), vmin=0, vmax=1)
    for i, j in np.ndindex((3, 3)):
        ax.text(j, i, labels.get((i, j), 'No tokens'), ha='center', va='center', fontsize=9,
                bbox=dict(facecolor=plt.rcParams['figure.facecolor'], edgecolor='none', alpha=.85, pad=2))
    ax.set_xticks(range(3), ['Early', 'Middle', 'Late'])
    ax.set_yticks(range(3), ['Early', 'Middle', 'Late'])
    ax.set_xlabel('Position within punctuation sentence')
    ax.set_ylabel('Position within diagnostic span')
    ax.set_title(title + ': alarms / tokens')
    figure.colorbar(image, ax=ax, label='Alarm rate')
    finish(figure, path)


def route_plots(frame, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selections = [('onset_minus_pre', 'prompt_mass'), ('early', 'noncopy_excess'),
                  ('late', 'noncopy_excess'), ('late', 'history_hhi'), ('late', 'shared_history_js')]
    for phase, metric in selections:
        group = frame[(frame.phase == phase) & (frame.metric == metric)]
        matrix = group.pivot(index='llm_layer', columns='llm_head', values='source_mean_delta')
        values = matrix.to_numpy()
        observed = np.isfinite(values)
        scale = max(float(np.max(abs(values[observed]))) if observed.any() else 0., 1e-8)
        figure, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', vmin=-scale, vmax=scale)
        ax.set_xlabel('Original LLM head index (not aligned between layers)')
        ax.set_ylabel('Original LLM layer index')
        ax.set_title(f'{phase}: {metric}\nPaired error minus normal, equal-source mean')
        figure.colorbar(image, ax=ax, label='Signed difference; missing = no observation')
        finish(figure, Path(directory) / (phase + '_' + metric))


def head_plots(frame, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for metric in ('auroc_delta', 'matched_auc_delta', 'matched_margin_delta'):
        figure, ax = plt.subplots(figsize=(9, 4.5))
        for (site, operation), group in frame.groupby(['site', 'operation'], sort=False):
            x = [f'L{row.llm_layer}' + (f'H{int(row.llm_head)}' if row.llm_head >= 0 else '') for row in group.itertuples()]
            ax.plot(x, group[metric], marker='.', label=site + '/' + operation)
        ax.axhline(0, linewidth=.8, linestyle='--')
        ax.set_ylabel('Masked minus original')
        ax.set_title(metric + ' (frozen detector sensitivity, not native LLM intervention)')
        ax.tick_params(axis='x', rotation=60)
        ax.legend()
        finish(figure, Path(directory) / metric)


def escaped_sentence(text, group, start, end):
    """Render original characters ONCE, even if subword offsets overlap."""
    bounds = {start, end}
    active = group[(group.char_end > start) & (group.char_start < end)]
    for row in active.itertuples():
        bounds.update((max(start, row.char_start), min(end, row.char_end)))
    bounds = sorted(bounds)
    fragments = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        selected = active[(active.char_start < right) & (active.char_end > left)]
        piece = html.escape(text[left:right])
        if not len(selected):
            fragments.append(piece)
            continue
        outcomes = sorted(set(selected.outcome))
        title = '; '.join(f'token={r.token} {r.outcome} score={r.score:.4f} span_offset={r.offset} sentence_offset={r.sentence_offset}'
                          for r in selected.itertuples())
        css = outcomes[0] if len(outcomes) == 1 else 'mixed'
        fragments.append(f'<span class="{css}" title="{html.escape(title, quote=True)}">{piece}</span>')
    return ''.join(fragments)


def token_gallery(samples, table, spans, sentences, path):
    page = ['<!doctype html><meta charset="utf-8"><title>CHARM localization</title>',
        '<style>body{font:16px sans-serif;max-width:1150px;margin:2em auto;line-height:1.8}p.text{white-space:pre-wrap}'
        '.TP{background:#d5ecd8}.FN{background:#f8dddd}.FP{background:#fff0c2}.TN{border-bottom:1px dotted #888}'
        '.mixed{outline:1px dashed}table{border-collapse:collapse}td,th{padding:4px 12px;border:1px solid #ccc}</style>',
        '<h1>错误 / 正常片段与句子位置</h1><p>TP=正确报警，FN=漏检，FP=误报，TN=正确不报警。悬停查看token、分数及位置。'
        '原文只显示一次；重复offset共同显示。按回答ID排序，不按成绩选例。句子由标点规则划分，不是语义事实边界。</p>']
    for sample in sorted(samples, key=lambda s: str(s['id'])):
        identity = str(sample['id'])
        group = table[table.id == identity]
        intervals = spans[spans.id == identity]
        columns = ['population', 'start', 'end', 'alarm_tokens', 'tokens', 'first_alarm_offset', 'crosses_sentences']
        page.append(f'<details><summary>{html.escape(identity)} | TP={int((group.outcome=="TP").sum())} '
                    f'FP={int((group.outcome=="FP").sum())}</summary>')
        page.append(intervals[columns].to_html(index=False, escape=True))
        for sentence in sentences[sentences.id == identity].itertuples():
            marked = escaped_sentence(str(sample['response']), group, sentence.char_start, sentence.char_end)
            page.append(f'<p>sentence {sentence.sentence}, chars [{sentence.char_start},{sentence.char_end})</p><p class="text">{marked}</p>')
        page.append('</details>')
    Path(path).write_text('\n'.join(page), encoding='utf-8')
