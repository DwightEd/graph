"""Readable node tokens, source tokens, bidirectional incidence and coverage."""

import html
import json

import numpy as np
import pandas as pd


def collect(output, filename):
    frames = []
    for path in sorted((output / 'samples').glob('*/metadata.json')):
        table_path = path.parent / filename
        if table_path.exists():
            metadata = json.loads(path.read_text())
            keys = ('id', 'source_id', 'task', 'generator', 'split', 'annotation_origin')
            frames.append(pd.read_csv(table_path).assign(**{key: metadata[key] for key in keys}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reverse_statistics(spans):
    rows = []
    keys = ['split', 'task', 'generator', 'threshold']
    for identity, group in spans.groupby(keys):
        measured = group.prior_state.ne(-1)
        paired = group[measured & group.normal_prior_state.ne(-1)]
        differences = paired.assign(difference=paired.prior_state-paired.normal_prior_state)
        sources = differences.groupby('source_id').difference.mean()
        interval = [np.nan, np.nan]
        if len(sources) >= 2:
            rng = np.random.default_rng(17)
            means = rng.choice(sources.to_numpy(), (1000, len(sources))).mean(axis=1)
            interval = np.quantile(means, [.025, .975])
        rows.append(dict(zip(keys, identity), spans=len(group), preceded=int(group.prior_state.eq(1).sum()),
            absent=int(group.prior_state.eq(0).sum()), unknown=int((~measured).sum()),
            at_decision=int(group.decision_state.eq(1).sum()), paired=len(paired),
            normal_preceded=int(paired.normal_prior_state.eq(1).sum()),
            error_preceded_in_pairs=int(paired.prior_state.eq(1).sum()),
            source_difference=sources.mean(), low=interval[0], high=interval[1]))
    return pd.DataFrame(rows)


def forward_statistics(positions):
    measured = positions[positions.state.ne(-1) & positions.full_followup & positions.current_gold.ge(0)]
    keys = ['split', 'task', 'generator', 'threshold', 'current_gold', 'state']
    result = measured.groupby(keys).agg(positions=('target', 'size'),
        future_spans=('future_span', 'sum'), sources=('source_id', 'nunique')).reset_index()
    result['future_span_fraction'] = result.future_spans/result.positions
    return result


def token_view(directory):
    saved = json.loads((directory / 'token_text.json').read_text())
    tokens = saved['tokens'][saved['prompt_length']:]
    nodes = pd.read_csv(directory / 'nodes.csv')
    selected = {int(row.node_token): row for row in nodes.itertuples()}
    gold = set()
    path = directory / 'spans.csv'
    if path.exists():
        for span in pd.read_csv(path).drop_duplicates(['onset', 'end']).itertuples():
            gold.update(range(span.onset, span.end))
    fragments = []
    for index, token in enumerate(tokens):
        color = '#fee2e2' if index in gold else 'transparent'
        border = 'border-bottom:3px solid #ca8a04;' if index in selected else ''
        title = f'node token={index}'
        if index in selected:
            title += f'; heads={selected[index].head_ids}; predicts target={index+1}'
        fragments.append(f'<span style="background:{color};{border}" title="{html.escape(title)}">{html.escape(token)}</span>')
    return '<p>黄色下划线：重锚候选；红底：已有完整金标的幻觉 token。未标注前缀不作正常判断。</p><pre>' + ''.join(fragments) + '</pre>'


def summarize(output, primary):
    collect(output, 'coverage.csv').to_csv(output / 'coverage.csv', index=False)
    nodes = collect(output, 'nodes.csv')
    nodes.to_csv(output / 'nodes.csv', index=False)
    links = collect(output, 'node_span_links.csv')
    links.to_csv(output / 'node_span_links.csv', index=False)
    positions, spans = collect(output, 'positions.csv.gz'), collect(output, 'spans.csv')
    result = dict(purpose='structural_reanchor_association_audit_not_causal_proof',
                  primary_shift=primary, nodes=len(nodes),
                  answers=len(list((output / 'samples').glob('*/metadata.json'))))
    if not positions.empty:
        positions.to_csv(output / 'positions.csv.gz', index=False)
        spans.to_csv(output / 'spans.csv', index=False)
        reverse_statistics(spans).to_csv(output / 'reverse_summary.csv', index=False)
        forward_statistics(positions).to_csv(output / 'forward_summary.csv', index=False)
        result.update(annotation_scope='official_ragtruth_complete_response',
                      spans=int(spans.threshold.eq(primary).sum()))
    else:
        claims = collect(output, 'claim_association.csv')
        claims.to_csv(output / 'claim_association.csv', index=False)
        result.update(annotation_scope='reviewed_local_claim_only_prefix_history_unreviewed',
                      reviewed_claims=int(claims.threshold.eq(primary).sum()))
    path = output / 'population_coverage.csv'
    if path.exists():
        population = pd.read_csv(path)
        result.update(official_answers=len(population), cached_answers=int(population.cached.sum()),
                      missing_answers=int((~population.cached).sum()))
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def write_review(output):
    fragments = ["<!doctype html><meta charset='utf-8'><title>Reanchor nodes</title>",
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}pre{white-space:pre-wrap}summary{cursor:pointer}</style>',
        '<h1>重锚结构候选：节点、读取来源与标注关系</h1>',
        '<p>节点是发生读取的 query token；target 是它预测的下一个 token。结构切换不等于正确证据被整合。未知历史不作正常标注。</p>']
    for path in sorted((output / 'samples').glob('*/metadata.json')):
        metadata = json.loads(path.read_text())
        fragments.append('<h2>' + html.escape(metadata['id']) + '</h2>')
        fragments.append('<p>' + html.escape(metadata['annotation_origin']) + '</p>')
        fragments.append(token_view(path.parent))
        for filename in ('claim_association.csv', 'spans.csv', 'nodes.csv'):
            file = path.parent / filename
            if file.exists():
                fragments.append('<h3>' + filename + '</h3>' + pd.read_csv(file).to_html(index=False, escape=True))
        sources = pd.read_csv(path.parent / 'source_reads.csv')
        for target, group in sources.groupby('target'):
            columns = ['layer', 'head', 'source', 'source_text', 'source_context', 'region',
                       'source_gain', 'source_gain_low', 'source_is_prior_event', 'same_token_as_node']
            fragments.append(f'<details><summary>target={target}：读取哪些 token</summary>'
                             + group[columns].to_html(index=False, escape=True) + '</details>')
    (output / 'review.html').write_text('\n'.join(fragments), encoding='utf-8')
