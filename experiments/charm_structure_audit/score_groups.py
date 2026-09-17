"""High/low scores are not truth labels. Explain node values before graph routes.

No training, prediction, edge loading, label-based score threshold or head selection.
All score tails are descriptive, defined inside each answer before using labels.
"""

from pathlib import Path
import html
import json
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .data import read_json, read_tables, write_json
from .positions import annotate


def surface_type(text):
    text = text.strip()
    if not text:
        return 'whitespace'
    if any(char.isdigit() for char in text):
        return 'contains_digit'
    if any(char.isalpha() for char in text):
        return 'word'
    return 'punctuation'


def score_tails(table, fraction=.2):
    """Use the same score cutoffs for both labels. Ties are never broken by gold."""
    if not 0 < fraction < .5:
        raise ValueError('tail-fraction must lie between 0 and 0.5')
    table = table.copy().reset_index(drop=True)
    grouped = table.groupby('id', sort=False).score
    table['low_cut'] = grouped.transform(lambda x: x.quantile(fraction))
    table['high_cut'] = grouped.transform(lambda x: x.quantile(1-fraction))
    table['score_tail'] = 'middle'
    distinct = table.low_cut < table.high_cut
    table.loc[distinct & (table.score <= table.low_cut), 'score_tail'] = 'low'
    table.loc[distinct & (table.score >= table.high_cut), 'score_tail'] = 'high'
    table.loc[~distinct, 'score_tail'] = 'tied'
    table['answer_rank'] = grouped.rank(method='average', pct=True)
    table['answer_score_bin'] = np.minimum(9, (table.answer_rank*10).astype(int))
    return table


def prepare_table(directory, fraction):
    table, spans = read_tables(directory/'test')
    threshold = float(read_json(directory/'threshold.json')['value'])
    if not np.array_equal(table.predicted, (table.score > threshold).astype(int)):
        raise ValueError('Saved alarms do not match this model\'s threshold')
    unused = [key for key in table if key.startswith('structure_') and key != 'structure_self_attention_mean']
    table = annotate(table.drop(columns=unused), spans)
    table = score_tails(table, fraction)
    table['surface'] = table.text.map(surface_type)
    table['label_name'] = np.where(table.gold == 1, 'error', 'normal')
    table['answer_position'] = (table.token+.5)/table.groupby('id').token.transform('size')
    table['answer_third'] = np.minimum(2, (table.answer_position*3).astype(int))
    # A normal run is only a contiguous unlabelled interval, not a correct claim.
    table['interval_id'] = table.groupby('id', sort=False).gold.transform(lambda x: x.ne(x.shift()).cumsum())
    table.loc[table.gold == 1, 'interval_id'] = -(table.loc[table.gold == 1, 'span_start']+1)
    groups = table.groupby(['id', 'interval_id'], sort=False)
    table['interval_offset'] = groups.cumcount()
    table['interval_third'] = np.minimum(2, ((table.interval_offset+.5)*3/groups.token.transform('size')).astype(int))
    return table, threshold


def summarize_groups(table):
    rows = []
    for (label, tail), group in table.groupby(['label_name', 'score_tail']):
        rows.append(dict(label=label, tail=tail, tokens=len(group), answers=group.id.nunique(),
            sources=group.source_id.nunique(), mean_score=float(group.score.mean()),
            alarms=int(group.predicted.sum()), mean_answer_position=float(group.answer_position.mean()),
            mean_self_attention=float(group.structure_self_attention_mean.mean())
            if 'structure_self_attention_mean' in group else None))
    return pd.DataFrame(rows)


def score_bins(table):
    return table.groupby('answer_score_bin').agg(tokens=('score', 'size'),
        errors=('gold', 'sum'), mean_score=('score', 'mean'),
        mean_answer_position=('answer_position', 'mean')).assign(
        error_fraction=lambda x: x.errors/x.tokens).reset_index()


def composition(table):
    rows = []
    for axis in ('surface', 'answer_third', 'interval_third'):
        counted = table.groupby(['label_name', 'score_tail', axis]).size().reset_index(name='tokens')
        counted['fraction_in_group'] = counted.tokens/counted.groupby(['label_name', 'score_tail']).tokens.transform('sum')
        counted = counted.rename(columns={axis: 'position_or_type'})
        counted['axis'] = axis
        rows.append(counted)
    return pd.concat(rows, ignore_index=True)


def fixed_pair_tokens(table, pairs, tier):
    """Same original paired windows and same relative offset; never pick low normals."""
    indexed = table.set_index(['id', 'token'])
    columns = ['id', 'source_id', 'error_start', 'normal_start', 'offset', 'length', 'half',
               'error_token', 'normal_token', 'error_tail', 'normal_tail', 'error_score',
               'normal_score', 'error_alarm', 'normal_alarm', 'error_text', 'normal_text', 'score_margin']
    rows = []
    for pair in pairs:
        if pair['tier'] != tier:
            continue
        identity = str(pair['id'])
        for offset in range(int(pair['length'])):
            a, b = int(pair['error_start'])+offset, int(pair['normal_start'])+offset
            error, normal = indexed.loc[(identity, a)], indexed.loc[(identity, b)]
            if error.gold != 1 or normal.gold != 0 or str(error.source_id) != str(pair['source_id']):
                raise ValueError('Fixed pair does not match the score table')
            rows.append([identity, str(pair['source_id']), pair['error_start'], pair['normal_start'],
                offset, pair['length'], 'back' if (offset+.5)/pair['length'] >= .5 else 'front',
                a, b, error.score_tail, normal.score_tail, error.score, normal.score,
                error.predicted, normal.predicted, error.text, normal.text, error.score-normal.score])
    result = pd.DataFrame(rows, columns=columns)
    for side in ('error_token', 'normal_token'):
        if result.duplicated(['id', side]).any():
            raise ValueError('Fixed tier reuses tokens; do not duplicate its observations')
    return result


def matched_groups(paired):
    rows = []
    for (half, tail), group in paired.groupby(['half', 'error_tail']):
        wins = (group.error_score > group.normal_score) + .5*(group.error_score == group.normal_score)
        rows.append(dict(half=half, error_tail=tail, tokens=len(group),
            pairs=len(group[['id', 'error_start']].drop_duplicates()), sources=group.source_id.nunique(),
            error_mean=float(group.error_score.mean()), normal_mean=float(group.normal_score.mean()),
            mean_margin=float(group.score_margin.mean()), same_offset_win=float(wins.mean()),
            error_hits=int(group.error_alarm.sum()), normal_fp=int(group.normal_alarm.sum())))
    return pd.DataFrame(rows)


def read_node_values(prepared, table):
    """Only x and token alignment metadata; never materialize edge_attr or embeddings."""
    arrays, shape = None, None
    for identity, group in tqdm(table.groupby('id', sort=False), desc='read original node x', unit='answer'):
        path = Path(prepared)/'graphs/test'/(str(identity)+'.npz')
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved['record_json']))
            current = int(saved['layers']), int(saved['heads'])
            values = saved['x'][int(saved['prompt_length']):]
            labels, offsets, response = saved['gold'], saved['offsets'], str(saved['response'])
        if str(record['id']) != identity or set(group.source_id) != {str(record['source_id'])}:
            raise ValueError('Node identity differs from score table')
        if not np.array_equal(group.token, np.arange(len(labels))) or not np.array_equal(group.gold, labels):
            raise ValueError('Original node positions/labels differ from score table')
        if group.text.tolist() != [response[a:b] for a, b in offsets]:
            raise ValueError('Original node text differs from score table')
        if arrays is None:
            shape = current
            arrays = np.empty((len(table), current[0]*current[1]), np.float32)
        if current != shape or values.shape != (len(group), shape[0]*shape[1]) or not np.isfinite(values).all():
            raise ValueError('Node channel geometry or values differ')
        arrays[group.index] = values
    return arrays, shape


def matched_channel_contrasts(values, table, paired):
    """Error-minus-normal contrasts. Score-conditioned subsets are descriptive only."""
    index = pd.Series(table.index, index=pd.MultiIndex.from_frame(table[['id', 'token']]))
    rows = []
    for (identity, start), group in paired.groupby(['id', 'error_start']):
        for region, selected in (('all', group), ('back', group[group.half == 'back'])):
            cohorts = [('all', selected)]
            cohorts += [('error_'+tail, selected[selected.error_tail == tail]) for tail in ('high', 'low')]
            for name, subset in cohorts:
                if subset.empty:
                    continue
                left = index.loc[list(zip(subset.id, subset.error_token))].to_numpy()
                right = index.loc[list(zip(subset.id, subset.normal_token))].to_numpy()
                rows.append(dict(source=str(subset.source_id.iloc[0]), unit=f'{identity}:{start}',
                    contrast='matched_'+region+'_'+name, observations=len(subset),
                    delta=(values[left]-values[right]).mean(axis=0)))
    return rows


def within_interval_contrasts(values, table):
    """High-minus-low inside SAME interval, third and surface class, separately by gold."""
    rows = []
    keys = ['id', 'source_id', 'interval_id', 'interval_third', 'surface', 'label_name']
    for coordinates, group in table.groupby(keys, sort=False):
        high = group.index[group.score_tail == 'high']
        low = group.index[group.score_tail == 'low']
        if not len(high) or not len(low):
            continue
        identity, source, interval, third, surface, label = coordinates
        rows.append(dict(source=source, unit=f'{identity}:{interval}', contrast=label+'_high_minus_low',
            observations=len(high)+len(low), delta=values[high].mean(axis=0)-values[low].mean(axis=0)))
    return rows


def summarize_channels(rows, shape, bootstrap):
    """Cells -> equal interval/pair means -> equal source means. No head ranking."""
    output = []
    metadata = pd.DataFrame([{k: v for k, v in row.items() if k != 'delta'} for row in rows])
    for name, indices in metadata.groupby('contrast').indices.items():
        subset = metadata.iloc[indices].reset_index(drop=True)
        values = np.stack([rows[i]['delta'] for i in indices])
        units, sources = [], []
        for (source, _), chosen in subset.groupby(['source', 'unit']).indices.items():
            units.append(values[chosen].mean(axis=0))
            sources.append(source)
        units, sources = np.asarray(units), np.asarray(sources)
        estimates = np.stack([units[sources == s].mean(axis=0) for s in np.unique(sources)])
        low, high = np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
        if len(estimates) > 1 and bootstrap:
            rng = np.random.default_rng(42)
            draws = [estimates[rng.integers(len(estimates), size=len(estimates))].mean(axis=0)
                     for _ in range(bootstrap)]
            low, high = np.quantile(draws, [.025, .975], axis=0)
        for channel, mean in enumerate(estimates.mean(axis=0)):
            output.append(dict(contrast=name, llm_layer=channel//shape[1], llm_head=channel%shape[1],
                source_mean_delta=mean, low=low[channel], high=high[channel],
                sources=len(estimates), intervals=len(units), cells=len(subset),
                observations=int(subset.observations.sum()), interpretation='descriptive_not_importance'))
    return pd.DataFrame(output)


def example_page(table, paired, path):
    page = ['<!doctype html><meta charset="utf-8"><h1>High / low token scores</h1>',
        '<p>Within-answer tails; NOT hallucination labels or new alarm thresholds. '
        'Examples are selected by score for inspection, not performance evaluation. All tokens are in CSV.</p>']
    fields = ['id', 'token', 'text', 'gold', 'score', 'predicted', 'offset', 'span_length', 'surface']
    for (label, tail), group in table[table.score_tail.isin(['high', 'low'])].groupby(['label_name', 'score_tail']):
        example = group.sort_values(['score', 'id', 'token'], ascending=[tail == 'low', True, True]).head(20)
        page.append('<h2>'+html.escape(label+' / '+tail)+'</h2>'+example[fields].to_html(index=False, escape=True))
    page.append('<h2>ALL fixed paired tokens</h2>'+paired.to_html(index=False, escape=True))
    Path(path).write_text('\n'.join(page), encoding='utf-8')


def run_highlow(args, output, pairs):
    from .score_plots import highlow_plots

    completed = 0
    for name in args.models:
        directory = Path(args.root)/name
        if not (directory/'test/tokens.csv').exists():
            print('No completed score table:', name, flush=True)
            continue
        table, threshold = prepare_table(directory, args.tail_fraction)
        destination = Path(output)/name
        destination.mkdir(parents=True, exist_ok=True)
        paired = fixed_pair_tokens(table, pairs, args.pair_tier)
        groups, bins = summarize_groups(table), score_bins(table)
        table.to_csv(destination/'tokens.csv.gz', index=False)
        paired.to_csv(destination/'paired_tokens.csv.gz', index=False)
        for filename, frame in (('score_groups', groups), ('score_bins', bins),
                                ('composition', composition(table)), ('matched_groups', matched_groups(paired))):
            frame.to_csv(destination/(filename+'.csv'), index=False)
        scalar_auc = None
        if 'structure_self_attention_mean' in table and table.gold.nunique() == 2:
            scalar_auc = float(roc_auc_score(table.gold, table.structure_self_attention_mean))
        channels = pd.DataFrame()
        if args.node_values:
            values, shape = read_node_values(args.prepared, table)
            rows = matched_channel_contrasts(values, table, paired) + within_interval_contrasts(values, table)
            if rows:
                channels = summarize_channels(rows, shape, args.bootstrap)
                channels.to_csv(destination/'node_value_contrasts.csv.gz', index=False)
        if args.fixed_output:
            from .compare_fixed_scores import compare_fixed_scores
            compare_fixed_scores(args.fixed_output, args.prepared, table, destination, args.tail_fraction)
        example_page(table, paired, destination/'examples.html')
        highlow_plots(groups, bins, channels, destination/'figures')
        write_json(destination/'protocol.json', dict(model=name, original_threshold=threshold,
            tail_fraction=args.tail_fraction, pair_tier=args.pair_tier,
            matched_pairs=len(paired[['id', 'error_start']].drop_duplicates()),
            channel_contrasts_available=not channels.empty,
            original_auroc=float(roc_auc_score(table.gold, table.score)) if table.gold.nunique() == 2 else None,
            original_mean_self_auroc=scalar_auc, reads_node_values=args.node_values,
            selection='Same answer score quantiles, label-blind; not a new detector threshold.',
            contrasts='Raw x only. No edge/neighbor descriptors are attributed to node_only.',
            limitations=['Source intervals are exploratory, uncorrected for multiple comparisons.',
                'High/low is score-conditioned; a difference is not proof of a hallucination mechanism.',
                'Matched normal windows keep original offsets; no low-score normal selection.',
                'Only same-interval/third/surface cells containing BOTH high and low contribute to that contrast.',
                'A scalar mean discards channel patterns. Its AUROC is not the full-vector AUROC.']))
        print(name+'\n'+groups.to_string(index=False), flush=True)
        completed += 1
    if not completed:
        raise ValueError('No completed model tables found for highlow')
    archive = Path(output)/'highlow_bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(Path(output).rglob('*')):
            if path.is_file() and (path.suffix in ('.json', '.csv', '.png') or path.name.endswith('.csv.gz')):
                bundle.add(path, arcname=path.relative_to(output), recursive=False)
    print('Compact review bundle:', archive, flush=True)
