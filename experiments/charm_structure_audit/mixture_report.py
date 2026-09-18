"""Labelled diagnostics AFTER unlabelled freezing; no test-directed fitting."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from scipy.stats import spearmanr, kstest
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score

from .data import read_json, write_json, save_scores
from .evaluate import source_interval
from .mixture_math import transform_values, estimate_components, raw_parameters


def labels_for(record, prepared):
    path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
    with np.load(path, allow_pickle=False) as saved:
        return saved['gold'].astype(bool)


def reference_directions(args, parts, transform, output):
    """Supervised LDA is a SEPARATE diagnostic; it cannot change frozen scores."""
    from .mixture import read_nodes
    from .model import load_checkpoint

    blocks = [read_nodes(row, args.prepared)[0] for row in parts['fit']]
    values = np.concatenate(blocks)
    labels = np.concatenate([labels_for(row, args.prepared) for row in parts['fit']])
    coordinates = transform_values(values, transform)
    lda_state = estimate_components(coordinates, labels)
    lda = raw_parameters(lda_state, transform)
    save_scores(output / 'supervised_lda_diagnostic.npz', **lda)
    model, _ = load_checkpoint(Path(args.root) / 'node_only/checkpoint.pt')
    weight = model.in_proj.weight.detach().cpu().numpy().astype(float)
    _, _, right = np.linalg.svd(weight, full_matrices=False)
    axis = right[0]
    axis *= np.sign(axis[np.argmax(abs(axis))])
    save_scores(output / 'supervised_P_axis.npz', axis=axis)
    return lda, axis


def ranking(labels, score):
    positive = int(labels.sum())
    negative = len(labels) - positive
    return dict(tokens=len(labels), positives=positive, negatives=negative,
        auroc=float(roc_auc_score(labels, score)) if positive and negative else None,
        ap=float(average_precision_score(labels, score)) if positive else None)


def rank_results(table):
    """Both component signs are disclosed. Never choose the better test sign."""
    scores = ['gaussian1_nll', 'mixture_nll', 'component1_odds', 'component0_odds',
              'supervised_lda', 'supervised_P_axis']
    rows, within = [], []
    for name in scores:
        for role in ('all', 'span_onset', 'continuation'):
            mask = np.ones(len(table), bool) if role == 'all' else (table.gold == 0) | (table.role == role)
            group = table[mask]
            rows.append(dict(score=name, role=role, **ranking(group.gold.to_numpy(bool), group[name])))
        for identity, group in table.groupby('id'):
            within.append(dict(score=name, id=identity, source_id=group.source_id.iloc[0],
                               **ranking(group.gold.to_numpy(bool), group[name])))
    return pd.DataFrame(rows), pd.DataFrame(within)


def score_table(args, parts, output, seeds, lda, axis):
    from .mixture import read_nodes

    rows = []
    for record in parts['test']:
        values, _ = read_nodes(record, args.prepared)
        gold = labels_for(record, args.prepared)
        graph_path = Path(args.prepared) / 'graphs/test' / (str(record['id']) + '.npz')
        with np.load(graph_path, allow_pickle=False) as graph:
            starts = np.zeros(len(gold), bool)
            from .matching import surface_class
            response, offsets = str(graph['response']), graph['offsets']
            token_text = [response[a:b] for a, b in offsets]
            for start, _ in graph['spans']:
                starts[int(start)] = True
        base = dict(id=str(record['id']), source_id=str(record['source_id']), token=np.arange(len(gold)),
                    position=(np.arange(len(gold)) + .5) / len(gold), gold=gold.astype(int),
                    text=token_text, surface=[surface_class(text) for text in token_text],
                    role=np.where(gold, np.where(starts, 'span_onset', 'continuation'), 'normal'))
        with np.load(output / 'frozen_scores/test' / (str(record['id']) + '.npz')) as saved:
            for seed in seeds:
                table = pd.DataFrame(base)
                table['seed'] = seed
                table['gaussian1_nll'] = -saved['gaussian1_log_density']
                table['mixture_nll'] = -saved[f'mixture_log_density_{seed}']
                table['component1_odds'] = saved[f'component_log_odds_{seed}']
                table['component0_odds'] = -table.component1_odds
                table['supervised_lda'] = values @ lda['coefficient'] + lda['intercept']
                table['supervised_P_axis'] = values @ axis
                table['component'] = (table.component1_odds > 0).astype(int)
                rows.append(table)
    return pd.concat(rows, ignore_index=True)


def compare_directions(states, transform, lda, axis, tables):
    records, stability = [], []
    vectors = {seed: raw_parameters(state, transform)['coefficient'] for seed, state in states.items()}
    def cosine(left, right):
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(left @ right / denominator) if denominator else None
    for seed, vector in vectors.items():
        group = tables[tables.seed == seed]
        records.append(dict(seed=seed, cosine_raw_P=cosine(vector, axis),
            cosine_standardized_P=cosine(vector * transform['scale'], axis * transform['scale']),
            cosine_raw_labelled_LDA=cosine(vector, lda['coefficient']),
            spearman_P=float(spearmanr(group.component1_odds, group.supervised_P_axis).statistic),
            spearman_labelled_LDA=float(spearmanr(group.component1_odds, group.supervised_lda).statistic)))
    seed_list = list(states)
    for i, left in enumerate(seed_list):
        for right in seed_list[i + 1:]:
            a, b = tables[tables.seed == left], tables[tables.seed == right]
            stability.append(dict(seed_a=left, seed_b=right, cosine_raw=cosine(vectors[left], vectors[right]),
                score_spearman=float(spearmanr(a.component1_odds, b.component1_odds).statistic),
                component_agreement_ARI=float(adjusted_rand_score(a.component, b.component))))
    return pd.DataFrame(records), pd.DataFrame(stability)


def projection_fit_checks(args, records, transform, state):
    """Held-out mixture CDF along fixed projections; no IID p-values claimed."""
    from .mixture import read_nodes

    width = len(state['difference'])
    directions = np.random.default_rng(731).normal(size=(12, width))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    length = np.linalg.norm(state['coefficient'])
    main_direction = state['coefficient'] / length if length else np.eye(width)[0]
    directions = np.vstack((main_direction, directions))
    projected_difference = directions @ state['difference']
    weight = float(state['weight'])
    variance = 1 - weight * (1 - weight) * projected_difference ** 2
    cdf_blocks, histogram = [], []
    for record in records:
        values, _ = read_nodes(record, args.prepared)
        projected = transform_values(values, transform) @ directions.T
        left = ndtr((projected + weight * projected_difference) / np.sqrt(variance))
        right = ndtr((projected - (1 - weight) * projected_difference) / np.sqrt(variance))
        cdf = (1 - weight) * left + weight * right
        cdf_blocks.append(cdf)
        for column in range(len(directions)):
            frequencies = np.histogram(cdf[:, column], bins=np.linspace(0, 1, 11))[0] / len(cdf)
            for bin_index, fraction in enumerate(frequencies):
                histogram.append(dict(source_id=str(record['source_id']), id=str(record['id']),
                    projection=column, bin=bin_index, fraction=fraction))
    all_cdf = np.concatenate(cdf_blocks)
    checks = [dict(projection=column, kind='mixture_direction' if column == 0 else 'fixed_random',
                   tokens=len(all_cdf), cdf_uniform_gap=float(kstest(all_cdf[:, column], 'uniform').statistic))
              for column in range(len(directions))]
    return pd.DataFrame(checks), pd.DataFrame(histogram)


def density_summary(output, bootstrap):
    source = pd.read_csv(output / 'density_by_answer.csv', dtype={'source_id': str})
    rows = []
    for (split, seed), group in source.groupby(['split', 'seed']):
        interval = source_interval(group, 'density_gain', bootstrap)
        rows.append(dict(split=split, seed=seed, answers=len(group), **interval))
    return pd.DataFrame(rows)


def write_report(output, frozen, metrics, density, direction, model_notes):
    best = frozen['seed_by_unlabelled_select_density']
    lines = ['# 两种高斯状态假设审计', '',
        f'无标签 SELECT 密度选择 seed={best}。全部种子仍保留；未用真假、P方向或AUROC选模型。', '',
        '## 三项检查', '1. density_summary.csv：两成分在留出来源上是否优于单高斯。',
        '2. direction_comparison.csv / seed_stability.csv：方向是否稳定，是否与监督P轴及有标签LDA接近。',
        '3. test_metrics.csv / component_counts.csv：发现的成分是否与真假有关，而不只对应位置/词面。', '',
        'P是128×1024矩阵；比较的是它的1024维第一右奇异向量与GMM的Sigma^{-1}(mu1-mu0)。',
        '共享协方差高斯 => 线性后验；线性可分 => 高斯不成立。不同系数还要结合输入缩放、分数相关性。', '',
        '**component0/1只是编号。两方向AUROC原样输出，不挑高的一侧，不宣称已获得无标签真假定向。**',
        'supervised_lda使用FIT标签；supervised_P_axis来自监督checkpoint。两者仅作诊断，不是无监督结果。',
        'density分数把低密度当异常；这项先验也可能与真假无关。未设置或调整测试报警阈值。', '',
        '## 密度假设边界',
        '输入有界于[0,1]；非退化高斯在实数空间无界，只能作近似，不可能完全等于attention分布。',
        'cdf_checks.csv是固定投影的留出拟合缺口，无token独立p值；较小缺口不证明联合高斯正确。',
        '所有EM收敛、先验占比和界外边际概率见model_diagnostics.csv。失败/未收敛的解不能当作证据。',
        '只有多起点，没有重抽FIT来源的稳定性实验；参数区间不表示训练方差。', '',
        '当前是节点假设检验，保留原GNN及全部旧实验。不把混合拟合当作已验证的新图方法。']
    (output / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Selected without labels:', best, flush=True)
    print(metrics[(metrics.seed == best) & (metrics.role == 'all')].to_string(index=False), flush=True)


def draw_plots(output, metrics, density, directions, best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    selected = density[density.split == 'test']
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(selected.seed.astype(str), selected['mean'])
    axis.axhline(0, linestyle='--')
    axis.set(xlabel='Initialization seed', ylabel='Two minus one Gaussian log density',
             title='Unlabelled TEST density; higher does not imply truth separation')
    figure.tight_layout()
    figure.savefig(folder / 'heldout_density.png', dpi=160)
    plt.close(figure)
    rows = metrics[(metrics.seed == best) & (metrics.role == 'all')]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(rows.score, rows.auroc)
    axis.axhline(.5, linestyle='--')
    axis.set(ylim=(0, 1), ylabel='Full-token AUROC', title='Both component signs; LDA and P are supervised diagnostics')
    axis.tick_params(axis='x', rotation=22)
    figure.tight_layout()
    figure.savefig(folder / 'separation.png', dpi=160)
    plt.close(figure)


def diagnose(args, output, parts, transform, states, frozen):
    """This is the first point at which gold and supervised weights are opened."""
    lda, axis = reference_directions(args, parts, transform, output)
    table = score_table(args, parts, output, list(states), lda, axis)
    metrics, within = [], []
    for seed, group in table.groupby('seed', sort=False):
        measured, answer = rank_results(group)
        measured['seed'], answer['seed'] = seed, seed
        metrics.append(measured)
        within.append(answer)
    metrics = pd.concat(metrics, ignore_index=True)
    within = pd.concat(within, ignore_index=True)
    within_summary = []
    for (seed, score), group in within.groupby(['seed', 'score']):
        valid = group.dropna(subset=['auroc'])
        within_summary.append(dict(seed=seed, score=score, mixed_answers=len(valid),
            **source_interval(valid, 'auroc', args.bootstrap)))
    density = density_summary(output, args.bootstrap)
    directions, stability = compare_directions(states, transform, lda, axis, table)
    best = frozen['seed_by_unlabelled_select_density']
    checks, histograms = projection_fit_checks(args, parts['test'], transform, states[best])
    model_notes = []
    for seed, state in states.items():
        raw = raw_parameters(state, transform)
        sigma = np.sqrt(np.diag(raw['covariance']))
        outside = ndtr(-raw['means'] / sigma) + ndtr((raw['means'] - 1) / sigma)
        outside = raw['priors'] @ outside
        model_notes.append(dict(seed=seed, converged=bool(state['converged']), iterations=int(state['iterations']),
            component1_prior=float(state['weight']), smaller_prior=float(min(state['weight'], 1 - state['weight'])),
            separation_mahalanobis=float(state['difference'] @ state['difference'] / state['denominator']),
            mean_marginal_mass_outside_0_1=float(outside.mean())))
    counts = table.groupby(['seed', 'component']).agg(tokens=('gold', 'size'), errors=('gold', 'sum'),
        mean_position=('position', 'mean')).reset_index()
    counts['error_fraction'] = counts.errors / counts.tokens
    outputs = dict(test_metrics=metrics, within_answer=within, within_summary=pd.DataFrame(within_summary), density_summary=density,
        direction_comparison=directions, seed_stability=stability, cdf_checks=checks,
        cdf_histograms=histograms, component_counts=counts, model_diagnostics=pd.DataFrame(model_notes))
    for name, frame in outputs.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    coefficients = pd.DataFrame(dict(channel=np.arange(len(axis)),
        layer=np.arange(len(axis)) // frozen['heads'], head=np.arange(len(axis)) % frozen['heads'],
        supervised_P=axis, supervised_LDA=lda['coefficient']))
    for seed, state in states.items():
        coefficients[f'mixture_{seed}'] = raw_parameters(state, transform)['coefficient']
    coefficients.to_csv(output / 'direction_coefficients.csv', index=False)
    composition = table.groupby(['seed', 'component', 'surface']).agg(
        tokens=('gold', 'size'), errors=('gold', 'sum')).reset_index()
    composition.to_csv(output / 'component_surface.csv', index=False)
    table.to_csv(output / 'test_tokens.csv.gz', index=False)
    write_report(output, frozen, metrics, density, directions, model_notes)
    draw_plots(output, metrics, density, directions, best)
    with tarfile.open(output / 'mixture_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.png'):
                archive.add(path, arcname=path.relative_to(output))
        archive.add(output / 'test_tokens.csv.gz', arcname='test_tokens.csv.gz')
