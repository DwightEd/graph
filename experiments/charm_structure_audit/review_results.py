"""Read completed audits, separate their claims, and bundle compact evidence."""

from pathlib import Path
import shutil
import tarfile

import pandas as pd

from .data import read_json, write_json
from .compare_models import run_compare


ARTIFACTS = ('span_counts.csv', 'paired_detection_counts.json', 'position_counts.csv',
    'position_counts_by_length.csv', 'span_sentence_grid.csv', 'sentences.csv',
    'layer_head_routes.csv.gz', 'depth_summary.csv', 'depth_contrasts.csv',
    'channel_effects.csv', 'role_changes.csv', 'changed_cells.csv', 'independent_minus_coupled.csv',
    'pair_effects.csv.gz', 'pair_scores.csv', 'pairs.json', 'models.csv', 'paired_comparison.csv', 'matched_positions.csv', 'graph_changes.csv',
    'node_channel_effects.csv', 'cohort_counts.csv', 'node_attribute_differences.csv.gz')


def audit_directories(args, output):
    root = Path(args.root)
    directories = [p for p in sorted(root.glob('audit_*')) if p.is_dir()]
    directories.extend(Path(p) for p in args.audit_inputs)
    selected = []
    for path in directories:
        config = path/'config.json'
        if path.resolve() == output.resolve():
            continue
        if config.exists() and read_json(config).get('mode') == 'review':
            continue
        selected.append(path.resolve())
    return sorted(set(selected))


def copy_completed(directory, destination):
    """Copy named result tables, not scores, graphs, model weights or old archives."""
    files = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink() or {'scores', 'pairs', 'samples', 'existing'} & set(path.relative_to(directory).parts[:-1]):
            continue
        if path.name not in ARTIFACTS and path.name not in ('protocol.json', 'config.json', 'units.json', 'replay.json'):
            continue
        target = destination/path.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files.append(target)
    return files


def route_overview(files, output):
    from .visualize import route_plots
    found = [path for path in files if path.name == 'layer_head_routes.csv.gz']
    for index, path in enumerate(found):
        frame = pd.read_csv(path)
        route_plots(frame, output/f'figures/routes_{index}')
        coverage = frame.groupby(['phase', 'metric']).agg(
            min_pairs=('pairs', 'min'), median_pairs=('pairs', 'median'), max_pairs=('pairs', 'max'),
            min_sources=('sources', 'min'), median_sources=('sources', 'median')).reset_index()
        coverage.to_csv(output/f'route_coverage_{index}.csv', index=False)


def channel_overview(files, output):
    from .visualize import head_plots
    for index, path in enumerate(p for p in files if p.name == 'channel_effects.csv'):
        frame = pd.read_csv(path)
        head_plots(frame, output/f'figures/full_charm_channels_{index}')


def markdown_table(frame, columns):
    selected = frame[columns]
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in selected.itertuples(index=False, name=None):
        values = [f'{v:.4f}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def write_review(output, inventory, files):
    models = pd.read_csv(output/'comparison/models.csv')
    sections = ['# 已完成审计：节点信号与完整模型分开解释', '',
        '本报告只读取已有结果；没有重跑任何神经网络。', '',
        '## 1. 节点模型本身能分多少？', markdown_table(models, ['model', 'auroc', 'ap', 'recall', 'fpr']), '',
        '同一词的两模型判对/判错见 comparison/decision_counts.csv 和 comparison/comparison_tokens.csv.gz。',
        '原校准阈值不同；full-only-correct 表示模型之间的行为差异，不等于消息传递的因果贡献。', '',
        '## 2. 相同正常/错误窗口',
        'comparison/paired_comparison.csv 只比较相同pair和相同区域；mean为source均衡差，普通配对差另列。', '',
        '## 3. 已有实验索引']
    counts = pd.read_csv(output/'comparison/decision_counts.csv')
    counts = counts[(counts.population == 'error') & (counts.region == 'all')]
    sections.insert(8, markdown_table(counts, ['group', 'tokens', 'denominator', 'fraction']))
    sections.append(markdown_table(pd.DataFrame(inventory), ['directory', 'tables', 'config_present']))
    sections.append('文件存在不自动保证样本全部完成；先核对相应protocol/config和有效配对数。每个来源目录独立保留，不混合不同运行。')
    for name, title in [('span_counts.csv', '位置与检出'), ('depth_summary.csv', '路由深度概括')]:
        for path in [p for p in files if p.name == name]:
            sections += ['', '### '+title+'：'+str(path.relative_to(output)),
                         markdown_table(pd.read_csv(path), list(pd.read_csv(path, nrows=0).columns))]
    found = {path.name for path in files}
    sections += ['', '## 4. 哪些问题仍缺证据？']
    for name, description in [('layer_head_routes.csv.gz', '逐层逐头路由关联'),
                              ('channel_effects.csv', '完整CHARM通道敏感性'),
                              ('node_channel_effects.csv', '节点模型自身的通道依赖')]:
        sections.append(f'- {description}：'+('已找到文件。' if name in found else '未找到，不补零、不推断。'))
    sections += ['', '## 5. 不能混用的结论',
        'routes是固定配对的阶段平均，包含TP和FN；不能称为“只有检出词”的逐头机制。',
        '旧heads固定分析完整charm_in，不是node_only。图里发现的关联不能替代节点模型的解释。',
        'prompt_mass下降只是保留attention质量变化，不自动等于正确证据丢失。',
        'head通道清零可能分布外；独立交换比共同交换更伤也可能来自数值非线性组合。',
        '两组图同时符合假设仍不能证明原LLM因果链。需匹配样本、有效分母、source区间和独立复验。']
    (output/'REPORT_zh.md').write_text('\n\n'.join(sections), encoding='utf-8')


def compact_bundle(output):
    archive = output/'review_bundle.tar.gz'
    files = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.png')]
    files += [p for p in output.rglob('*.csv.gz')]
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(set(files)):
            tar.add(path, arcname='review/'+path.relative_to(output).as_posix(), recursive=False)
    return archive


def run_review(args, output, pairs):
    comparison = output/'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    run_compare(args, comparison, pairs)
    # Refresh only our generated copies; never alter the input experiment directories.
    if (output/'existing').exists():
        shutil.rmtree(output/'existing')
    inventory, files = [], []
    for index, directory in enumerate(audit_directories(args, output)):
        target = output/'existing'/f'{index:02d}_{directory.name}'
        saved = copy_completed(directory, target)
        inventory.append(dict(directory=str(directory), tables=len(saved),
                              config_present=(directory/'config.json').exists()))
        files.extend(saved)
    if not inventory:
        inventory.append(dict(directory='no completed audit directories', tables=0, config_present=False))
    route_overview(files, output)
    channel_overview(files, output)
    write_json(output/'inventory.json', inventory)
    write_review(output, inventory, files)
    archive = compact_bundle(output)
    print('报告：', output/'REPORT_zh.md')
    print(f'小结果包：{archive} ({archive.stat().st_size/1024**2:.2f} MiB)')
