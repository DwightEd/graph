"""逐答保存实验单位和配对结果，再汇总；不输出虚构的检测分数。"""

from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np

from .measurements import METRICS
from .statistics import source_bootstrap, source_effects


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def pair_description(answer, pair):
    row = asdict(pair)
    row['entropy_gap'] = finite_number(pair.entropy_gap)
    for name, span in [('error', pair.error), ('control', pair.control)]:
        start = int(answer.offsets[span.start, 0])
        end = int(answer.offsets[span.end - 1, 1])
        row[name]['text'] = answer.text[start:end]
    return row


def save_answer(output, answer, pairs, channels, values, counts):
    directory = Path(output) / 'samples'
    metadata = dict(
        response_id=answer.response_id, source_id=answer.source_id, task=answer.task,
        generator=answer.generator, split=answer.split, gold_spans=len(answer.spans),
        matched_pairs=len(pairs), entropy_available=bool(np.isfinite(answer.entropy).any()),
        pairs=[pair_description(answer, pair) for pair in pairs],
    )
    path = directory / f'{answer.response_id}.npz'
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, channels=channels, readings=values, counts=counts,
                        metric_names=METRICS, metadata=json.dumps(metadata, ensure_ascii=False))
    temporary.replace(path)
    return metadata


def load_saved(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        channels = archive['channels']
        values = archive['readings']
    return metadata, channels, values


def collect_groups(output):
    groups = {}
    inventory = []
    for path in sorted((Path(output) / 'samples').glob('*.npz')):
        metadata, channels, values = load_saved(path)
        inventory.append(metadata)
        if not metadata['matched_pairs']:
            continue
        key = (metadata['split'], metadata['task'], metadata['generator'])
        groups.setdefault(key, []).append((metadata, channels, values))
    return groups, inventory


def group_arrays(answers):
    """不同导出允许包含不同head；按物理身份对齐，不把缺head补成零。"""
    identities = sorted({tuple(channel) for _, channels, _ in answers for channel in channels})
    lookup = {identity: index for index, identity in enumerate(identities)}
    effects = []
    sources = []
    for metadata, channels, readings in answers:
        aligned = np.full((metadata['matched_pairs'], len(identities), len(METRICS)), np.nan)
        columns = [lookup[tuple(channel)] for channel in channels]
        difference = readings[:, :, 0] - readings[:, :, 1]
        aligned[:, columns] = difference.transpose(1, 0, 2)
        effects.append(aligned)
        sources.extend([metadata['source_id']] * metadata['matched_pairs'])
    return identities, sources, np.concatenate(effects)


def summarize_group(key, answers, bootstrap):
    identities, sources, effects = group_arrays(answers)
    clustered = source_effects(sources, effects)
    mean, lower, upper, count = source_bootstrap(clustered, bootstrap)
    rows = []
    for channel_index, (layer, head) in enumerate(identities):
        for metric_index, metric in enumerate(METRICS):
            coordinate = (channel_index, metric_index)
            rows.append(dict(
                split=key[0], task=key[1], generator=key[2], layer=layer, head=head,
                metric=metric, error_minus_control=finite_number(mean[coordinate]),
                ci_low=finite_number(lower[coordinate]), ci_high=finite_number(upper[coordinate]),
                sources=int(count[coordinate]),
                pairs=int(np.isfinite(effects[:, channel_index, metric_index]).sum()),
            ))
    return rows


def summarize(output, bootstrap=200):
    groups, inventory = collect_groups(output)
    rows = []
    for key, answers in sorted(groups.items()):
        rows.extend(summarize_group(key, answers, bootstrap))
    if rows:
        with (Path(output) / 'paired_effects.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        purpose='label-assisted mechanism discovery, not detector evaluation',
        answers=len(inventory), gold_spans=sum(row['gold_spans'] for row in inventory),
        matched_pairs=sum(row['matched_pairs'] for row in inventory),
        entropy_available_answers=sum(row['entropy_available'] for row in inventory),
        inference='paired source means; all head intervals are exploratory, uncorrected for multiplicity',
        evidence_applicability='not measured without independently reviewed source roles',
    )
    write_json(Path(output) / 'summary.json', summary)
    write_json(Path(output) / 'matched_pairs.json', inventory)
    return summary
