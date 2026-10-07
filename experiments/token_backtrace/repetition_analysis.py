"""Within-answer controls and source-level statistics for repeated head patterns."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def match_tokens(row, repeat_counts, categories=None):
    """Unique controls: same token/type, position, prior error status, repetition bin."""
    labels = row['labels'].astype(bool)
    identities = row['token_ids'] if categories is None else categories
    position = (np.arange(len(labels)) + .5) / len(labels)
    prior_error = np.r_[False, np.cumsum(labels[:-1]) > 0]
    repetition = np.minimum(repeat_counts[:, 0], 2)
    available = ~labels
    pairs = []
    for target in np.where(labels)[0]:
        eligible = available & (identities == identities[target])
        eligible &= prior_error == prior_error[target]
        eligible &= repetition == repetition[target]
        eligible &= abs(position - position[target]) <= .2
        controls = np.where(eligible)[0]
        if not len(controls):
            continue
        selected = controls[np.argmin(abs(controls - target))]
        available[selected] = False
        pairs.append((int(target), int(selected)))
    return pairs


def token_category(text):
    stripped = text.strip()
    if '\n' in text:
        return 'newline'
    if not stripped:
        return 'space'
    if stripped.isdigit():
        return 'number'
    if any(character.isalpha() for character in stripped):
        return 'word_start' if text[0].isspace() else 'word_piece'
    return 'punctuation'


def ranking(labels, values):
    valid = np.isfinite(values)
    truth, score = labels[valid], values[valid]
    return dict(auroc=float(roc_auc_score(truth, score)),
                ap=float(average_precision_score(truth, score)),
                tokens=int(valid.sum()), wrong=int(truth.sum()))


def source_bootstrap(sources, values, draws=1000):
    """Average within each source, then bootstrap source means."""
    source_means = []
    for source in sorted(set(sources)):
        selected = np.asarray(sources) == source
        finite = selected & np.isfinite(values)
        if finite.any():
            source_means.append(np.mean(np.asarray(values)[finite]))
    generator = np.random.default_rng(42)
    means = np.asarray(source_means)
    bootstrap = generator.choice(means, (draws, len(means)), replace=True).mean(axis=1)
    return dict(mean=float(means.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                sources=len(means), pairs=int(np.isfinite(values).sum()))


def pair_statistics(rows, features, pairs, names):
    sources, differences = [], {name: [] for name in names}
    concordances = {name: [] for name in names}
    for row in rows:
        for target, control in pairs[row['id']]:
            sources.append(row['source_id'])
            for name in names:
                values = features[row['id']][name]
                difference = float(values[target] - values[control])
                differences[name].append(difference)
                concordances[name].append(float(difference > 0) + .5 * float(difference == 0)
                                          if np.isfinite(difference) else np.nan)
    return {name: dict(difference=source_bootstrap(sources, values),
                      concordance=source_bootstrap(sources, concordances[name]))
            for name, values in differences.items()}


def select_native_pairs(rows, features, strict, relaxed, recognized=8):
    """Gold/teacher-guided mechanism cohort, frozen before native interventions."""
    candidates = []
    critical = {'12297', '12219', '12471', '17199'}
    for row in rows:
        if not strict[row['id']] or row['id'] in critical:
            continue
        risk = features[row['id']]['teacher']
        target, control = max(strict[row['id']], key=lambda pair: risk[pair[0]] - risk[pair[1]])
        candidates.append(dict(id=row['id'], source_id=row['source_id'], target=target,
            control=control, matching='exact_BPE', kind='teacher_recognized',
            teacher_gap=float(risk[target] - risk[control])))
    selected = sorted(candidates, key=lambda row: row['teacher_gap'], reverse=True)[:recognized]
    for row in rows:
        if row['id'] not in critical:
            continue
        pairs = strict[row['id']] or relaxed[row['id']]
        if not pairs:
            continue
        risk = features[row['id']]['teacher']
        target, control = max(pairs, key=lambda pair: risk[pair[0]] - risk[pair[1]])
        selected.append(dict(id=row['id'], source_id=row['source_id'], target=target,
            control=control, matching='exact_BPE' if strict[row['id']] else 'surface_type',
            kind='critical_miss', teacher_gap=float(risk[target] - risk[control])))
    return selected
