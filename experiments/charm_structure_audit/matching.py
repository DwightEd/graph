"""Score-blind normal/error pairing. Statistics are NOT model inputs.

The three tiers and tolerances are the original cluster_audit protocol.
Prefer reusing saved pairs.json when comparing already completed models.
"""

import numpy as np

from .positions import merge_spans


NAMES = ('internal_density', 'internal_edge_share', 'internal_mass_share',
         'lag1_share', 'hub_concentration', 'log_mean_edge_mass', 'log_in_degree')
CALIPERS = np.array([.10, .10, .10, .10, .10, .50, .50])
TIERS = ('context', 'cluster', 'cluster_heads')


def surface_class(text):
    text = text.strip()
    if not text:
        return 'empty'
    if any(c.isdigit() for c in text):
        return 'number'
    return 'word' if any(c.isalpha() for c in text) else 'punctuation'


def incoming_statistics(graph):
    count = len(graph['x'])
    mass = np.zeros_like(graph['x'], dtype=float)
    weight_log = np.zeros_like(mass)
    nonzero = np.zeros_like(mass)
    edge_mass = np.zeros(graph['edge_index'].shape[1])
    for start in range(0, len(edge_mass), 4096):
        weights = graph['edge_attr'][start:start+4096].astype(float)
        target = graph['edge_index'][1, start:start+4096]
        np.add.at(mass, target, weights)
        np.add.at(weight_log, target, weights*np.log(np.maximum(weights, 1e-30)))
        np.add.at(nonzero, target, weights > 0)
        edge_mass[start:start+4096] = weights.sum(axis=1)
    entropy = np.log(np.maximum(mass, 1e-30)) - np.divide(weight_log, mass, out=np.zeros_like(mass), where=mass>0)
    entropy = np.divide(entropy, np.log(np.maximum(nonzero, 2)), out=np.zeros_like(mass), where=mass>0)
    # Prefix sums avoid retaining a large feature block for each candidate length.
    values = np.stack((graph['x'], mass, entropy), axis=1)
    prefix = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)))
    incoming_count = np.bincount(graph['edge_index'][1], minlength=count)
    return prefix, edge_mass, incoming_count


def structure(graph, edge_mass, counts, start, length):
    source, target = graph['edge_index']
    prompt = int(graph['prompt_length'])
    start = start + prompt
    inside_target = (target >= start) & (target < start+length)
    history = inside_target & (source >= prompt)
    inside = inside_target & (source >= start)
    number, mass = int(inside.sum()), float(edge_mass[inside].sum())
    total = float(edge_mass[inside_target].sum())
    hub = np.bincount(source[inside]-start, minlength=length)
    return np.array([number/(length*(length-1)/2), number/history.sum() if history.any() else 0.,
        mass/total if total else 0., float(np.mean(target[inside]-source[inside] == 1)) if number else 0.,
        float(np.square(hub/number).sum()) if number else 0.,
        np.log(max(mass/number, 1e-12)) if number else np.log(1e-12),
        np.log1p(counts[start:start+length].mean())])


def normal_candidates(sample, start, length):
    labels = sample['gold'].astype(bool)
    prefix = np.r_[0, np.cumsum(labels)]
    offsets = sample['offsets']
    text = str(sample['response'])
    classes = np.array([surface_class(text[a:b]) for a, b in offsets])
    invalid = np.r_[0, np.cumsum(offsets[:, 1] <= offsets[:, 0])]
    token_ids = sample['token_ids'][int(sample['prompt_length']):]
    repeat = np.array([1-len(np.unique(token_ids[i:i+length]))/length for i in range(len(labels)-length+1)])
    positions = np.arange(len(repeat))
    selected = (prefix[length:]-prefix[:-length] == 0) & (invalid[length:]-invalid[:-length] == 0)
    selected &= (prefix[positions] > 0) == (prefix[start] > 0)
    selected &= abs(positions-start)/len(labels) <= .25
    selected &= (classes[positions] == classes[start]) & (abs(repeat-repeat[start]) <= .15)
    return positions[selected], repeat


def candidate_pairs(graph, sample):
    prefix, mass, counts = incoming_statistics(graph)
    prompt = int(graph['prompt_length'])
    options, skipped = {}, []
    for start, end in merge_spans(sample['spans']):
        length = end-start
        if length < 2 or np.any(sample['offsets'][start:end, 1] <= sample['offsets'][start:end, 0]):
            skipped.append(dict(error_start=start, length=length, reason='singleton_or_nontext'))
            continue
        normals, repeat = normal_candidates(sample, start, length)
        expected = structure(graph, mass, counts, start, length)
        marginal = (prefix[prompt+end]-prefix[prompt+start])/length
        rows = []
        for normal in normals:
            actual = structure(graph, mass, counts, int(normal), length)
            delta = abs(actual-expected)/CALIPERS
            difference = abs((prefix[prompt+normal+length]-prefix[prompt+normal])/length-marginal)
            rms, maximum = np.sqrt(np.mean(difference**2, axis=1)), difference.max(axis=1)
            rows.append(dict(error_start=start, normal_start=int(normal), length=length,
                cluster_ok=bool(np.all(delta <= 1) and expected[0]>0 and actual[0]>0),
                heads_ok=bool(np.all(rms <= .05) and np.all(maximum <= .25)),
                structure_distance=float(np.sqrt(np.mean(delta**2))),
                position_gap=abs(int(normal)-start)/len(sample['gold']), repeat_gap=float(repeat[normal]-repeat[start]),
                error_structure=expected.tolist(), normal_structure=actual.tolist(),
                head_rms=rms.tolist(), head_max=maximum.tolist()))
        options[start] = rows
    return options, skipped


def match_answer(graph, sample):
    options, skipped = candidate_pairs(graph, sample)
    pairs, status = [], []
    for tier in TIERS:
        candidates = {}
        for start, rows in options.items():
            eligible = [r for r in rows if (tier=='context' or r['cluster_ok']) and (tier!='cluster_heads' or r['heads_ok'])]
            candidates[start] = sorted(eligible, key=lambda r: (
                r['position_gap'] if tier=='context' else r['structure_distance'], r['position_gap'], r['normal_start']))
        used = np.zeros(len(sample['gold']), bool)
        for start in sorted(candidates, key=lambda s:(len(candidates[s]), s)):
            chosen = None
            for row in candidates[start]:
                normal, length = row['normal_start'], row['length']
                if not used[normal:normal+length].any():
                    chosen = row
                    used[normal:normal+length] = True
                    break
            status.append(dict(error_start=start, tier=tier, candidates=len(candidates[start]), matched=chosen is not None))
            if chosen is not None:
                pairs.append(dict(chosen, tier=tier, id=str(sample['id']), source_id=str(sample['source_id'])))
    return pairs, status, skipped
