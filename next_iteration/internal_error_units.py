"""N9 controlled error units. Inputs and construction labels are separate artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20260914
LAYERS = [7, 15, 23, 31]
TEMPLATES = [
    ("location", "resident", "home city", ["Lima", "Oslo", "Rome", "Bern"]),
    ("contents", "container", "contents", ["silver coins", "copper keys", "glass beads", "paper clips"]),
    ("identifier", "device", "access code", ["4821", "7364", "9152", "2683"]),
    ("time", "appointment", "scheduled day", ["Monday", "Tuesday", "Friday", "Sunday"]),
    ("duration", "stage", "duration", ["12 minutes", "18 minutes", "24 minutes", "36 minutes"]),
    ("polarity", "facility", "operating status", ["open", "closed", "active", "inactive"]),
    ("tax", "income category", "tax rate", ["12 percent", "18 percent", "24 percent", "36 percent"]),
    ("color", "product", "color", ["dark blue", "pale green", "bright red", "light gray"]),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fixtures():
    """No prediction features may use the construction truth returned separately."""
    cases, gold = [], {}
    rng = np.random.default_rng(SEED)
    for ti, (kind, noun, field, values) in enumerate(TEMPLATES):
        for group in range(12):
            sid = f'{kind}-{group:02d}'
            split = 'train' if group < 6 else 'calibration' if group < 8 else 'test'
            names = [f'{noun} {chr(65 + group)}ra', f'{noun} {chr(65 + group)}no']
            vi = rng.permutation(len(values))[:2]
            va, vb = values[int(vi[0])], values[int(vi[1])]
            target = group % 2
            prefix = '' if group % 3 == 0 else 'The source contains two separate entries. '
            unit_prefix = f'For {names[target]}, the {field} is '
            unit_tail = ' according to the entry in this record.'
            after = ' The two entries refer to different items.'
            for binding in range(2):
                statements = [f'The {field} of {names[i]} is {[va, vb][i ^ binding]}.' for i in range(2)]
                for order in range(2):
                    source = '\n'.join(statements[::1 if order == 0 else -1])
                    for draft in range(2):
                        wrong = binding ^ draft
                        value = [va, vb][target ^ draft]
                        response = prefix + unit_prefix + value + unit_tail + after
                        start = len(prefix + unit_prefix)
                        rid = f'{sid}-b{binding}-o{order}-d{draft}'
                        cases.append(dict(id=rid, source_id=sid, family=kind, split=split,
                            source=source, response=response,
                            instruction='Describe the specified entry accurately using only the source. '
                                        f'Focus on {names[target]}.'))
                        gold[rid] = dict(source_id=sid, family=kind, split=split,
                            condition='misbound' if wrong else 'supported',
                            core=[start, start + len(value)] if wrong else None,
                            candidate_value=[start, start + len(value)],
                            unit=[len(prefix), len(prefix + unit_prefix + value + unit_tail)],
                            scope=[start, len(prefix + unit_prefix + value + unit_tail)] if wrong else None,
                            pair_id=f'{sid}-o{order}-d{draft}', target_value=value,
                            correct_counterpart=f'{sid}-b{binding}-o{order}-d{binding}')
    return cases, gold


def tokenize_case(case, tokenizer):
    source = case['source']
    content = f"<SOURCE>\n{source}\n</SOURCE>\n{case['instruction']}"
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'Answer faithfully using the supplied source.'},
        {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt + case['response'], add_special_tokens=False,
                        return_offsets_mapping=True)
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded['input_ids'][:len(pids)] != pids:
        raise ValueError('prompt/response BPE boundary changed')
    p = len(pids)
    if any(b <= len(prompt) or a < len(prompt) for a, b in encoded['offset_mapping'][p:]):
        raise ValueError('response token crosses prompt boundary')
    slo = prompt.index('<SOURCE>\n') + len('<SOURCE>\n')
    shi = slo + len(source)
    row = {k: case[k] for k in ['id', 'source_id', 'family', 'split', 'response']}
    row.update(prompt=prompt, prompt_length=p, token_ids=encoded['input_ids'],
        source_mask=[max(a, slo) < min(b, shi) for a, b in encoded['offset_mapping'][:p]],
        offsets=[[a-len(prompt), b-len(prompt)] for a,b in encoded['offset_mapping'][p:]],
        mode='controlled_teacher_forcing',
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        response_sha256=hashlib.sha256(case['response'].encode()).hexdigest())
    return row


def token_targets(offsets, gold):
    """Scope is semantic completion suffix, NOT a hallucination-token label."""
    offsets = np.asarray(offsets, dtype=int)
    targets = {k: np.zeros(len(offsets), np.int8) for k in ['onset', 'core', 'scope']}
    for name in ['core', 'scope']:
        interval = gold.get(name)
        if interval is not None:
            a, b = interval
            targets[name] = (np.maximum(offsets[:, 0], a) < np.minimum(offsets[:, 1], b)).astype(np.int8)
    positions = np.flatnonzero(targets['core'])
    if len(positions):
        onset = gold.get('onset_token', int(positions[0]))
        targets['onset'][onset] = 1
        targets['core'][:onset] = 0
        targets['scope'][:onset] = 0
    return targets


def prepare(output, model, natural):
    from transformers import AutoTokenizer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases, gold = fixtures()
    rows = [tokenize_case(c, tok) for c in cases]
    by_id = {r['id']: r for r in rows}
    for row in rows:
        g = gold[row['id']]
        if g['core']:
            other = by_id[g['correct_counterpart']]
            a, b = row['token_ids'][row['prompt_length']:], other['token_ids'][other['prompt_length']:]
            first = next(i for i, (x,y) in enumerate(zip(a,b)) if x != y)
            g['onset_token'] = first
            if not (row['offsets'][first][1] > g['core'][0] and row['offsets'][first][0] < g['core'][1]):
                raise ValueError('first divergent token is outside the factual core')
        t = token_targets(row['offsets'], g)
        if g['core'] and (t['onset'].sum() != 1 or not t['core'].sum() or not t['scope'].sum()):
            raise ValueError('incomplete construction annotation')
    natural_path = Path(natural)
    natural_rows = [json.loads(s) for s in natural_path.read_text().splitlines()]
    natural_rows = [r for r in natural_rows if r['split'] == 'development']
    if len(natural_rows) != 32:
        raise ValueError('fixed natural development must contain 32 responses')
    for r in natural_rows:
        row = {k: r[k] for k in ['id', 'source_id', 'prompt', 'response', 'prompt_length',
                               'token_ids', 'source_mask', 'offsets', 'prompt_sha256', 'response_sha256']}
        row.update(id='natural-' + str(r['id']), source_id='natural-' + str(r['source_id']),
                   family=r['task'], split='natural', mode='observer_replay')
        rows.append(row)
    with (output/'inputs.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(output/'gold.json', gold)
    for split in ['train', 'calibration', 'test']:
        write_json(output/f'gold_{split}.json', {k:v for k,v in gold.items() if v['split']==split})
    write_json(output/'manifest.json', dict(complete=True, controlled=768, natural=32,
        groups=96, seed=SEED, model=str(Path(model).resolve()),
        tokenizer_metadata_sha256={p.name:digest(p) for p in Path(model).glob("*.json")},
        natural_roster_sha256=digest(natural_path),
        inputs_sha256=digest(output/'inputs.jsonl'), gold_sha256=digest(output/'gold.json'),
        code_sha256=digest(__file__), max_tokens=max(len(r['token_ids']) for r in rows),
        split_gold_sha256={s:digest(output/f'gold_{s}.json') for s in ['train','calibration','test']},
        full_semantic_unit_truth='construction only; natural not available'))
    print(json.dumps(dict(prepared=len(rows), controlled=768, natural=32)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--natural', required=True)
    args = p.parse_args()
    prepare(args.output, args.model, args.natural)
