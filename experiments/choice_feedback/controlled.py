"""Balanced value/order/query-role controls, fixed head and candidate contrast."""
import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL
from .native import full_margin


def scenarios(tokenizer):
    templates = [('original', 'current'), ('initial', 'updated'),
                 ('starting', 'latest'), ('previous', 'present')]
    values = [(20, 24), (30, 36), (40, 48), (50, 60)]
    result = []
    for family, pair, reverse, swap, query in product(range(4), range(4), range(2), range(2), range(2)):
        roles = templates[family]
        numbers = values[pair][::(-1 if swap else 1)]
        clauses = [f'The {role} price is ${value}.' for role, value in zip(roles, numbers)]
        description = ' '.join(clauses[::(-1 if reverse else 1)])
        text = (f'Question: {description} Calculate 20% of the {roles[query]} price.\n'
                f'Answer: To calculate 20% of the {roles[query]} price, multiply 0.20 by $')
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        candidates = [tokenizer.encode(str(value), add_special_tokens=False) for value in numbers]
        assert all(len(ids) == 1 for ids in candidates)
        begin = text.index(f'${numbers[0]}.') + 1
        end = begin + len(str(numbers[0]))
        keys = [i for i, (left, right) in enumerate(encoded['offset_mapping'])
                if min(right, end) > max(left, begin)]
        assert keys
        result.append(dict(family=family, pair=pair, reverse=reverse, swap=swap, query=query,
            text=text, tokens=encoded['input_ids'], candidates=[ids[0] for ids in candidates],
            source_keys=keys, roles=roles, numbers=numbers))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = scenarios(tokenizer)
    write_json(args.output / 'plan.json', dict(rows=rows, head=[17, 27], dose=.5,
        generated_labels='query role and value mapping, evaluation only',
        no_training=True, natural_generalization=False,
        scope='128 constructed prompts; 4 lexical variants of one arithmetic schema'))
    torch.manual_seed(42)
    model = load_model(MODEL)
    started = perf_counter()
    measured = []
    for index, row in enumerate(rows):
        probe = dict(layer=17, head=27, receiver=len(row['tokens']) - 1)
        margins = [float(full_margin(model, row['tokens'], probe['receiver'],
                   row['candidates'], probe, row['source_keys'], dose)[0]) for dose in (0., .5)]
        measured.append(dict(**row, margin=margins[0], source_effect=margins[1] - margins[0],
            correct=bool((margins[0] > 0) == (row['query'] == 0))))
        if (index + 1) % 16 == 0:
            print('controlled', index + 1, len(rows), flush=True)
    write_json(args.output / 'measurements.json', measured)
    groups = {}
    for family in range(4):
        subset = [row for row in measured if row['family'] == family]
        effects = [[row['source_effect'] for row in subset if row['query'] == query] for query in (0, 1)]
        groups[str(family)] = dict(roles=subset[0]['roles'], count=len(subset),
            accuracy=float(np.mean([row['correct'] for row in subset])),
            source_effect_by_query=[float(np.mean(effect)) for effect in effects])
    paired = []
    for left, right in zip(measured[::2], measured[1::2]):
        assert left['query'] == 0 and right['query'] == 1
        paired.append(dict(role_margin_change=right['margin'] - left['margin'],
            source_role_interaction=right['source_effect'] - left['source_effect']))
    write_json(args.output / 'summary.json', dict(status='complete', prompts=len(rows),
        native_forwards=2 * len(rows), accuracy=float(np.mean([row['correct'] for row in measured])),
        families=groups, pairs=paired, seconds=perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        claim='controlled input-role sensitivity only; no learned role probe or natural detector'))


if __name__ == '__main__':
    main()
