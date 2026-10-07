"""Exact saved-token worlds and prompt-only source masks for native interventions."""
from difflib import SequenceMatcher
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from state_audit.dataset import Example
from state_audit.tokenization import encode_prompt
from experiments.native_support.ragtruth_benchmark.data import encode_source
from .grounded_projection_data import DATASET, MODEL, write_json


ATTENTION = Path('/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b')


def encode_passage(source, tokenizer, passage):
    """Explicit passage-number diagnosis; this mask is not inferred applicability."""
    prompt = source['prompt']
    start = prompt.index(f'passage {passage}:')
    marker = f'\n\npassage {passage+1}:' if passage < 3 else '\nIn case the passages'
    stop = prompt.index(marker, start)
    span = dict(id=f'passage {passage}', start=start, end=stop)
    example = Example(str(source['source_id']), str(source['source_id']), prompt, [span])
    encoded = encode_prompt(tokenizer, example, 'chat')
    return dict(prompt_with_source=encoded['prompt_ids'],
                source_mask=(np.asarray(encoded['key_sources']) >= 0).tolist())


def transfer_source_mask(canonical, token_ids):
    mapping = SequenceMatcher(a=canonical['prompt_with_source'], b=token_ids, autojunk=False)
    mask = np.zeros(len(token_ids), dtype=bool)
    seen = np.zeros(len(canonical['source_mask']), dtype=bool)
    original = np.asarray(canonical['source_mask'], bool)
    for first, second, count in mapping.get_matching_blocks():
        mask[second:second + count] = original[first:first + count]
        seen[first:first + count] = True
    if not seen[original].all():
        raise ValueError('Canonical source token did not match the saved attention world')
    return mask


def prepare_native(output, directory_name='native'):
    pairs = json.loads((output / 'native_pairs.json').read_text())
    index = {str(r['sample_id']): r for r in map(json.loads, (ATTENTION / 'test/index.jsonl').open())}
    sources = {str(r['source_id']): r for r in map(json.loads, (DATASET / 'source_info.jsonl').open())}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    inputs = []
    directory = output / directory_name
    directory.mkdir(exist_ok=False)
    for pair in pairs:
        record = index[pair['id']]
        with np.load(ATTENTION / 'test' / record['path']) as saved:
            tokens = saved['token_ids'].tolist()
            prompt = int(saved['response_idx'])
            diagonal = saved['attention_diagonal'].reshape(1024, len(tokens))
            np.savez_compressed(directory / (pair['id'] + '_original_self.npz'),
                selected=diagonal[:, prompt + np.asarray([pair['target'], pair['control']])])
        canonical = encode_source(sources[pair['source_id']], tokenizer)
        if 'source_passage' in pair:
            canonical = encode_passage(sources[pair['source_id']], tokenizer, pair['source_passage'])
        mask = transfer_source_mask(canonical, tokens[:prompt])
        inputs.append(dict(**pair, token_ids=tokens, prompt_length=prompt,
            source_mask=mask.tolist(), source_tokens=int(mask.sum()),
            target_text=tokenizer.decode([tokens[prompt + pair['target']]]),
            control_text=tokenizer.decode([tokens[prompt + pair['control']]])))
    write_json(directory / 'inputs.json', dict(model=MODEL, cases=inputs,
        selection='gold/teacher-guided matched mechanism cohort',
        exact_original_token_world=True, generator='Llama2-7B', observer='Llama3.1-8B',
        source_mask='canonical mask transferred only through exact equal token blocks'))
    return inputs
