"""Source-only quote ownership targets; natural answers/annotations are not read.

Both values occur in every source. Swapping their owners reverses support for
the same candidate, so source membership and candidate frequency cannot label
the task. Labels concern a literal JSON field, not open-domain factual truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


CACHE = Path('outputs/native_support_ragtruth_all/source_first_v1')
TEMPLATES = {
    'quote_field': 'Return exactly the quote field for {owner}. Use the source objects, '
                   'preserve the field text, and output its value only.',
    'object_lookup': 'Look up object {owner} in the source. Copy its quote value verbatim '
                     'as your entire answer, without extra text.',
    'exact_field': 'The requested JSON path is {owner}.quote. Answer with the exact '
                   'string stored at this path and nothing else.',
}


def source_settings(source_id, partition):
    """Independent stable choices do not inspect candidates or target labels."""
    def index(axis, count):
        value = hashlib.sha256(f'{axis}:{source_id}'.encode()).digest()
        return int.from_bytes(value[:8], 'big') % count

    owners = ('owner_a', 'owner_b')
    key_order = owners if index('json_order', 2) == 0 else owners[::-1]
    templates = tuple(TEMPLATES)[:2] if partition == 'fit' else tuple(TEMPLATES)
    return dict(query_owner=owners[index('query_owner', 2)], key_order=key_order,
                template_id=templates[index('instruction_template', len(templates))])


def balanced_source_settings(sources):
    """Balance owner/order within the declared roster using independent hashes.

    Selection depends on source identities only. Expanding this roster defines
    a new program dataset and must not reuse an earlier capture protocol.
    """
    settings = {}
    for partition in ('fit', 'dev'):
        identities = [row['source_id'] for row in sources if row['partition'] == partition]
        ordered = sorted(identities, key=lambda identity: hashlib.sha256(
            f'query_owner:{identity}'.encode()).digest())
        middle = (len(ordered) + 1) // 2
        for owner, group in zip(('owner_a', 'owner_b'), (ordered[:middle], ordered[middle:])):
            by_order = sorted(group, key=lambda identity: hashlib.sha256(
                f'json_order:{identity}'.encode()).digest())
            split = (len(by_order) + 1) // 2
            for index, identity in enumerate(by_order):
                settings[identity] = source_settings(identity, partition)
                settings[identity]['query_owner'] = owner
                settings[identity]['key_order'] = (('owner_a', 'owner_b') if index < split
                                                    else ('owner_b', 'owner_a'))
    return settings


def decode_source_spans(tokenizer, token_ids, source_mask):
    """Decode contiguous source spans independently; never concatenate gaps."""
    spans = []
    start = None
    for position, included in enumerate([*source_mask, False]):
        if included and start is None:
            start = position
        if not included and start is not None:
            text = tokenizer.decode(token_ids[start:position], skip_special_tokens=False)
            spans.append(dict(start=start, stop=position, text=text))
            start = None
    return spans


def literal_quote_candidates(tokenizer, spans, value_tokens=8):
    """Use whole-word substrings of decoded source, with no model-made targets."""
    candidates = []
    seen = set()
    for span in spans:
        text = span['text']
        words = list(re.finditer(r'\S+', text))
        for index, word in enumerate(words):
            if word.group().lower() == 'passage':
                continue
            for stop in range(min(index + 12, len(words)), index + 1, -1):
                quote = text[word.start():words[stop - 1].end()]
                ids = tokenizer.encode(quote, add_special_tokens=False)
                if 3 <= len(ids) <= value_tokens and '\ufffd' not in quote:
                    break
            else:
                continue
            if quote not in seen:
                candidates.append(dict(text=quote, ids=ids, span_start=span['start'],
                    span_stop=span['stop'], char_start=word.start(), char_stop=words[stop - 1].end()))
                seen.add(quote)
    return candidates


def select_quote_pair(candidates, source_id):
    """Stable lexical selection uses source identity, not natural error labels."""
    ordered = sorted(candidates, key=lambda row: hashlib.sha256(
        (str(source_id) + ':' + row['text']).encode()).hexdigest())
    for left in ordered:
        for right in ordered:
            if (left['text'] != right['text'] and
                    not left['text'].startswith(right['text']) and
                    not right['text'].startswith(left['text'])):
                return left, right
    raise ValueError(f'{source_id}: source has no distinct non-prefix quote pair')


def render_program_prompt(tokenizer, values, world, query_owner='owner_a',
                          key_order=('owner_a', 'owner_b'), template_id='quote_field'):
    """The query is identical across counterfactual owner assignments."""
    assigned = values if world == 'original' else values[::-1]
    assignments = dict(owner_a=dict(quote=assigned[0]), owner_b=dict(quote=assigned[1]))
    facts = {owner: assignments[owner] for owner in key_order}
    source = json.dumps(facts, ensure_ascii=False)
    instruction = TEMPLATES[template_id].format(owner=query_owner)
    user = instruction + '\nSource objects:\n' + source
    prompt = tokenizer.apply_chat_template([dict(role='user', content=user)],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    begin = prompt.index(source)
    end = begin + len(source)
    mask = [right > begin and left < end for left, right in encoded['offset_mapping']]
    return dict(prompt=prompt, prompt_with_source=encoded['input_ids'], source_mask=mask,
        source_text=source, source_char_span=[begin, end], facts=facts, query_owner=query_owner,
        source_key_order=list(key_order), query_is_first=key_order[0] == query_owner,
        template_id=template_id, template_heldout=template_id == 'exact_field')


def encode_answer(tokenizer, prompt, answer):
    """Answer tokenization preserves the complete chat prefix exactly."""
    complete = tokenizer.encode(prompt['prompt'] + answer, add_special_tokens=False)
    prefix = prompt['prompt_with_source']
    if complete[:len(prefix)] != prefix:
        raise ValueError('Appending a quote changed prompt tokenization')
    return complete[len(prefix):]


def common_prefix_length(left, right):
    for position, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return position
    return min(len(left), len(right))


def program_token_targets(answer_ids, correct_ids, first_divergence):
    """Each token asks whether its complete candidate prefix matches the field.

    Shared candidate prefixes are excluded from the compatibility loss. At the
    first divergent position states are identical but candidate IDs differ;
    a prechoice reader must therefore receive the current candidate explicitly.
    """
    compatibility = [int(answer_ids[:position + 1] == correct_ids[:position + 1])
                     for position in range(len(answer_ids))]
    labels = [1 - value for value in compatibility]
    valid = [position >= first_divergence for position in range(len(answer_ids))]
    return dict(compatibility=compatibility, labels=labels, valid=valid)


def source_controls(tokenizer, source_id, partition, quote_pair, settings=None):
    """Four records: two source assignments crossed with two literal proposals."""
    values = [row['text'] for row in quote_pair]
    settings = source_settings(source_id, partition) if settings is None else settings
    records = []
    for world in ('original', 'swapped'):
        prompt = render_program_prompt(tokenizer, values, world, **settings)
        proposals = [encode_answer(tokenizer, prompt, value) for value in values]
        divergence = common_prefix_length(*proposals)
        if divergence == min(map(len, proposals)):
            raise ValueError('Quote token sequences must diverge before either ends')
        correct_index = values.index(prompt['facts'][prompt['query_owner']]['quote'])
        first_value_index = values.index(prompt['facts'][prompt['source_key_order'][0]]['quote'])
        for proposal_index, answer_ids in enumerate(proposals):
            targets = program_token_targets(answer_ids, proposals[correct_index], divergence)
            records.append(dict(prompt, **targets, id=f'program_{source_id}_{world}_{proposal_index}',
                source_id=str(source_id), partition=partition, task='QA', split='train',
                generator='source_program', tokens=len(answer_ids), answer_ids=answer_ids,
                answer_text=values[proposal_index], correct_answer_ids=proposals[correct_index],
                correct_answer_text=values[correct_index], world=world,
                proposal_index=proposal_index, correct_index=correct_index,
                source_world='a' if world == 'original' else 'b',
                candidate='A' if proposal_index == 0 else 'B',
                swap_group=f'{source_id}:{proposal_index}',
                candidate_is_first=proposal_index == first_value_index,
                first_divergence=divergence, extraction=quote_pair,
                family='literal_quote_owner', label_meaning='1=field-prefix incompatible',
                candidate_input_required=True))
    return records


def duplicate_owner_controls(tokenizer, source_id, partition, quote, settings=None):
    """Repeated values under different owners are valid, not synthetic negatives."""
    records = []
    settings = source_settings(source_id, partition) if settings is None else settings
    for owner in ('owner_a', 'owner_b'):
        prompt = render_program_prompt(tokenizer, [quote['text'], quote['text']],
            'original', owner, settings['key_order'], settings['template_id'])
        answer_ids = encode_answer(tokenizer, prompt, quote['text'])
        records.append(dict(prompt, **program_token_targets(answer_ids, answer_ids, 0),
            id=f'program_{source_id}_duplicate_{owner}', source_id=str(source_id),
            partition=partition, task='QA', split='train', generator='source_program',
            tokens=len(answer_ids), answer_ids=answer_ids, answer_text=quote['text'],
            correct_answer_ids=answer_ids, correct_answer_text=quote['text'],
            world='duplicate', proposal_index=0, correct_index=0, first_divergence=0,
            extraction=[quote], family='duplicate_owner_valid',
            label_meaning='1=field-prefix incompatible', candidate_input_required=True))
    return records


def select_train_sources(cache, fit_limit=96, dev_limit=32):
    """Use the project's frozen natural source partition, never answer labels."""
    from experiments.probabilistic_detection.data import select_records

    manifest = json.loads((cache / 'manifest.json').read_text())
    selected = select_records(manifest, 'QA', 'train', fit_limit, dev_limit)
    sources = {}
    for record in selected:
        sources.setdefault(str(record['source_id']), dict(source_id=str(record['source_id']),
            partition=record['partition'], source_file=record['source_file']))
    return manifest, list(sources.values())


def program_coverage(records, sources):
    partitions = {}
    token_sets, value_sets, template_sets, source_sets = {}, {}, {}, {}
    for partition in ('fit', 'dev'):
        selected = [row for row in records if row['partition'] == partition]
        token_sets[partition] = {token for row in selected for token in row['answer_ids']}
        value_sets[partition] = {row['answer_text'] for row in selected}
        template_sets[partition] = {row['template_id'] for row in selected}
        source_sets[partition] = {row['source_id'] for row in selected}
        scored = sum(sum(row['valid']) for row in selected)
        errors = sum(sum(label for label, valid in zip(row['labels'], row['valid']) if valid)
                     for row in selected)
        partitions[partition] = dict(sources=len({r['source_id'] for r in selected}),
            requested_sources=sum(row['partition'] == partition for row in sources),
            records=len(selected), answer_tokens=sum(row['tokens'] for row in selected),
            scored_tokens=scored, incompatible_tokens=errors,
            distinct_target_ids=len(token_sets[partition]), distinct_values=len(value_sets[partition]),
            template_record_counts={template: sum(row['template_id'] == template for row in selected)
                                    for template in sorted(template_sets[partition])},
            shortcut_diagnostics=shortcut_diagnostics(selected))
    return dict(partitions=partitions, requested_sources=len(sources),
        source_disjoint=not bool(source_sets['fit'] & source_sets['dev']),
        unseen_dev_target_ids=len(token_sets['dev'] - token_sets['fit']),
        repeated_fit_dev_values=len(value_sets['fit'] & value_sets['dev']),
        unseen_dev_values=len(value_sets['dev'] - value_sets['fit']),
        templates=sorted({row['template_id'] for row in records}),
        unseen_dev_templates=sorted(template_sets['dev'] - template_sets['fit']),
        template_holdout=bool(template_sets['dev'] - template_sets['fit']),
        families=sorted({row['family'] for row in records}),
        natural_answers_read=False, natural_annotations_read=False,
        task_scope='literal JSON ownership/prefix compatibility; not semantic entailment')


def shortcut_diagnostics(records):
    """Show positional and candidate-label marginals for the crossed task."""
    selected = [row for row in records if row['family'] == 'literal_quote_owner']
    if not selected:
        return dict(records=0)
    first = [row for row in selected if row['candidate_is_first']]
    later = [row for row in selected if not row['candidate_is_first']]
    incompatible = lambda rows: sum(row['proposal_index'] != row['correct_index'] for row in rows) / len(rows)
    return dict(records=len(selected), query_first_fraction=sum(row['query_is_first'] for row in selected) / len(selected),
        candidate_first_fraction=len(first) / len(selected),
        incompatible_record_fraction=incompatible(selected),
        incompatible_given_candidate_first=incompatible(first),
        incompatible_given_candidate_later=incompatible(later),
        query_owner_counts={owner: sum(row['query_owner'] == owner for row in selected)
                            for owner in ('owner_a', 'owner_b')},
        key_order_counts={order: sum(','.join(row['source_key_order']) == order for row in selected)
                          for order in ('owner_a,owner_b', 'owner_b,owner_a')})


def build_source_programs(cache, tokenizer, fit_limit=96, dev_limit=32,
                          value_tokens=8, include_duplicates=False):
    """Read only source caches, select two exact quotes, and render controls."""
    cache = Path(cache)
    _, sources = select_train_sources(cache, fit_limit, dev_limit)
    settings = balanced_source_settings(sources)
    records = []
    for source in sources:
        source_input = json.loads((cache / source['source_file']).read_text())
        spans = decode_source_spans(tokenizer, source_input['prompt_with_source'],
                                    source_input['source_mask'])
        quotes = literal_quote_candidates(tokenizer, spans, value_tokens)
        pair = select_quote_pair(quotes, source['source_id'])
        configured = settings[source['source_id']]
        records.extend(source_controls(tokenizer, source['source_id'], source['partition'], pair, configured))
        if include_duplicates:
            records.extend(duplicate_owner_controls(tokenizer, source['source_id'], source['partition'], pair[0], configured))
    return records, program_coverage(records, sources)


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-limit', type=int, default=96)
    parser.add_argument('--dev-limit', type=int, default=32)
    parser.add_argument('--value-tokens', type=int, default=8)
    parser.add_argument('--include-duplicates', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    records, coverage = build_source_programs(args.cache, tokenizer, args.fit_limit,
        args.dev_limit, args.value_tokens, args.include_duplicates)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'programs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    print(json.dumps(coverage), flush=True)


if __name__ == '__main__':
    main()
