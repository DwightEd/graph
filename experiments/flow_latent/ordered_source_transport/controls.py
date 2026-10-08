"""Balanced source-derived controls; no natural answers or annotations."""
import ast
import copy
import json

from ..provenance_joint_state.source_prechoice_pilot import source_value, set_source_value


def render_control(tokenizer, data, path, family, cue):
    source = json.dumps(data, ensure_ascii=False)
    instruction = 'Return True or False only.'
    prefix = f'Unverified working note: the candidate value is {cue}. Final value:'
    words = (' True', ' False')
    if family == 'polarity':
        instruction = ('Complete "The business does" with provide for True or not for False. '
                       'This is a controlled rendering of the requested boolean feature.')
        suggested = 'provide' if cue == 'True' else 'not'
        prefix = f'Unverified working note: the candidate word is {suggested}. The business does'
        words = (' provide', ' not')
    user = ('Read the structured source and resolve the exact JSON path. '
            'The working note is unverified; the source defines the answer. ' + instruction +
            '\nStructured source:\n' + source + '\nJSON path: ' + json.dumps(path))
    prompt = tokenizer.apply_chat_template([dict(role='user', content=user)],
                                           tokenize=False, add_generation_prompt=True) + prefix
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    begin = prompt.index(source)
    end = begin + len(source)
    source_mask = [right > begin and left < end for left, right in encoded['offset_mapping']]
    candidates = []
    for word in words:
        complete = tokenizer.encode(prompt + word, add_special_tokens=False)
        assert complete[:-1] == encoded['input_ids'], 'Candidate must append exactly one token'
        candidates.append(complete[-1])
    return dict(prompt=prompt, token_ids=encoded['input_ids'], source_mask=source_mask,
                candidate_ids=candidates, candidate_words=list(words), source_char_span=[begin, end])


def make_controls(tokenizer, rows, cohort):
    all_sources = sorted({row['source_id'] for row in rows})
    fit_sources = set(all_sources[:24])
    boolean_rows = [row for row in rows if row['family'] == 'boolean_attribute']
    fit_rows = [row for row in boolean_rows if row['source_id'] in fit_sources]
    dev_rows = [row for row in boolean_rows if row['source_id'] not in fit_sources]
    selected = fit_rows + dev_rows
    if cohort == 'pilot':
        selected = fit_rows[:6] + dev_rows[:2]
    controls = []
    for row in selected:
        literal = row['original_prompt'].split('Structured data:\n', 1)[1].rsplit('\nOverview:', 1)[0]
        original = ast.literal_eval(literal)
        swapped = copy.deepcopy(original)
        set_source_value(swapped, row['left'], row['value_right'])
        set_source_value(swapped, row['right'], row['value_left'])
        for world, data in [('original', original), ('swapped', swapped)]:
            for role in ('left', 'right'):
                path = row[role]
                truth = source_value(data, path)
                assert type(truth) is bool
                for family in ('literal', 'polarity'):
                    for cue in ('True', 'False'):
                        control = render_control(tokenizer, data, path, family, cue)
                        identity = f"{row['source_id']}/{world}/{role}/{family}/{cue}"
                        control.update(id=identity, source_id=row['source_id'], world=world,
                            role=role, family=family, cue=cue, path=path, correct=int(not truth),
                            cohort='fit' if row['source_id'] in fit_sources else 'dev')
                        controls.append(control)
    return controls
