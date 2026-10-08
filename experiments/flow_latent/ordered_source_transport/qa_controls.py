"""QA-source literal quote ownership/availability, not factual entailment labels."""
import json

from ..provenance_joint_state.source_control_coverage import ROSTER, SOURCE_FILE, passage_records


def render_quote_control(tokenizer, source, quote, family, cue):
    if family == 'qa_owner':
        instruction = ('Which of passages 2 and 3 contains the exact quotation? '
                       'Use literal containment. Answer with 2 or 3 only.')
        suggested = '2' if cue == 'True' else '3'
        prefix = f'Unverified working note: the candidate passage is {suggested}. The matching source is Passage '
        words = ('2', '3')
    else:
        instruction = ('Complete "Passage 3 does" with provide if passage 3 contains the exact '
                       'quotation and not otherwise. This is literal quote availability.')
        suggested = 'provide' if cue == 'True' else 'not'
        prefix = f'Unverified working note: the candidate word is {suggested}. Passage 3 does'
        words = (' provide', ' not')
    user = ('Resolve the quotation from the source passages. Working notes are unverified. ' +
            instruction + '\nSource passages:\n' + source + '\nExact quotation:\n' + quote)
    prompt = tokenizer.apply_chat_template([dict(role='user', content=user)],
                                           tokenize=False, add_generation_prompt=True) + prefix
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    begin = prompt.index(source)
    end = begin + len(source)
    mask = [right > begin and left < end for left, right in encoded['offset_mapping']]
    candidates = []
    for word in words:
        complete = tokenizer.encode(prompt + word, add_special_tokens=False)
        assert complete[:-1] == encoded['input_ids']
        candidates.append(complete[-1])
    return dict(prompt=prompt, token_ids=encoded['input_ids'], source_mask=mask,
                candidate_ids=candidates, candidate_words=list(words), source_char_span=[begin, end])


def make_qa_controls(tokenizer):
    roster = json.loads(ROSTER.read_text())
    cohorts = {}
    for cohort, limit in [('fit', 12), ('dev', 4)]:
        for row in roster[cohort][:limit]:
            cohorts[row['source_id']] = cohort
    sources = {}
    for line in SOURCE_FILE.open():
        row = json.loads(line)
        identity = str(row['source_id'])
        if identity in cohorts:
            assert row['task_type'] == 'QA'
            sources[identity] = dict(passage_records(row['source_info']['passages']))
    controls = []
    for identity in sorted(sources):
        original = sources[identity]
        # The chat template strips trailing whitespace from the final user field.
        quotes = {role: original[role][:160].rstrip() for role in (2, 3)}
        feasible = all(len(quote) >= 8 and sum(quote in text for text in original.values()) == 1
                       for quote in quotes.values())
        if not feasible:
            continue
        swapped = dict(original)
        swapped[2], swapped[3] = original[3], original[2]
        for world, passages in [('original', original), ('swapped', swapped)]:
            source = '\n\n'.join(f'passage {i}:{passages[i]}' for i in (1, 2, 3))
            for role, quote in quotes.items():
                owners = [i for i in (2, 3) if quote in passages[i]]
                assert len(owners) == 1
                for family in ('qa_owner', 'qa_availability'):
                    correct = owners[0] - 2 if family == 'qa_owner' else int(quote not in passages[3])
                    for cue in ('True', 'False'):
                        control = render_quote_control(tokenizer, source, quote, family, cue)
                        control.update(id=f'{identity}/{world}/{role}/{family}/{cue}', source_id=identity,
                            world=world, role=str(role), family=family, cue=cue, quote=quote,
                            correct=correct, cohort=cohorts[identity])
                        controls.append(control)
    return controls
