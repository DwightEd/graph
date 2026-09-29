"""Conservative surface constraints with explicit source witnesses and abstention.

This parser is a hypothesis, not a general entailment oracle. Duration alignment
uses lexical event overlap; its interval proof is conditional on that alignment.
No example IDs, gold spans, detector thresholds, or model predictions enter it.
"""

import re

import numpy as np


NUMBERS = dict(zip('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(1, 21)))
NUMBER_PATTERN = '|'.join(NUMBERS)
DURATION = re.compile(r'\bfor\s+(?:(?P<bound>more than|less than|at least|at most|over|under|exactly)\s+)?'
    rf'(?P<number>\d+(?:\.\d+)?|{NUMBER_PATTERN})\s+(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b', re.I)
COPULA = re.compile(r"(?P<subject>[a-z][a-z' -]{0,90}?)\s+(?P<verb>is|are|was|were)\s+(?P<neg>not\s+)?(?P<predicate>[a-z-]+)\b", re.I)
UNSUPPORTED_SCOPE = re.compile(r'\b(if|unless|might|may|could|would|should|either|neither|and|or|but)\b', re.I)
COMMON = set('this that these those with from have been were then they their there into about which where when more than days hours weeks months years police said says say only also after before'.split()) | set(NUMBERS)


def clauses(text):
    for match in re.finditer(r'[^,;.!?\n]+', text):
        if not UNSUPPORTED_SCOPE.search(match[0]):
            yield match


def subject_key(text):
    return re.sub(r'^(the|a|an)\s+', '', ' '.join(text.lower().split()))


def copula_facts(text, source):
    result = []
    for clause in clauses(text):
        match = COPULA.search(clause[0])
        if match is None:
            continue
        tail = clause[0][match.end():].strip()
        # Bare source predicates avoid erasing arguments or negation scope.
        if source and tail:
            continue
        if match['neg'] and tail:
            continue
        tense = 'present' if match['verb'].lower() in ('is', 'are') else 'past'
        result.append(dict(key=(subject_key(match['subject']), match['predicate'].lower(), tense),
            negative=bool(match['neg']), start=clause.start() + match.start(),
            core_end=clause.start() + match.end(), end=clause.end(), text=clause[0]))
    return result


def polarity_claims(source, answer):
    facts = copula_facts(source, source=True)
    claims = []
    for claim in copula_facts(answer, source=False):
        matches = [fact for fact in facts if fact['key'] == claim['key']]
        polarities = {fact['negative'] for fact in matches}
        if len(polarities) != 1:
            continue
        contradiction = claim['negative'] not in polarities
        claims.append(dict(start=claim['start'], end=claim['end'] if contradiction else claim['core_end'],
            status='contradiction' if contradiction else 'supported', schema='bare_copula_polarity',
            witnesses=matches, assumption='exact surface subject/predicate and tense; restricted scope'))
    return claims


def interval(bound, value):
    if bound in ('more than', 'over'):
        return value, False, float('inf'), False
    if bound == 'at least':
        return value, True, float('inf'), False
    if bound in ('less than', 'under'):
        return 0., True, value, False
    if bound == 'at most':
        return 0., True, value, True
    return value, True, value, True


def interval_relation(source, answer):
    low, low_closed, high, high_closed = source
    other_low, other_low_closed, other_high, other_high_closed = answer
    disjoint = high < other_low or other_high < low
    disjoint |= high == other_low and not (high_closed and other_low_closed)
    disjoint |= other_high == low and not (other_high_closed and low_closed)
    if disjoint:
        return 'contradiction'
    lower_subset = low > other_low or (low == other_low and (not low_closed or other_low_closed))
    upper_subset = high < other_high or (high == other_high and (not high_closed or other_high_closed))
    return 'supported' if lower_subset and upper_subset else 'unknown'


def sentence_words(text, start, stop):
    left = max(text.rfind(mark, 0, start) for mark in '.!?\n') + 1
    ends = [text.find(mark, stop) for mark in '.!?\n']
    right = min([position for position in ends if position >= 0] or [len(text)])
    return set(re.findall(r'[a-z]{4,}', text[left:right].lower())) - COMMON


def duration_facts(text):
    result = []
    for match in DURATION.finditer(text):
        number = match['number'].lower()
        value = NUMBERS[number] if number in NUMBERS else float(number)
        result.append(dict(start=match.start(), end=match.end(), text=match[0],
            unit=match['unit'].lower().rstrip('s'), bound=(match['bound'] or '').lower(),
            value=value, anchors=sorted(sentence_words(text, match.start(), match.end()))))
    return result


def duration_claims(source, answer):
    facts = duration_facts(source)
    claims = []
    for claim in duration_facts(answer):
        matches = [fact for fact in facts if fact['unit'] == claim['unit']]
        # Several source durations can refer to different events; abstain.
        if len(matches) != 1:
            continue
        fact = matches[0]
        anchors = set(fact['anchors']) & set(claim['anchors'])
        if len(anchors) < 2:
            continue
        status = interval_relation(interval(fact['bound'], fact['value']), interval(claim['bound'], claim['value']))
        if status == 'unknown':
            continue
        claims.append(dict(start=claim['start'], end=claim['end'], status=status,
            schema='duration_interval', witnesses=[fact], anchors=sorted(anchors),
            assumption='unique same-unit source duration and >=2 shared content words; event alignment heuristic'))
    return claims


def token_constraints(source, answer, offsets):
    claims = polarity_claims(source, answer) + duration_claims(source, answer)
    support = np.zeros(len(offsets), bool)
    contradiction = np.zeros(len(offsets), bool)
    offsets = np.asarray(offsets)
    for claim in claims:
        overlap = (offsets[:, 0] < claim['end']) & (offsets[:, 1] > claim['start'])
        if claim['status'] == 'supported':
            support |= overlap
        else:
            contradiction |= overlap
    status = contradiction.astype(np.int8) - support.astype(np.int8)
    return status, claims


def apply_constraints(native, status):
    result = np.asarray(native).copy()
    result[status == 1] = 1.
    result[status == -1] = 0.
    return result
