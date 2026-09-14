"""B8 synthetic relation-routing fixture, no model, annotations or detector fit.

Two independent source/draft assignments crossed with both fact orders. Every
assignment has the same lexical multiset. A no-draft cell is a context baseline,
not a length-matched causal contrast. Natural P7 confirmation is untouched.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Family:
    name: str
    template: str
    roles: tuple[str, str]
    values: tuple[str, str]
    answer: str
    target: str


FAMILIES = (
    Family('stage', '{role} baking, wait for {value} minutes.',
           ('After', 'Before'), ('12', '20'), 'After baking, wait for 12 minutes.', '12'),
    Family('object', 'Steam the {role} for {value} minutes.',
           ('beans', 'carrots'), ('8', '12'), 'Steam the beans for 8 minutes.', '8'),
    Family('entity', 'The {role} box contains a {value}.',
           ('blue', 'red'), ('key', 'coin'), 'The blue box contains a key.', 'key'),
    Family('polarity', 'The restaurant has {role}: {value}.',
           ('WiFi', 'parking'), ('yes', 'no'), 'The restaurant has WiFi.', 'WiFi'),
    Family('time', 'On {role}, the restaurant closes at {value}.',
           ('Monday', 'Tuesday'), ('22:30', '02:30'),
           'On Monday, the restaurant closes at 22:30.', '22:30'),
    Family('quantifier', 'Project {role} took {value} four days.',
           ('Alpha', 'Beta'), ('exactly', 'more than'),
           'Project Alpha took exactly four days.', 'exactly'),
    Family('condition', 'For {role} shipments, the fee is {value} dollars.',
           ('domestic', 'foreign'), ('5', '9'),
           'For domestic shipments, the fee is 5 dollars.', '5'),
    Family('tax_binding', 'Income from {role} is taxed at {value} percent.',
           ('pensions', 'wages'), ('0', '5'),
           'Income from pensions is taxed at 0 percent.', '0'),
)


def lexical_bag(text):
    return Counter(re.findall(r'\w+|[^\w\s]', text))


def facts(family, assignment, order):
    if assignment not in (0, 1) or order not in (0, 1):
        raise ValueError('binary factors required')
    clauses = [family.template.format(role=role, value=family.values[i ^ assignment])
               for i, role in enumerate(family.roles)]
    return ' '.join(clauses[::1 if order == 0 else -1])


def build():
    rows = []
    for family in FAMILIES:
        match = list(re.finditer(re.escape(family.target), family.answer))
        if len(match) != 1:
            raise ValueError(f'nonunique target: {family.name}')
        target_span = list(match[0].span())
        reference_bag = lexical_bag(facts(family, 0, 0))
        for source_assignment in (0, 1):
            for source_order in (0, 1):
                source = facts(family, source_assignment, source_order)
                assert lexical_bag(source) == reference_bag
                common = dict(family=family.name, source=source, answer=family.answer,
                              target=family.target, target_span=target_span,
                              source_assignment=source_assignment, source_order=source_order,
                              constructed_expected='A' if source_assignment == 0 else 'B',
                              ground_truth_type='synthetic_by_construction_not_RAGTruth',
                              instruction='Report only source-supported facts.')
                rows.append(dict(common, id=f'{family.name}/s{source_assignment}/o{source_order}/none',
                                 draft=None, draft_assignment=None, draft_order=None,
                                 draft_agrees_with_source=None))
                for draft_assignment in (0, 1):
                    for draft_order in (0, 1):
                        draft = facts(family, draft_assignment, draft_order)
                        assert lexical_bag(draft) == reference_bag
                        rows.append(dict(common,
                            id=f'{family.name}/s{source_assignment}/o{source_order}/d{draft_assignment}/r{draft_order}',
                            draft=draft, draft_assignment=draft_assignment, draft_order=draft_order,
                            draft_agrees_with_source=draft_assignment == source_assignment))
    assert len(rows) == 160 and len({r['id'] for r in rows}) == 160
    return rows


def write_fixture(output):
    out = Path(output)
    rows = build()
    out.mkdir(parents=True, exist_ok=False)
    raw = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()
    (out/'inputs.jsonl').write_bytes(raw)
    snapshot = Path(__file__).read_bytes()
    (out/'executed_builder.py').write_bytes(snapshot)
    meta = dict(status='synthetic_inputs_only', rows=len(rows), families=len(FAMILIES),
                source_assignments=2, source_orders=2, draft_cells=5,
                all_families_retained=True, model_forwards=0, natural_annotations_read=0,
                lexical_multiset_verified=True, token_length_match='pending_real_tokenizer_check',
                inputs_sha256=hashlib.sha256(raw).hexdigest(),
                builder_sha256=hashlib.sha256(snapshot).hexdigest(),
                limitations=['No natural efficacy or original-generator claim.',
                             'Token lengths must be checked before interpreting relation-only contrast.',
                             'No-draft versus draft contrast changes context length.',
                             'Eight hand-written families are not population-general evidence.'])
    (out/'manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    print(json.dumps(write_fixture(p.parse_args().output)))
