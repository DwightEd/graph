"""Source-only feasibility census; no natural response or annotation inputs."""
import json
import re
from collections import Counter
from pathlib import Path

from .matrix_audit import OUTPUT, sha256, write_json


ROSTER = Path('outputs/cached_head_transport_20261007_v1/roster.json')
CONTROLS = Path('outputs/full_head_message_design_20261007/role_controls.json')
SOURCE_FILE = Path('../../data/RAGTruth/dataset/source_info.jsonl')


def passage_records(text):
    sections = re.split(r'(?i)passage (\d+):', text)
    return [(int(sections[index]), sections[index + 1].strip())
            for index in range(1, len(sections), 2)]


def unique_quote_owners(passages):
    """A literal quote checks ownership only, never natural factual entailment."""
    owners = []
    for identity, text in passages:
        quote = text[:256]
        if quote and sum(quote in candidate for _, candidate in passages) == 1:
            owners.append(identity)
    return owners


def count_sources(roster, controls):
    cohorts = {name: {row['source_id'] for row in rows} for name, rows in roster.items()}
    program_sources = sorted({row['source_id'] for row in controls})
    program_fit = set(program_sources[:24])
    program_dev = set(program_sources[24:])
    tasks = {name: Counter() for name in cohorts}
    quote_counts = {name: dict(sources=0, owners=0) for name in ('fit', 'dev')}
    records = []
    for line in SOURCE_FILE.open():
        source = json.loads(line)
        identity = str(source['source_id'])
        for name, selected in cohorts.items():
            if identity in selected:
                tasks[name][source['task_type']] += 1
                if name in quote_counts:
                    passages = passage_records(source['source_info']['passages'])
                    owners = unique_quote_owners(passages)
                    feasible = len(owners) >= 2
                    quote_counts[name]['sources'] += feasible
                    quote_counts[name]['owners'] += len(owners) if feasible else 0
                    records.append(dict(source_id=identity, cohort=name,
                                        passage_count=len(passages), unique_quote_owners=owners))
    program = {}
    for name, selected in [('fit', program_fit), ('dev', program_dev)]:
        chosen = [row for row in controls if row['source_id'] in selected]
        program[name] = {family: dict(sources=len({row['source_id'] for row in chosen
                                                  if row['family'] == family}),
                                     controls=sum(row['family'] == family for row in chosen))
                         for family in sorted({row['family'] for row in controls})}
    summary = dict(cohort_tasks={name: dict(values) for name, values in tasks.items()},
                   data2txt_controls=program, qa_literal_quote_ownership=quote_counts,
                   data2txt_program_natural_test_overlap=sorted(set(program_sources) & cohorts['test']),
                   qa_factual_binding_targets_implemented=False,
                   qa_negation_scope_targets_implemented=False,
                   rendered_target_subword_contracts_implemented=False,
                   natural_response_or_label_files_read=False, llm_forwards=0,
                   interpretation='Counts only; quote ownership is not factual binding, and Data2txt-to-QA transfer is untested.')
    return summary, records


def main():
    roster = json.loads(ROSTER.read_text())
    controls = json.loads(CONTROLS.read_text())
    summary, records = count_sources(roster, controls)
    summary['inputs'] = {str(path): sha256(path) for path in (ROSTER, CONTROLS, SOURCE_FILE, Path(__file__))}
    write_json(OUTPUT / 'SOURCE_CONTROL_COVERAGE.json', summary)
    write_json(OUTPUT / 'QA_QUOTE_OWNER_INVENTORY.json', records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
