"""B8 CPU-only complete factorial/tokenizer contract; never scores a model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from .binding_factorial_fixture import build, write_fixture
from .local_grounding_reasoned import messages, add_audit

P7_SHA='f6bb3332d3743733d5d8337e6753978c4ff7ded4a2d6476121bfd9f10a54589f'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_lengths(rows, prompts):
    groups=defaultdict(list)
    for row,ids in zip(rows,prompts,strict=True):
        groups[(row['family'], row['draft'] is None)].append(len(ids))
    return {f'{family}/{"no_draft" if absent else "draft"}':
            dict(rows=len(lengths),min=min(lengths),max=max(lengths),
                 matched=len(set(lengths))==1) for (family,absent),lengths in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    parent=Path(__file__).with_name('local_grounding_reasoned.py')
    if sha(parent)!=P7_SHA: raise ValueError('P7 query implementation changed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows=build()
    queries=[]
    for row in rows:
        msg=messages(row['source'],row['instruction'],row['answer'],row['target_span'])
        if row['draft'] is not None: msg=add_audit(msg,row['draft'],False)
        ids=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        queries.append(dict(id=row['id'],messages=msg,input_ids=ids))
    lengths=check_lengths(rows,[q['input_ids'] for q in queries])
    out=Path(args.output)
    meta=write_fixture(out)
    # Preserve every family and any failure rather than finding a favorable subset.
    (out/'tokenized_queries.json').write_text(json.dumps(queries,ensure_ascii=False)+'\n',encoding='utf-8')
    report=dict(status='pass' if all(v['matched'] for v in lengths.values()) else 'unmatched_lengths',
        rows=160,families=8,model_forwards=0,natural_annotations_read=0,
        parent_scorer_sha256=P7_SHA,fixture_manifest_sha256=sha(out/'manifest.json'),
        tokenizer_metadata_sha256={p.name:sha(p) for p in Path(args.model).glob('*.json')},
        tokenized_queries_sha256=sha(out/'tokenized_queries.json'),length_groups=lengths,
        prompt_scope='P7 unchanged final verifier, synthetic facts only, no generated audit or reasoning',
        qualification='Within-family draft assignment/order/source contrasts length-matched iff pass. No-draft comparison is NOT length-matched. No original-generator, natural efficacy, or universal mechanism claim.',
        executed_code_sha256=sha(__file__))
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (out/'executed_preflight.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))
    if report['status']!='pass': raise SystemExit(2)


if __name__=='__main__': main()
