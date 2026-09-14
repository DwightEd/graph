"""CPU-only reconstruction of P5/P6/P7 final-judge input differences, no labels."""
import ast, hashlib, json
from pathlib import Path
from . import local_grounding as p5, local_grounding_review as p6, local_grounding_reasoned as p7
from .development_assess_scoped import DEV_IDS, frozen_records

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/p7_reasoned_development_20260914_v1/draft_input_contract.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    specs=[('P5',p5,'p5_local_development_20260914_v3_fp32'),
           ('P6',p6,'p6_review_development_20260914_v1'),
           ('P7',p7,'p7_reasoned_development_20260914_v1')]
    manifests={};records={};trees={}
    for name,module,directory in specs:
        root=ROOT/'outputs'/directory
        m,rows=frozen_records(root)
        assert {str(r['id']) for r in rows}==DEV_IDS and m['settings']['phase']=='development'
        assert sha(module.__file__)==sha(root/'executed_code.py')==m['code_sha256']
        manifests[name]=m;records[name]={str(r['id']):r for r in rows}
        trees[name]=ast.parse(Path(module.__file__).read_text())
    assert p5.SYSTEM==p6.SYSTEM==p7.SYSTEM
    identity={}
    for name in ['messages','units','local_sentence','common_prefix','align_scores','fp32_linear','tokenize','summarize','verify']:
        nodes=[next(n for n in ast.walk(trees[k]) if isinstance(n,ast.FunctionDef) and n.name==name) for k in ['P5','P6','P7']]
        identity[name]=len({ast.dump(n) for n in nodes})==1
    assert all(identity.values()),identity
    setups=[]
    for key in ['P5','P6','P7']:
        main_node=next(n for n in trees[key].body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=next(n for n in main_node.body if isinstance(n,ast.Try))
        end=next(i for i,n in enumerate(block.body) if isinstance(n,ast.FunctionDef) and n.name=='tokenize')
        setups.append(ast.dump(ast.Module(body=block.body[:end],type_ignores=[])))
    setup_identical=len(set(setups))==1
    # Retain a mismatch report if implementation/setup differs; do not silently
    # call it a controlled contrast. The final report requires this boolean.
    for field in ['precision','label_ids','model_metadata_sha256','versions']:
        assert manifests['P5'][field]==manifests['P6'][field]==manifests['P7'][field],field
    assert all(m['settings']['batch_size']==4 for m in manifests.values())
    inputs=ROOT.parents[1]/'research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl'
    assert sha(inputs)==p5.INPUT_SHA==p6.INPUT_SHA==p7.INPUT_SHA
    rows=[json.loads(line) for line in inputs.read_text().splitlines()]
    rows=[r for r in rows if str(r['id']) in DEV_IDS]
    checks=[];nwords=0
    for row in rows:
        rid=str(row['id']);lo,hi=row['source_span']
        source=row['prompt'][lo:hi]
        instruction=row['prompt'][:lo]+'[SOURCE OMITTED HERE]'+row['prompt'][hi:]
        spans=p5.units(row['response'])
        for key in ['P5','P6','P7']:
            r=records[key][rid]
            assert r['word_spans']==[list(s) for s in spans]
            assert r['response_sha256']==row['response_sha256'] and r['prompt_sha256']==row['prompt_sha256']
        digests={key:hashlib.sha256() for key in ['base','P6','P7']}
        for span in spans:
            base=p5.messages(source,instruction,row['response'],span)
            assert base==p6.messages(source,instruction,row['response'],span)==p7.messages(source,instruction,row['response'],span)
            digests['base'].update(json.dumps(base,ensure_ascii=False,separators=(',',':')).encode())
            for key,module in [('P6',p6),('P7',p7)]:
                d=records[key][rid]['draft_audit']
                changed=module.add_audit(base,d['text'],d['truncated'])
                marker='</ANSWER>\n';position=base[1]['content'].index(marker)+len(marker)
                prefix=base[1]['content'][:position];suffix=base[1]['content'][position:]
                assert changed[0]==base[0] and changed[1]['content'].startswith(prefix) and changed[1]['content'].endswith(suffix)
                assert changed[1]['content']!=base[1]['content']
                digests[key].update(json.dumps(changed,ensure_ascii=False,separators=(',',':')).encode())
            nwords+=1
        checks.append(dict(id=rid,words=len(spans),reconstructed_prompt_stream_sha256={k:h.hexdigest() for k,h in digests.items()}))
    report=dict(status='PASS' if setup_identical else 'WARN_SETUP_DIFF',responses=len(rows),words=nwords,
        annotations_opened=0,model_forwards=0,helper_ast_identity=identity,model_setup_ast_identical=setup_identical,
        final_base_messages_exactly_identical=True,only_prompt_insertion='entire draft audit text plus truncation/completeness warning',
        score_model='same fixed weights/metadata/precision/batch4/final nonthinking deterministic classifier',
        interpretation='Reconstructed final-judge input contrast, not a new GPU intervention or sentence-level binding causal test. Entire audit payload including length/warning changes jointly; does not prove original-generator causality.',
        code_sha256=sha(__file__),input_sha256=sha(inputs),
        scorer_sha256={k:m['code_sha256'] for k,m in manifests.items()},checks=checks)
    with OUTPUT.open('x') as f:json.dump(report,f,indent=2)
    OUTPUT.with_suffix('.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
