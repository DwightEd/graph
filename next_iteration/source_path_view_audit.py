"""CPU-only typed-source round-trip audit on original development inputs."""
from collections import Counter
import hashlib,json
from pathlib import Path
from .source_path_view import prepare

BASE=Path('/share/home/tm902089733300000/a903202310/lys')
INPUTS=BASE/'research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl'
OUT=BASE/'research/graph/outputs/source_path_view_development_20260914_v1'
EXPECTED='3258b79c7886dc2f6f93ff6c6e9e38c695a773b7dab2ddfc4cba271a3baac38f'
DEV_IDS={'11649','11650','11847','11848','13083','13084','13515','13516',
 '13857','13858','15603','15604','15939','15940','17019','17020','1773','1774',
 '219','220','3687','3688','4449','4450','4707','4708','7305','7306','741','742','9021','9022'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if sha(INPUTS)!=EXPECTED:raise ValueError('original roster identity changed')
    rows=[json.loads(line) for line in INPUTS.read_text().splitlines()]
    rows=[r for r in rows if str(r['id']) in DEV_IDS]
    if len(rows)!=32:raise ValueError('original development32 required')
    selected=[r for r in rows if r['task']=='Data2txt']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(BASE/'models/Qwen3-8B',local_files_only=True)
    OUT.mkdir(parents=True,exist_ok=False)
    results=[]
    errors=[]
    for row in selected:
        a,b=row['source_span'];source=row['prompt'][a:b]
        try:
            view=prepare(source)
            record=dict(id=row['id'],source_id=row['source_id'],source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                source_tokens=len(tokenizer.encode(source,add_special_tokens=False)),
                view_tokens=len(tokenizer.encode(view['text'],add_special_tokens=False)),
                scalar_types=dict(Counter(n['kind'] for n in view['tree']['nodes'])),
                false_count=sum(n['kind']=='bool' and n['value'] is False for n in view['tree']['nodes']),
                null_count=sum(n['kind']=='NoneType' for n in view['tree']['nodes']),**view)
            path=OUT/f"response_{row['id']}.json"
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            results.append(dict(id=row['id'],source_id=row['source_id'],path=path.name,sha256=sha(path),
                source_tokens=record['source_tokens'],view_tokens=record['view_tokens'],
                false_count=record['false_count'],null_count=record['null_count'],nodes=len(view['tree']['nodes'])))
        except Exception as exc:errors.append(dict(id=row['id'],error=repr(exc)))
    report=dict(status='pass' if not errors else 'failure_retained',responses=len(selected),
        sources=len({str(r['source_id']) for r in selected}),results=results,errors=errors,
        input_sha256=EXPECTED,executed_code_sha256=sha(__file__),
        projection_code_sha256=sha(Path(__file__).with_name('source_path_view.py')),
        model_forwards=0,natural_annotations_read=0,
        scope='Original development Data2txt source-only representation audit; no response scoring, no P7 confirmation access.',
        claim_ceiling='Round-trip exact parsed values and bindings; does not establish detection improvement or semantic graph necessity.')
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'executed_projection.py').write_bytes(Path(__file__).with_name('source_path_view.py').read_bytes())
    print(json.dumps(report))
    if errors:raise SystemExit(2)

if __name__=='__main__':main()
