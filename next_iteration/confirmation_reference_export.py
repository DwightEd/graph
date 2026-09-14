"""Label-free fixed64 confirmation export of existing observer baseline measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
from .grounding_contrast import digest, validate_rows, write_json

INPUT_SHA = 'bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',required=True)
    p.add_argument('--population',required=True)
    p.add_argument('--phase',choices=['validation'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if digest(args.inputs)!=INPUT_SHA:raise ValueError('input hash')
    rows=list(map(json.loads,Path(args.inputs).read_text().splitlines()))
    validate_rows(rows)
    if len(rows)!=64 or len({str(r['source_id']) for r in rows})!=32:
        raise ValueError('fixed64 confirmation denominator changed')
    if any(r['split']!='validation' or r['official_split']!='test' for r in rows):
        raise ValueError('not fixed official-test cohort')
    population=Path(args.population)
    assert (population/'COMPLETE').read_text().strip()=='completed=17790 failed=0 total=17790'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(settings=vars(args),input_sha256=INPUT_SHA,code_sha256=digest(__file__),
                  complete=False,planned_ids=[str(r['id']) for r in rows],completed_ids=[],
                  provenance='preexisting Llama3 observer replay, NOT original generators; no new model forwards; no annotations',
                  parent_settings_sha256=digest(population/'settings.json'),parent_manifests={})
    (out/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'manifest.json',manifest)
    for r in rows:
        root=population/'responses'/str(r['id'])
        m=json.loads((root/'manifest.json').read_text())
        for name,h in m['files'].items():
            if digest(root/name)!=h:raise ValueError('parent artifact hash')
        parent=json.loads((root/'record.json').read_text())
        for key in ['id','source_id','task','prompt','response','response_sha256','prompt_length']:
            if parent[key]!=r[key]:raise ValueError(f'parent identity mismatch:{key}')
        with np.load(root/'tokens.npz',allow_pickle=False) as tokens:
            if not np.array_equal(tokens['token_ids'],r['token_ids']) or not np.array_equal(tokens['offsets'],r['offsets']):
                raise ValueError('parent input token/offset mismatch')
        with np.load(root/'metrics.npz',allow_pickle=False) as z:
            scores={'population_native_entropy':z['base__entropy'].tolist(),
                    'population_native_nll':(-z['base__saved_logp']).tolist(),
                    'population_negative_margin':(-z['base__margin']).tolist(),
                    'population_source_small_js':z['source_01__js'].tolist(),
                    'population_history_small_js':z['history_01__js'].tolist(),
                    'population_source_support':(-z['source_01__saved_logp_change']).tolist(),
                    'population_history_support':(-z['history_01__saved_logp_change']).tolist(),
                    'population_history_minus_source_support':(z['source_01__saved_logp_change']-z['history_01__saved_logp_change']).tolist(),
                    'population_source_permute_js':z['source_permute__js'].tolist(),
                    'population_mlp_small_js':z['mlp_01__js'].tolist()}
        n=len(r['offsets'])
        if any(len(v)!=n or not np.isfinite(v).all() for v in scores.values()):raise ValueError('invalid baseline')
        record={k:r[k] for k in ['id','source_id','task','split','offsets','response_sha256','prompt_sha256']}
        record.update(token_ids=r['token_ids'][r['prompt_length']:],scores=scores)
        write_json(out/f"response_{r['id']}.json",record)
        manifest['completed_ids'].append(str(r['id']))
        manifest['parent_manifests'][str(r['id'])]=digest(root/'manifest.json')
    manifest.update(complete=True,model_forwards=0,output_sha256={p.name:digest(p) for p in sorted(out.glob('response_*.json'))})
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'complete':True,'responses':len(rows),'phase':args.phase,'model_forwards':0}))


if __name__=='__main__':main()
