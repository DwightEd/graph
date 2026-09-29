"""Fill previously missing source/routing observations; reuse official caches."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from state_audit.model import load_model
from state_audit.attribution import _projection_gram
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.native_support.ragtruth_benchmark.capture import collect_with_source
from experiments.native_support.evidence_contrast.capture import collect_condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    torch.set_num_threads(4)
    model, grams = None, None
    started = perf_counter()
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        if row['cached']:
            (directory/'observations.npz').symlink_to((Path(row['cached'])/'observations.npz').resolve())
            continue
        if model is None:
            model,_ = load_model(manifest['model'],device='cuda:0',dtype='bfloat16')
            grams = {i:_projection_gram(model,i) for i in range(len(model.layers))}
        present, heads = collect_with_source(model,row['source'],row['response'],grams,256,32)
        absent = collect_condition(model,row['source']['prompt_without_source'],
            row['response']['answer_ids'],row['response']['units'],256,32,row['key'])
        observed = dict(token_id=present['token_id'],
            source_local=absent['local'].astype(float)-present['local'],
            source_full=absent['full'].astype(float)-present['full'],
            **{name:present[name] for name in ('raw_route','raw_attention','entropy')})
        np.savez_compressed(directory/'observations.npz',**observed)
        np.savez_compressed(directory/'routing_heads.npz',**heads)
        np.savez_compressed(directory/'likelihoods.npz',with_full=present['full'],with_local=present['local'],
            without_full=absent['full'],without_local=absent['local'])
        print('captured',row['key'],'seconds',round(perf_counter()-started,1),flush=True)
    write_json(args.output/'capture_complete.json',dict(status='complete',answers=len(manifest['records']),
        new_answers=sum(r['cached'] is None for r in manifest['records']),seconds=perf_counter()-started,
        dtype='bfloat16 original source-first adapter; uptake edges use separate FP32 observer cache',labels_used=False))


if __name__=='__main__':
    main()
