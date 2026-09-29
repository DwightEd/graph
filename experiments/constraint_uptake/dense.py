"""Check whether sparse within-step observations hid GSM8K failure signals."""
import argparse
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.run import load_model
from .capture import capture_record
from .score import record_scores
from .reweight import record as reweight_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=args.resume)
    weighted = args.output/'weighted'
    weighted.mkdir(exist_ok=args.resume)
    manifest = read_json(args.previous/'manifest.json')
    manifest['dense_previous'] = str(args.previous.resolve())
    for row in manifest['records']:
        if row['dataset']=='gsm8k':
            row['positions'] = list(range(len(row['answer'])))
    write_json(args.output/'manifest.json',manifest)
    write_json(weighted/'manifest.json',manifest)
    model = load_model(manifest['model'])
    results = []
    started = perf_counter()
    for row in manifest['records']:
        if row['dataset']=='ragtruth':
            target = weighted/row['key']
            target.mkdir(exist_ok=args.resume)
            for name in ('scores.npz','candidate_scores.npz'):
                shutil.copyfile(args.previous/row['key']/name,target/name)
            continue
        saved_path = args.output/row['key']/'readout.npz'
        if args.resume and saved_path.exists():
            saved = np.load(saved_path)
            results.append(dict(key=row['key'],queries=len(saved['positions']),
                max_margin_replay_error=float(saved['replay_error']),reused_capture=True))
        else:
            results.append(capture_record(model,row,args.output,manifest['query_batch']))
        if args.resume and (weighted/row['key']/'scores.npz').exists():
            continue
        record_scores(row,args.output)
        reweight_record(row,args.output,weighted)
        write_json(args.output/'progress.json',dict(completed=results,seconds=perf_counter()-started))
        print('dense',row['key'],len(row['positions']),round(perf_counter()-started,1),flush=True)
    write_json(args.output/'capture_complete.json',dict(status='complete',new_records=results,
        seconds=perf_counter()-started,ragtruth='six prior score arrays copied unchanged'))
    frozen = read_json(args.previous/'scores_frozen.json')
    frozen['scope'] = 'same v2 fixed formulas, all GSM8K response positions; no score or threshold tuning'
    write_json(weighted/'scores_frozen.json',frozen)


if __name__=='__main__':
    main()
