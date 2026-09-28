"""Apply frozen uncertainty detector to the existing same-question natural pairs."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

from experiments.path_conflict.paired_inputs import claim_trace
from experiments.native_support.evidence_contrast.views import unit_intervals
from .features import uncertainty_features
from .run import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    args = parser.parse_args()
    cases = read_json(Path('experiments/path_conflict/paired_cases.json'))
    samples = [json.loads(line) for line in (args.samples/'samples.jsonl').read_text().splitlines()]
    lookup = {(str(row['source_id']), row['seed']): row for row in samples}
    model = joblib.load(args.run/'QA/models.joblib')['uncertainty']
    thresholds = read_json(args.run/'QA/selection.json')['thresholds']
    packs = Path(read_json(args.run/'protocol.json')['packs'])
    training = read_json(packs/'QA_train.json')['records']
    fit_sources = {r['source_id'] for r in training if r['partition'] == 'fit'}
    rows = []
    for case in cases:
        for side in ('supported', 'unsupported'):
            sample = lookup[case['source_id'], case[side]['seed']]
            trace, span = claim_trace(args.samples, sample, case[side], include_attention=False)
            # Saved original-generation entropy is in bits; full RAGTruth cache uses nats.
            state_path = args.states/Path(sample['trace']).name
            with np.load(state_path) as saved:
                entropy = saved['logit_entropy']*np.log(2.)
            surprisal = trace['log_normalizer']-trace['chosen_logit']
            units = np.zeros(len(entropy), dtype=int)
            for uid, unit in enumerate(unit_intervals(trace, 128)):
                units[unit['start']:unit['stop']] = uid
            prompt = int(trace['prompt_length'])
            pieces = trace['token_text'][prompt:].tolist()
            valid = np.array([bool(p) for p in pieces]) & ~trace['special_mask'][prompt:]
            positions = np.flatnonzero(valid)
            features = uncertainty_features(entropy[valid], surprisal[valid], units[valid], positions)
            scores = dict(entropy=features[:, 0], entropy_peak4=features[:, 6],
                entropy_peak8=features[:, 7], uncertainty=model.decision_function(features))
            region = (positions >= span[0]) & (positions < span[1])
            rows.append(dict(case=case['case_id'], side=side, claim=case[side]['target'],
                source_seen_in_detector_fit=case['source_id'] in fit_sources,
                annotation='reviewed local claim only; whole-answer correctness unknown', span=span,
                onset_entropy_bits=float(entropy[span[0]]/np.log(2.)),
                methods={name: dict(threshold=thresholds[name],
                    alarms=int((value[region] > thresholds[name]).sum()), tokens=int(region.sum()),
                    onset_score=float(value[np.searchsorted(positions, span[0])])) for name, value in scores.items()}))
    write_json(args.run/'natural_pairs.json', dict(rows=rows, natural_answers_used_for_fit=False,
        source_exposure_warning='headwear source 14315 occurs in detector fit; its transfer score is not source-held-out',
        transfer='RAGTruth QA observer uncertainty readout applied to original Llama3.1 natural samples'))
    print(rows, flush=True)


if __name__ == '__main__':
    main()
