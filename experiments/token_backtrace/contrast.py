"""Exposed, manually specified source-relation diagnostics; never detector rules."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL


DIAGNOSTICS = {
    '15604': dict(target=100, source_start=240, source_stop=246,
        expected=' private pension income is not taxed', wrong=' taxed', alternative=' not',
        donors=dict(original=' private pension income is not taxed',
            flip=' private pension income is taxed',
            equivalent1=' private pension income is untaxed',
            equivalent2=' private pension income is exempt from taxation')),
    '219': dict(target=22, source_start=41, source_stop=46,
        expected='For more than four days', wrong=' four', alternative=' more',
        donors=dict(original='For more than four days', flip='For exactly four days',
            equivalent1='For over four days', equivalent2='For longer than four days')),
}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    write_json(args.output / 'protocol.json', dict(cases=DIAGNOSTICS,
        manually_specified=True, labels_used_in_case_design=True,
        objective='fixed wrong-token minus alternative-token logit margin within each case',
        limits='source text edits change tokenization/positions; not internal node patches; next token is not a whole semantic claim'))
    records = read_json('outputs/token_backtrace_20260930_pilot/manifest.json')['records']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = load_model(MODEL)
    results = []
    for row in records:
        spec = DIAGNOSTICS[row['key']]
        start, stop, target = spec['source_start'], spec['source_stop'], spec['target']
        assert tokenizer.decode(row['prompt'][start:stop]) == spec['expected']
        wrong = tokenizer.encode(spec['wrong'], add_special_tokens=False)
        alternative = tokenizer.encode(spec['alternative'], add_special_tokens=False)
        assert len(wrong) == len(alternative) == 1
        assert wrong[0] == row['response']['answer_ids'][target]
        for name, text in spec['donors'].items():
            donor = tokenizer.encode(text, add_special_tokens=False)
            if name == 'original':
                assert donor == row['prompt'][start:stop]
            prefix = row['prompt'][:start] + donor + row['prompt'][stop:]
            ids = prefix + row['response']['answer_ids'][:target]
            hidden = model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False).last_hidden_state[0, -1]
            logits = model.lm_head(hidden).float()
            logp = logits.log_softmax(-1)
            result = dict(key=row['key'], target=target, donor=name, prompt_length=len(prefix),
                margin=float(logits[wrong[0]] - logits[alternative[0]]),
                wrong_logp=float(logp[wrong[0]]), alternative_logp=float(logp[alternative[0]]))
            results.append(result)
            print(result, flush=True)
    write_json(args.output / 'results.json', dict(results=results, forward_calls=len(results)))


if __name__ == '__main__':
    main()
