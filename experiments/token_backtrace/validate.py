"""Check native target alignment and embedding-gate derivatives on real 8B inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL


@torch.no_grad()
def native_target(model, ids, token_id, root=None, change=0.):
    embeddings = model.model.embed_tokens(torch.tensor([ids], device=model.device))
    if root is not None:
        embeddings[:, root] *= 1 + change
    hidden = model.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state[0, -1]
    return float(model.lm_head(hidden).float().log_softmax(-1)[token_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    records = read_json(args.trace / 'manifest.json')['records']
    model = load_model(MODEL)
    checks = []
    for row in records:
        positions = (96, 100) if row['key'] == '15604' else (22, 30)
        with np.load(args.trace / row['key'] / 'trace.npz') as saved:
            for target in positions:
                ids = row['prompt'] + row['response']['answer_ids'][:target]
                token_id = row['response']['answer_ids'][target]
                original = native_target(model, ids, token_id)
                effect = saved['root_effect'][target]
                root = int(np.argmax(np.abs(effect)))
                for epsilon in (.001, .005):
                    positive = native_target(model, ids, token_id, root, epsilon)
                    negative = native_target(model, ids, token_id, root, -epsilon)
                    finite = (positive - negative) / (2 * epsilon)
                    checks.append(dict(key=row['key'], target=target, root=root, epsilon=epsilon,
                        analytic=float(effect[root]), finite=finite,
                        slope_error=abs(finite - float(effect[root])),
                        prefix_error=abs(original - float(saved['logp'][target]))))
    write_json(args.trace / 'numerical_checks.json', dict(checks=checks, forward_calls=20,
        description='exact prefix recomputation vs full causal pass; symmetric embedding gate'))
    assert all(row['prefix_error'] < .001 for row in checks)
    assert all(row['slope_error'] < .02 + .05 * abs(row['analytic']) for row in checks)
    print('8 finite differences and 4 target/prefix checks passed', flush=True)


if __name__ == '__main__':
    main()
