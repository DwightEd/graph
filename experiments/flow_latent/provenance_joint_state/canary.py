"""Check replay message gradients against a full native 8B computation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection_data import write_json


def native_gradient(adapter, case, position, actual, rival, layer=15):
    leaves = []

    def start_gradient(module, inputs):
        leaf = inputs[0].detach().requires_grad_(True)
        leaves.append(leaf)
        return (leaf,)

    handle = adapter.layers[layer].self_attn.o_proj.register_forward_pre_hook(start_gradient)
    try:
        tokens = case['source']['prompt_with_source'] + case['response']['answer_ids']
        hidden = adapter.native.model(adapter.input_ids(tokens), use_cache=False).last_hidden_state[0, position]
        logits = adapter.native.lm_head(hidden)
        margin = logits[actual] - logits[rival]
        gradient = torch.autograd.grad(margin, leaves[0])[0][0, position]
        return gradient.reshape(32, 128).detach().cpu().numpy(), float(margin.detach())
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.measurement / 'inputs.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    checks = []
    for identity in ('15604', '12219'):
        case = next(c for c in inputs['cases'] if c['id'] == identity)
        with np.load(args.measurement / identity / 'observations.npz') as record:
            messages = np.load(args.measurement / identity / 'channel_messages.npy', mmap_mode='r')
            for row in (len(record['token_id']) // 2, len(record['token_id']) - 1):
                position = len(case['source']['prompt_with_source']) + row - 1
                gradient, margin = native_gradient(adapter, case, position,
                    int(record['token_id'][row]), int(record['rival'][row]))
                measured = np.einsum('chd,hd->ch', messages[row, 15].astype(float), gradient)
                expected = record['signed'][row, 15]
                # Saved full messages are float16; include that quantization error.
                relative = np.linalg.norm(measured - expected) / max(np.linalg.norm(expected), 1e-8)
                checks.append(dict(id=identity, token=row, relative_signed_error=float(relative),
                    margin_error=abs(margin - float(record['margin'][row]))))
                assert relative < .005
                assert checks[-1]['margin_error'] < 2e-4
    write_json(args.output, dict(status='PASS', native_full_gradients=len(checks),
        raw_message_precision='float16; native gradient FP32', checks=checks,
        peak_memory=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
