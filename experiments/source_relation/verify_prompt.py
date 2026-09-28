"""Reconstruct original first-answer attention from independently captured Q/K."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    capture = read_json(args.output/'capture_complete.json')
    errors = {}
    for source,record in capture['verification'].items():
        directory = args.output/'sources'/source
        q = np.load(directory/'query.npy',mmap_mode='r')
        k = np.load(directory/'key.npy',mmap_mode='r')
        original = np.load(Path(manifest['base'])/record['reference']/'attention.npy',mmap_mode='r')
        length = q.shape[2]
        layer_errors = []
        for layer in range(q.shape[0]):
            query = q[layer,:,-1]
            key = np.repeat(k[layer],q.shape[1]//k.shape[1],axis=0)
            logits = np.einsum('hd,hkd->hk',query,key)/query.shape[-1]**.5
            weight = np.exp(logits-logits.max(-1,keepdims=True))
            weight /= weight.sum(-1,keepdims=True)
            expected = original[layer,:,0,:length]
            layer_errors.append(float(np.linalg.norm(weight-expected)/np.linalg.norm(expected)))
        assert max(layer_errors)<.0005,(source,max(layer_errors))
        errors[source] = layer_errors
    result = dict(status='passed',sources=len(errors),query='last prompt input predicts first answer token',
        max_relative_error=max(max(value) for value in errors.values()),layer_errors=errors)
    write_json(args.output/'prompt_attention_verification.json',result)
    print('prompt Q/K attention replay',result['sources'],result['max_relative_error'])


if __name__ == '__main__':
    main()
