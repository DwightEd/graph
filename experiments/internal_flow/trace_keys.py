"""Follow up frozen selected heads at source-key resolution, without interventions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.path_conflict.native import NativeRun
from .run import groups_for, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output/'manifest.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
    for probe in manifest['probes']:
        directory = args.output/probe['case']
        selected = json.loads((directory/'intervention_plan.json').read_text())[0]
        layer, head = selected['layer'], selected['head']
        ids = torch.tensor([probe['prefix_ids']], device='cuda:0')
        with torch.inference_mode(), NativeRun(model, ids.shape[1], groups_for(probe, ids.shape[1]),
                probe['candidate_ids'], trace_heads=[(layer, head)], trace_window=0) as run:
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        name = f'L{layer}H{head}'
        np.savez_compressed(directory/'selected_key_routes.npz', **run.routes)
        rows = []
        for position, token_id in enumerate(probe['prefix_ids']):
            rows.append(dict(layer=layer, head=head, position=position,
                token=tokenizer.decode([token_id]), is_source=position in probe['groups']['source'],
                is_evidence=position in probe['groups']['evidence'],
                attention=float(run.routes[name+'_attention'][0, position]),
                value_norm=float(run.routes[name+'_value_norm'][position]),
                projected_message_norm=float(run.routes[name+'_source_write_norm'][position]),
                local_lens_support=float(run.routes[name+'_source_lens_support'][position])))
        pd.DataFrame(rows).to_csv(directory/'selected_key_routes.csv', index=False)
        print(json.dumps(dict(case=probe['case'], layer=layer, head=head,
                              reconstruction_max=max(run.reconstruction))), flush=True)
    save_json(args.output/'key_trace_completed.json', dict(cases=len(manifest['probes']),
        forward_calls=len(manifest['probes']), selected_after_main_pilot=True, source_deletion=False))


if __name__ == '__main__':
    main()
