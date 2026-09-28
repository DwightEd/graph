"""Descriptive internal-state summaries; no AUROC or mechanism significance claim."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(directory, probe):
    baseline = json.loads((directory/'baselines.json').read_text())
    sequence = json.loads((directory/'sequence_baseline.json').read_text())
    effects = pd.read_csv(directory/'interventions.csv').set_index('name')
    chosen = effects.loc['evidence_050']
    heads = pd.read_csv(directory/'before_heads.csv')
    head = heads[(heads.layer == chosen.layer) & (heads['head'] == chosen['head'])
                 & (heads.source_group == 'evidence')].iloc[0]
    contrasts = pd.read_csv(directory/'state_contrasts.csv')
    final = contrasts[(contrasts.layer == 31) & (contrasts.site == 'mlp')].set_index('comparison')
    return dict(case=probe['case'], source_id=probe['source_id'],
        prefix_tokens=len(probe['prefix_ids']), correct_tokens=len(probe['variants']['correct'])-len(probe['prefix_ids']),
        wrong_tokens=len(probe['variants']['wrong'])-len(probe['prefix_ids']),
        before_first_margin=baseline['before']['margin'], sequence_margin=sequence['correct']-sequence['wrong'],
        layer=int(chosen.layer), head=int(chosen['head']), evidence_attention=head.attention_mass,
        evidence_local_projection=head.local_linear_support,
        reroute_sequence_delta=chosen.sequence_margin_change,
        random_sequence_delta=effects.loc['equal_norm_random'].sequence_margin_change,
        sham_error=abs(effects.loc['sham'].sequence_margin_change),
        correct_wrong_cosine=final.loc['correct_vs_wrong'].cosine,
        correct_equivalent_cosine=final.loc['correct_vs_equivalent'].cosine,
        reconstruction_max=max(row['reconstruction_max'] for row in baseline.values()))


def plot_results(root, frame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name in ('wifi', 'duration', 'intimate'):
        table = pd.read_csv(root/name/'before_lens.csv')
        table = table[table.site == 'after_mlp']
        axes[0].plot(table.layer, table.margin, label=name)
    axes[0].axhline(0, color='grey', linewidth=.6)
    axes[0].set(xlabel='Layer (zero based)', ylabel='Fixed candidate logit-lens margin',
                title='Descriptive residual readout before branching')
    axes[0].legend(frameon=False)
    positions = np.arange(len(frame))
    axes[1].barh(positions-.17, frame.reroute_sequence_delta, height=.32, label='Evidence reroute, dose .5')
    axes[1].barh(positions+.17, frame.random_sequence_delta, height=.32, label='Equal-norm random direction')
    axes[1].set_yticks(positions, frame['case'])
    axes[1].axvline(0, color='grey', linewidth=.6)
    axes[1].set(xlabel='Change in full candidate log-probability margin', title='Native downstream intervention')
    axes[1].legend(frameon=False, fontsize=7)
    keys = pd.read_csv(root/'duration/selected_key_routes.csv')
    keys = keys[keys.is_evidence]
    axes[2].bar(keys.token.str.strip(), keys.attention, color='#26828e')
    axes[2].set(ylabel='Native attention probability', title='Duration: L22H13 source-key reading')
    fig.text(.5, .01, '7 exposed local contrasts / 5 sources; observer replay; selected heads and candidates are oracle-assisted. No detection result.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(root/'internal_flow.png', dpi=180)
    fig.savefig(root/'internal_flow.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output/'manifest.json').read_text())
    completed = json.loads((args.output/'completed.json').read_text())
    key_completed = json.loads((args.output/'key_trace_completed.json').read_text())
    frame = pd.DataFrame([summarize(args.output/probe['case'], probe) for probe in manifest['probes']])
    frame.to_csv(args.output/'summary.csv', index=False)
    checks = dict(cases=len(frame), sources=int(frame.source_id.nunique()),
        first_candidate_correct_preferred=int((frame.before_first_margin > 0).sum()),
        full_candidate_correct_preferred=int((frame.sequence_margin > 0).sum()),
        positive_reroute_margin_change=int((frame.reroute_sequence_delta > 0).sum()),
        all_measured_sham_error_max=float(frame.sham_error.max()),
        native_reconstruction_error_max=float(frame.reconstruction_max.max()),
        correct_wrong_closer_than_correct_equivalent=int((frame.correct_wrong_cosine > frame.correct_equivalent_cosine).sum()),
        actual_main_forward_calls=len(frame)*completed['model_forward_calls_per_case'],
        followup_key_forward_calls=key_completed['forward_calls'], new_auroc=None)
    (args.output/'summary.json').write_text(json.dumps(checks, indent=2)+'\n')
    plot_results(args.output, frame)
    print(frame[['case','before_first_margin','sequence_margin','layer','head','evidence_attention','reroute_sequence_delta']].to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
