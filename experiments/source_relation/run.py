"""Run five local rounds; preserve completed stages and previous raw caches."""
import argparse
from pathlib import Path
import subprocess
import sys


def stage(module,output,marker,extra=()):
    if (output/marker).exists():
        print('Preserving completed stage:',module,output,flush=True)
        return
    subprocess.run([sys.executable,'-m','experiments.'+module,'--output',str(output),*extra],check=True)


def evaluate(output):
    for module,marker in [('message_js.evaluate','evaluation.json'),
                           ('source_relation.verify','verification.json'),
                           ('message_js.score_verify','score_verification.json'),
                           ('head_state_readout.audit','TOKEN_AUDIT.html'),
                           ('source_relation.audit','mechanism_summary.json')]:
        stage(module,output,marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix',default='outputs/source_relation_new')
    args = parser.parse_args()
    first,second,third,fourth,fifth = [Path(args.output_prefix+'_v'+str(i)) for i in (1,2,3,4,5)]
    stage('source_relation.capture',first,'capture_complete.json')
    stage('source_relation.verify_prompt',first,'prompt_attention_verification.json')
    stage('source_relation.measure',first,'features_complete.json')
    stage('source_relation.score',first,'scores_frozen.json')
    evaluate(first)
    stage('source_relation.addresses',first,'address_token_ledger.json')
    stage('source_relation.refine',second,'scores_frozen.json',('--first',str(first)))
    evaluate(second)
    stage('source_relation.kernel',third,'features_complete.json',('--first',str(first)))
    stage('source_relation.score',third,'scores_frozen.json',('--previous',str(second)))
    evaluate(third)
    stage('source_relation.measure',fourth,'features_complete.json',
          ('--source-cache',str(first),'--direction','forward'))
    stage('source_relation.score',fourth,'scores_frozen.json',('--previous',str(second)))
    evaluate(fourth)
    stage('source_relation.combine',fifth,'scores_frozen.json',('--first',str(first),'--forward',str(fourth)))
    evaluate(fifth)
    if not (fifth/'report_complete.json').exists():
        subprocess.run([sys.executable,'-m','experiments.source_relation.report',
                        '--output-prefix',args.output_prefix],check=True)


if __name__ == '__main__':
    main()
