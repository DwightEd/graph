"""Run both predeclared pilot iterations and the matched-strength null control."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def invoke(module,*arguments):
    command = [sys.executable,'-m','experiments.anchored_flow.'+module,*map(str,arguments)]
    print('RUN', ' '.join(command),flush=True)
    subprocess.run(command,check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ['PYTHONPATH'] = os.pathsep.join((str(root),str(root/'teaching/state_audit/src')))
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    first = Path(str(args.output_prefix)+'_v1')
    second = Path(str(args.output_prefix)+'_v2')
    control = Path(str(args.output_prefix)+'_control')
    for stage in ('prepare','capture','edges','score','evaluate','audit','report'):
        invoke(stage,'--output',first)
    invoke('revise','--previous',first,'--output',second)
    for stage in ('evaluate','audit','report'):
        invoke(stage,'--output',second)
    invoke('controls','--previous',second,'--output',control)
    for stage in ('evaluate','audit','report'):
        invoke(stage,'--output',control)


if __name__=='__main__':
    main()
