"""One-command capture -> entry/continuation fitting -> frozen evaluation."""
import argparse
from pathlib import Path

from .local_capture import capture
from .two_stage import train_and_evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('all','capture','fit'),default='all')
    p.add_argument('--population',default='../reanchor/outputs/ragtruth_population_20260912')
    p.add_argument('--annotations',default='/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl')
    p.add_argument('--features',default='outputs/s11_local_attention')
    p.add_argument('--output',default='outputs/s11_two_stage')
    p.add_argument('--tasks',nargs='+',choices=('all','QA','Summary','Data2txt'),default=['QA'])
    p.add_argument('--generator',default='llama-2-7b-chat')
    p.add_argument('--model',default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dtype',choices=('bfloat16','float16','float32'),default='bfloat16')
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--max-channels',type=int,default=64,help='train-only unlabelled variance selection; 0 keeps all')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--resume',action='store_true',help='reuse matching complete response captures')
    p.add_argument('--no-oracle',action='store_true',help='skip separately labelled diagnostic after frozen evaluation')
    args=p.parse_args(argv)
    if args.window<1 or args.bootstrap<0 or args.max_channels<0 or args.max_tokens<1:
        p.error('invalid window, bootstrap or channel limit')
    if 'all' in args.tasks and args.tasks!=['all']:p.error('--tasks all cannot be combined with task names')
    if args.phase in ('all','fit') and Path(args.output).exists():
        raise FileExistsError('results already exist: use --phase capture to resume capture, or choose a new --output')
    if args.phase in ('all','capture'):
        capture(args.population,args.features,tasks=tuple(args.tasks),generator=args.generator,
                model_path=args.model,device=args.device,window=args.window,max_tokens=args.max_tokens,
                dtype=args.dtype,resume=args.resume)
    if args.phase in ('all','fit'):
        train_and_evaluate(args.features,args.annotations,args.output,max_channels=args.max_channels,
                           bootstrap=args.bootstrap,oracle=not args.no_oracle)


if __name__=='__main__':main()
