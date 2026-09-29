"""Fresh-output, label-free capture, followed by frozen calibration and evaluation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'capture', 'evaluate'), required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        from .prepare import prepare
        rows = prepare(args.output)
        print('prepared', len(rows), sum(len(row['response']['answer_ids']) for row in rows))
    elif args.stage == 'capture':
        from .capture import capture
        capture(args.output)
    else:
        from .evaluate import evaluate
        evaluate(args.output)


if __name__ == '__main__':
    main()
