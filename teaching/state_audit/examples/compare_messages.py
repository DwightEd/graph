"""Standalone demo: compare two saved continuations and restore a downstream message."""

import argparse
import json
from pathlib import Path

from state_audit.analysis.interactions import adaptation_effects, factorial_effects
from state_audit.experiments.messages import MessageSite, measure_messages, restore_messages
from state_audit.pairing import load_answer
from state_audit.pipeline import load_run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--rival-sample", type=int, default=1)
    parser.add_argument("--target", type=int, default=3)
    args = parser.parse_args()
    left = load_answer(args.run / "samples" / f"{args.sample:06d}")
    right = load_answer(args.run / "samples" / f"{args.rival_sample:06d}")
    prefix = left["token_ids"][: left["prompt_length"] + args.target]
    other_prefix = right["token_ids"][: right["prompt_length"] + args.target]
    if prefix != other_prefix:
        raise ValueError("This example compares continuations at an identical saved prefix")
    candidates = [[answer["response_ids"][args.target]] for answer in (left, right)]
    model = load_run_model(args.run, None)
    sites = [
        MessageSite("early", 0, (0,), (len(prefix) - 1,)),
        MessageSite("late", 1, (0,), (len(prefix) - 1,)),
    ]

    def evaluate(operations):
        return measure_messages(model, prefix, candidates, sites, operations)

    baseline = evaluate([])
    cut_a, cut_b = evaluate([sites[0].deletion()]), evaluate([sites[1].deletion()])
    both = evaluate([site.deletion() for site in sites])
    retained = dict(zip(("11", "01", "10", "00"), (baseline, cut_a, cut_b, both)))
    restored = restore_messages(evaluate, [sites[1]], baseline[1], [sites[0].deletion()], cut_a)
    output = dict(
        purpose="software_demo_not_a_hallucination_result",
        factorial=factorial_effects(
            {key: value[0]["sum_margin"] for key, value in retained.items()}
        ),
        adaptation=adaptation_effects(
            *[value[0]["sum_margin"] for value in (baseline, cut_a, restored)]
        ),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
