"""Construct same-slot semantic controls; these are synthetic facts, not RAGTruth outcomes."""

import argparse
from pathlib import Path

from state_audit.storage import write_json


def templates():
    return [
        dict(
            name="onion",
            task="stage_binding",
            question="How long should the onions cook after removing the sausages?",
            prefix="The onions should cook for",
            candidates=[" 12 minutes.", " 18 minutes."],
            base=[
                "Before removing the sausages, cook the onions for 12 minutes.",
                "After removing the sausages, cook the onions for 18 minutes.",
            ],
            applicability_swap=[
                "After removing the sausages, cook the onions for 12 minutes.",
                "Before removing the sausages, cook the onions for 18 minutes.",
            ],
            value_swap=[
                "Before removing the sausages, cook the onions for 18 minutes.",
                "After removing the sausages, cook the onions for 12 minutes.",
            ],
            paraphrase=[
                "Prior to removing the sausages, cook the onions for 12 minutes.",
                "Once the sausages have been removed, cook the onions for 18 minutes.",
            ],
        ),
        dict(
            name="headwear",
            task="scope_binding",
            question="What do the other adults wear?",
            prefix="The other adults wear a",
            candidates=[" gold cap.", " cloth cap."],
            base=["Only the ruler wears a gold cap.", "The other adults wear a cloth cap."],
            applicability_swap=[
                "The other adults wear a gold cap.",
                "Only the ruler wears a cloth cap.",
            ],
            value_swap=["Only the ruler wears a cloth cap.", "The other adults wear a gold cap."],
            paraphrase=[
                "A gold cap is reserved for the ruler.",
                "Adults other than the ruler wear a cloth cap.",
            ],
        ),
    ]


def build_cases():
    cases = []
    for template in templates():
        for variant in ("base", "applicability_swap", "value_swap", "paraphrase"):
            facts = template[variant]
            prompt = (
                "Use only these records.\n"
                + "\n".join(facts)
                + "\nQuestion: "
                + template["question"]
            )
            name = template["name"] + ":" + variant
            row = dict(
                id=name,
                source_id=template["name"],
                dataset="synthetic_semantic_controls",
                task=template["task"],
                response_prefix=template["prefix"],
                example=dict(id=name, source_id=template["name"], prompt=prompt),
                candidates=template["candidates"],
                preferred=1 if variant in ("base", "paraphrase") else 0,
                source_quotes=dict(evidence=facts, value_source=[facts[0]]),
                metadata=dict(
                    variant=variant,
                    annotation="program_constructed_support",
                    history_status="fixed_response_prefix",
                    candidate_order="fixed_by_value",
                ),
            )
            if variant in ("base", "applicability_swap"):
                other = "applicability_swap" if variant == "base" else "base"
                row.update(donor=template["name"] + ":" + other, donor_kind="applicability_swap")
            cases.append(row)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, dict(model=args.model, template="chat", cases=build_cases()))


if __name__ == "__main__":
    main()
