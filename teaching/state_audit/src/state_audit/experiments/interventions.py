"""Fixed-answer causal comparisons, independent of the chosen representations."""

from pathlib import Path

import numpy as np
import torch

from ..intervention import intervene
from ..operations.spec import load_operation
from ..pairing import load_answer
from ..storage import read_json, write_json


def score_targets(model, answer, targets, operations=()) -> dict:
    if any(target < 0 or target >= len(answer["response_ids"]) for target in targets):
        raise IndexError("targets must refer to saved response token indices")
    positions = [answer["prompt_length"] + target - 1 for target in targets]
    target_ids = [answer["response_ids"][target] for target in targets]
    with torch.inference_mode(), intervene(model, operations):
        hidden = model.forward(answer["token_ids"][:-1])[positions]
        logp, entropy = model.score(hidden, model.input_ids(target_ids)[0])
    return dict(target_logp=logp.cpu().tolist(), logit_entropy=entropy.cpu().tolist())


def compare_conditions(model, answer, targets, conditions) -> dict:
    baseline = score_targets(model, answer, targets)
    results = {}
    for name, operations in conditions.items():
        scores = score_targets(model, answer, targets, operations)
        delta = np.asarray(scores["target_logp"]) - baseline["target_logp"]
        results[name] = dict(**scores, delta_logp=delta.tolist())
    return dict(
        purpose="fixed_answer_intervention_not_detector_evaluation",
        targets=targets,
        baseline=baseline,
        conditions=results,
        effect_definition="intervened_minus_baseline_target_logp_nats",
    )


def factorial_interaction(model, answer, targets, group_a, group_b) -> dict:
    """Two groups, each containing arbitrarily many operations/heads/layers."""
    conditions = dict(a_only=group_a, b_only=group_b, both=[*group_a, *group_b])
    result = compare_conditions(model, answer, targets, conditions)
    full = np.asarray(result["baseline"]["target_logp"])
    logps = {name: np.asarray(row["target_logp"]) for name, row in result["conditions"].items()}
    effect_with_b = logps["both"] - logps["b_only"]
    effect_without_b = logps["a_only"] - full
    result["interaction"] = (effect_with_b - effect_without_b).tolist()
    return result


def run_plan(model, root: Path, sample: int, plan_path: Path, output: Path):
    plan = read_json(plan_path)
    conditions = {}
    for name, rows in plan["conditions"].items():
        conditions[name] = [load_operation(row, plan_path.parent) for row in rows]
    answer = load_answer(root / "samples" / f"{sample:06d}")
    result = compare_conditions(model, answer, plan["targets"], conditions)
    result.update(sample=sample, plan=plan)
    write_json(output, result)
    return result
