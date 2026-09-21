"""Run four operations on the local demo; results are software examples, not evidence."""

import argparse
from pathlib import Path

import numpy as np

from state_audit.experiments.interventions import compare_conditions
from state_audit.operations import Delete, Inject, Replace, Steer, Target
from state_audit.pipeline import load_run_model
from state_audit.state import ModelState
from state_audit.storage import read_json, write_json


def run(root: Path):
    model = load_run_model(root, "cpu")
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    state = ModelState.open(directory / "trace")
    target_token = 3
    query = answer["prompt_length"] + target_token - 1
    residual = Target("residual_after", layers=(0,), positions=(query,))
    heads = Target("head_readout", layers=(0,), positions=(query,), heads=(0, 1, 2))
    donor = state.at("residual_after", [query], layer=0)
    # A toy direction for exercising the API; no claim about its semantic meaning.
    direction = np.ones(donor.shape[-1])
    conditions = {
        "delete_three_heads": [Delete(heads)],
        "same_state_replace": [Replace(residual, donor)],
        "inject_residual": [Inject(residual, direction, scale=0.01)],
        "steer_residual": [Steer(residual, direction, amount=0.1)],
    }
    result = compare_conditions(model, answer, [target_token], conditions)
    write_json(root / "interventions" / "example.json", result)
    for name, scores in result["conditions"].items():
        print(name, scores["delta_logp"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args().run)
