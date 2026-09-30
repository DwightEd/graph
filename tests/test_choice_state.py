"""Scientific invariants for the conditional choice and history path model."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teaching/state_audit/src"))
from experiments.native_support.choice_cache import validate_row


def row_for(roots, tail=0):
    roots = np.asarray(roots, dtype=float)
    if roots.ndim == 1:
        roots = roots[:, None]
    target = len(roots) - 3
    groups = np.r_[0, 2, 3, np.ones(target, dtype=int)]
    positive = np.zeros((4, roots.shape[1]))
    negative = np.zeros_like(positive)
    np.add.at(positive, groups, np.maximum(roots, 0))
    np.add.at(negative, groups, np.maximum(-roots, 0))
    logits = np.r_[0., -roots.sum(0)]
    return dict(root_token_choice=roots, group_ids=groups, root_positive=positive,
                root_negative=negative, candidate_logits=logits,
                candidate_ids=np.r_[10 + target, np.arange(20, 20 + roots.shape[1])],
                candidate_tail_mass=np.asarray(tail), margin=roots.sum(0),
                ledger_error=np.zeros(roots.shape[1]), entropy=np.asarray(1.),
                surprisal=np.asarray(-np.log(softmax(logits)[0] * (1 - tail))),
                target=np.asarray(target), query=np.asarray(2 + target), token_id=np.asarray(10 + target))


class ChoiceStateTests(unittest.TestCase):


    def test_boundary_rejects_off_by_one_and_inconsistent_provenance(self):
        response = dict(id="answer", prompt_length=3, token_ids=[1, 2, 3, 10, 11])
        row = row_for([0, 0, 0, 1])
        validate_row(row, response, 1, row["group_ids"], 1)
        row["query"] += 1
        with self.assertRaisesRegex(ValueError, "prediction alignment"):
            validate_row(row, response, 1, row["group_ids"], 1)
        row["query"] -= 1
        row["root_positive"][1, 0] += 1
        with self.assertRaisesRegex(ValueError, "token/group provenance"):
            validate_row(row, response, 1, row["group_ids"], 1)


if __name__ == "__main__":
    unittest.main()
