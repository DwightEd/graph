"""Scientific invariants for the conditional choice and history path model."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teaching/state_audit/src"))
from experiments.native_support.choice_cache import validate_row
from experiments.native_support.choice_state import choice_factors, read_state, source_deficit, summarize_states
from experiments.native_support.choice_state_run import BASELINES, run


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
    def test_native_conditional_odds_and_factor_product(self):
        row = row_for([[-2., 1.], [1., 3.], [.5, -1.]])
        factors = choice_factors(row)
        product = factors["group_choice_factor"].prod(0)
        assert_allclose(product / product.sum(), softmax(row["candidate_logits"]))
        assert_allclose(factors["choice_reconstruction_error"], 0, atol=1e-15)
        # A group factor is deliberately not the LM probability conditioned on that source alone.
        self.assertFalse(np.allclose(factors["group_choice_factor"][0], softmax(row["candidate_logits"])))

    def test_positive_history_carries_and_source_support_refreshes(self):
        first = read_state(row_for([-1, 0, 0]), 1, 3, np.empty((0, 5, 2)))[1]
        carried = read_state(row_for([0, 0, 0, 1]), 1, 3, first[None])[1]
        refreshed = read_state(row_for([1, 0, 0, 0, 1]), 1, 3, np.stack([first, carried]))[1]
        assert_allclose(first[0], [0, 1])
        assert_allclose(carried, first)
        assert_allclose(refreshed[0], [.5, .5])
        assert_allclose(refreshed.sum(), 1)

    def test_history_opposition_does_not_flip_unrelated_contrasts(self):
        previous = read_state(row_for([-1, 0, 0]), 1, 3, np.empty((0, 5, 2)))[1]
        result = read_state(row_for([0, 0, 0, -1]), 1, 3, previous[None])
        assert_allclose(result[1][0], [0, 0])
        assert_allclose(result[1][1], [0, 1])
        assert_allclose(result[-1], [0])

    def test_tail_and_zero_attribution_are_unresolved(self):
        row = row_for([0, 0, 0], tail=.4)
        factors, state, *_ = read_state(row, 1, 3, np.empty((0, 5, 2)))
        assert_allclose(factors["unobserved_alternative_mass"], 4 / 7)
        assert_allclose(state[-1, 0], 1)
        assert_allclose(state.sum(), 1)
        self.assertTrue(np.isnan(source_deficit(state[None], 1)[0][0]))

    def test_tiny_negative_cached_tail_cannot_become_negative_conditional_mass(self):
        row = row_for([[13.25, 13.625, 14.], [0, 0, 0], [0, 0, 0]])
        row["candidate_tail_mass"] = np.asarray(-1.1920929e-7)
        row["surprisal"] = np.asarray(3.81469e-6)
        factors, state, *_ = read_state(row, 1, 3, np.empty((0, 5, 2)))
        self.assertGreater(factors["unobserved_alternative_mass"], 0)
        self.assertTrue((state >= 0).all())
        assert_allclose(state.sum(), 1)

    def test_source_deficit_accepts_inherited_support_and_separates_unknown(self):
        first = read_state(row_for([1, 0, 0]), 1, 3, np.empty((0, 5, 2)))[1]
        carried = read_state(row_for([0, 0, 0, 1], tail=.2), 1, 3, first[None])[1]
        score, lower, upper = source_deficit(carried[None], 1)
        assert_allclose(score, 0)
        assert_allclose(lower, 0)
        self.assertGreater(upper[0], 0)

    def test_candidate_permutation_does_not_match_ranks_across_time(self):
        previous = read_state(row_for([[-1, 2], [0, 0], [0, 0]]), 1, 3, np.empty((0, 5, 2)))[1]
        row = row_for([[1, -1], [0, 0], [0, 0], [3, 2]], tail=.1)
        permuted = {name: value.copy() for name, value in row.items()}
        for name in ("root_positive", "root_negative", "root_token_choice", "margin", "ledger_error"):
            permuted[name] = permuted[name][..., ::-1]
        for name in ("candidate_logits", "candidate_ids"):
            permuted[name] = permuted[name][[0, 2, 1]]
        original = read_state(row, 1, 3, previous[None])
        reordered = read_state(permuted, 1, 3, previous[None])
        assert_allclose(original[1], reordered[1])
        assert_allclose(original[-1], reordered[-1])

    def test_recursion_and_reverse_exposure_match_triangular_solutions(self):
        rows = [row_for([-1, 1, 0]), row_for([1, 0, 0, 2]), row_for([0, 1, 0, 1, 3])]
        states = np.zeros((3, 5, 2))
        local = np.zeros_like(states)
        history = np.zeros((3, 3))
        for target, row in enumerate(rows):
            result = read_state(row, 1, 3, states[:target])
            states[target], local[target], history[target, :target] = result[1], result[2], result[-1]
        inverse = np.linalg.inv(np.eye(3) - history)
        assert_allclose(states.reshape(3, -1), inverse @ local.reshape(3, -1))
        scores = summarize_states(states, local, history, 1)
        assert_allclose(scores["state_mass_error"], 0, atol=1e-15)
        expected = (inverse.T @ local[:, 0, 1]) / (inverse.T @ np.ones(3))
        assert_allclose(scores["descendant_opposition"], expected)
        # Forward prefix states do not depend on adding later rows; only the offline readout can.
        assert_allclose(states[0], read_state(rows[0], 1, 3, states[:0])[1])

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

    def test_directory_and_pack_score_identically_without_annotations_or_torch(self):
        response = dict(id="answer", source_id="source", prompt_length=3,
                        token_ids=[1, 2, 3, 10, 11], token_text=["a", "b", "c", "d", "e"])
        settings = dict(model="fixture", responses=[response])
        sources = dict(blocks=[[0]], group_ids=[0, 2, 3, 1, 1])
        files = {"settings.json": json.dumps(settings).encode(),
                 "value_transport/capture_settings.json": json.dumps(settings).encode(),
                 "value_transport/capture/0000/sources.json": json.dumps(sources).encode()}
        for target, row in enumerate([row_for([-1, 0, 0]), row_for([0, 0, 0, 1])]):
            row["attention"] = np.full((1, 2, target + 3), 1 / (target + 3))
            stream = io.BytesIO()
            np.savez(stream, **row)
            files[f"value_transport/capture/0000/token_{target:06d}.npz"] = stream.getvalue()
        baseline = {name: np.array([.1, .2]) for name in BASELINES}
        baseline.update(target=np.arange(2), query=np.array([2, 3]), token_id=np.array([10, 11]))
        stream = io.BytesIO()
        np.savez(stream, **baseline)
        files["value_transport/responses/0000/scores.npz"] = stream.getvalue()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ZipFile(root / "pack.zip", "w") as archive:
                for name, content in files.items():
                    path = root / "input" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    if name.endswith(".npz") and "/capture/" in name:
                        with np.load(io.BytesIO(content)) as saved:
                            arrays = {key: saved[key] for key in saved.files}
                        attention = arrays.pop("attention")
                        groups = arrays["group_ids"].copy()
                        groups[-1] = 4
                        arrays["head_group_attention"] = np.stack([
                            attention[..., groups == group].sum(-1) for group in range(5)
                        ], axis=-1)
                        stream = io.BytesIO()
                        np.savez(stream, **arrays)
                        content = stream.getvalue()
                    archive.writestr(name, content)
            for source, output in ((root / "input", root / "directory"), (root / "pack.zip", root / "packed")):
                summary = run(SimpleNamespace(input=source, output=output, annotations=None))
                self.assertEqual(summary["evaluation_status"], "unavailable")
            with np.load(root / "directory/responses/0000/state.npz") as left:
                with np.load(root / "packed/responses/0000/state.npz") as right:
                    for field in left.files:
                        assert_allclose(left[field], right[field])
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
