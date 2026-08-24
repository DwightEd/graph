import torch

from experiments.causal_walk_audit.config import GrammarConfig
from experiments.causal_walk_audit.grammar import GrammarAccumulator


def _sequence(repeats: int = 20):
    pattern = [0, 0, 1]
    values = pattern * repeats
    q = torch.zeros(len(values), 1, 2)
    q[torch.arange(len(values)), 0, torch.tensor(values)] = 1.0
    return q


def test_order_two_backoff_improves_non_markov_sequence():
    accumulator = GrammarAccumulator(
        1,
        2,
        config=GrammarConfig(alpha=0.1, backoff_tau=1.0),
    )
    for _ in range(8):
        accumulator.update(_sequence())
    grammar = accumulator.freeze()
    q = _sequence(10)
    surprise, order1, _, weight = grammar.score(q)
    assert float((order1[2:] - surprise[2:]).mean()) > 0.05
    assert float(weight[2:].mean()) > 0.5
