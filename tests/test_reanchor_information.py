import numpy as np

from experiments.unsupervised_token_graph.reanchor import InformationDiagnostics, ReanchorAnalyzer


def test_reanchor_excludes_tokens_without_history():
    attention = np.zeros((1, 1, 8, 12), dtype=float)
    prompt = 4
    for t in range(8):
        q = prompt - 1 + t
        attention[0, 0, t, :q + 1] = 1 / (q + 1)
    result = ReanchorAnalyzer(window=2).run(attention, prompt)
    assert not result.valid_history[0]
    assert not result.event[0]
    assert np.isfinite(result.reanchor_ratio).all()


def test_information_diagnostics_separates_shifted_distributions():
    result = InformationDiagnostics.histogram_kl([.9, .95, .8], [.1, .2, .15])
    assert result["kl"] > 0
    assert result["js"] > 0
    assert result["overlap"] < 1


def test_reanchor_can_select_layers_and_heads():
    attention = np.zeros((2, 2, 5, 7), dtype=float)
    attention[0, 0, :, :5] = 1 / 5
    attention[1, 1, :, :5] = 1 / 5
    all_heads = ReanchorAnalyzer(layers=[0], heads=[0]).run(attention, 3)
    assert all_heads.waad.shape == (5,)


def test_reanchor_types_are_defined_in_the_canonical_module():
    from experiments.unsupervised_token_graph.reanchor import ReanchorResult

    assert ReanchorAnalyzer.__name__ == "ReanchorAnalyzer"
    assert ReanchorResult.__name__ == "ReanchorResult"
    assert ReanchorAnalyzer.__module__ == "experiments.unsupervised_token_graph.reanchor"
    assert "reanchor_ratio" in ReanchorResult.__dataclass_fields__
