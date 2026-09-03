from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "experiments/attention_mechanism_audit/shortcut.py",
    '''    evidence_carrier = context(mean_attention, delta_value, history)
    evidence_gate = context(delta_attention, mean_value, history)
    autonomous_history = context(a_no_evidence, v_no_evidence, history) - context(
        a_no_both, v_no_both, history
    )

    rewired_carrier = torch.zeros_like(evidence_carrier)
''',
    '''    evidence_carrier = context(mean_attention, delta_value, history)
    evidence_gate = context(delta_attention, mean_value, history)
    no_evidence_history = context(a_no_evidence, v_no_evidence, history)
    evidence_conditioned_history = full_history - no_evidence_history
    relay_reconstruction = evidence_carrier + evidence_gate
    relay_closure_error = (
        evidence_conditioned_history - relay_reconstruction
    ).flatten(1).norm(dim=-1)
    relay_scale = torch.maximum(
        evidence_conditioned_history.flatten(1).norm(dim=-1),
        relay_reconstruction.flatten(1).norm(dim=-1),
    )
    if not torch.all(
        relay_closure_error <= 5e-5 + 5e-5 * relay_scale
    ):
        raise ValueError("evidence relay midpoint decomposition does not close")
    autonomous_history = no_evidence_history - context(
        a_no_both, v_no_both, history
    )

    rewired_carrier = torch.zeros_like(evidence_carrier)
''',
)
replace_once(
    "experiments/attention_mechanism_audit/shortcut.py",
    '''    return {
        "route_gram": route_gram,
        "head_gram": head_gram,
        "rewire_valid": rewire_valid,
    }
''',
    '''    return {
        "route_gram": route_gram,
        "head_gram": head_gram,
        "rewire_valid": rewire_valid,
        "relay_closure_error": relay_closure_error,
    }
''',
)

replace_once(
    "experiments/attention_mechanism_audit/capture.py",
    '''            "shortcut_rewire_valid": torch.zeros(
                layers, response_tokens, dtype=torch.bool
            ),
        }
''',
    '''            "shortcut_rewire_valid": torch.zeros(
                layers, response_tokens, dtype=torch.bool
            ),
            "shortcut_relay_closure_error": torch.zeros(
                layers, response_tokens, dtype=torch.float32
            ),
        }
''',
)

replace_once(
    "experiments/attention_mechanism_audit/tests/test_capture.py",
    '''    assert trace["shortcut_rewire_valid"].shape == (2, 4)
''',
    '''    assert trace["shortcut_rewire_valid"].shape == (2, 4)
    assert trace["shortcut_relay_closure_error"].shape == (2, 4)
    assert trace["shortcut_relay_closure_error"].max() < 1e-4
''',
)
replace_once(
    "experiments/attention_mechanism_audit/tests/test_collect.py",
    '''        "shortcut_rewire_valid": torch.zeros(
            layers, response_tokens, dtype=torch.bool
        ),
    }
''',
    '''        "shortcut_rewire_valid": torch.zeros(
            layers, response_tokens, dtype=torch.bool
        ),
        "shortcut_relay_closure_error": torch.zeros(layers, response_tokens),
    }
''',
)
replace_once(
    "experiments/attention_mechanism_audit/tests/test_evaluate.py",
    '''        "shortcut_rewire_valid": torch.ones(layers, tokens, dtype=torch.bool),
    }
''',
    '''        "shortcut_rewire_valid": torch.ones(layers, tokens, dtype=torch.bool),
        "shortcut_relay_closure_error": torch.zeros(layers, tokens),
    }
''',
)
replace_once(
    "experiments/attention_mechanism_audit/tests/test_shortcut.py",
    '''    assert geometry["rewire_valid"].item()
''',
    '''    assert geometry["rewire_valid"].item()
    assert geometry["relay_closure_error"].max() < 1e-6
''',
)

replace_once(
    "experiments/attention_mechanism_audit/METHOD.md",
    '''The direct evidence write and the exact history-root write for `noE - noEH`
complete the observed vector set. Capture stores their aggregate and per-head
Gram matrices. It also swaps adjacent response value endpoints before the two
''',
    '''The direct evidence write and the exact history-root write for `noE - noEH`
complete the observed vector set. Capture verifies the midpoint identity for
every layer and prediction event, then stores its closure error together with
the aggregate and per-head Gram matrices. It also swaps adjacent response value
endpoints before the two
''',
)
replace_once(
    "experiments/attention_mechanism_audit/AGENTS.md",
    '''- Capture, for every `(layer,target)`, the residual-space Gram of the full
  strict-history write, direct-evidence write, evidence-conditioned history
''',
    '''- Capture, for every `(layer,target)`, and fail if the exact midpoint relay
  decomposition does not close. Store the residual-space Gram of the full
  strict-history write, direct-evidence write, evidence-conditioned history
''',
)
