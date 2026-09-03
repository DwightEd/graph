from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "experiments/attention_mechanism_audit/evaluate.py",
    '''    for offset, name in enumerate(SHORTCUT_SCORE_NAMES):
        validity_name = f"{name}__valid"
        valid = np.asarray(arrays[validity_name], dtype=bool)
        valid &= np.isfinite(arrays[name])
''',
    '''    for offset, name in enumerate(SHORTCUT_SCORE_NAMES):
        validity_name = f"{name}__valid"
        if name not in arrays or validity_name not in arrays:
            result[name] = {
                "valid_tokens": 0,
                "valid_sources": 0,
                "auroc": None,
                "average_precision": None,
                "ap_lift": None,
                "auroc_ci95": [None, None],
                "average_precision_ci95": [None, None],
                "unavailable_reason": "shortcut route arrays are not present",
            }
            continue
        valid = np.asarray(arrays[validity_name], dtype=bool)
        valid &= np.isfinite(arrays[name])
''',
)
replace_once(
    "experiments/attention_mechanism_audit/tests/test_evaluate.py",
    '''    assert audit["shortcut_route_candidate_mean__valid"].all()
''',
    '''    # This fixture's history is completely explained by the relay span, so
    # no residual exists on which an autonomous alignment could be defined.
    assert not audit["shortcut_route_candidate_mean__valid"].any()
''',
)
