from pathlib import Path

path = Path("experiments/attention_mechanism_audit/shortcut.py")
text = path.read_text(encoding="utf-8")
old = '''    "shortcut_route_candidate_mean": (
        "observed route incompleteness times positive autonomous residual alignment"
    ),
'''
new = '''    "shortcut_route_candidate_mean": (
        "observed route incompleteness times positive signed autonomous support"
    ),
'''
if text.count(old) != 1:
    raise RuntimeError("shortcut candidate definition changed")
path.write_text(text.replace(old, new), encoding="utf-8")
