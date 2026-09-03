from pathlib import Path

path = Path(".bootstrap/apply_shortcut_route_refactor.py")
text = path.read_text(encoding="utf-8")
old = '''anchor = "## Pipeline\\n"
if text.count(anchor) != 1:
    raise RuntimeError("README pipeline anchor changed")
'''
new = '''anchor = "## Raw controls and evaluation boundary\\n"
if text.count(anchor) != 1:
    raise RuntimeError("README raw-controls anchor changed")
'''
if text.count(old) != 1:
    raise RuntimeError("shortcut refactor README-anchor block changed")
text = text.replace(old, new)
old_write = '''Path(path).write_text(text.replace(anchor, insert + anchor), encoding="utf-8")

path = "experiments/attention_mechanism_audit/METHOD.md"
'''
new_write = '''text = text.replace(anchor, insert + anchor)
text = text.replace(
    "Schema 8 must be recaptured into\\n"
    "`outputs/<observer-model>/dual_register_state/{train,test}/`. Older capture\\n"
    "directories are preserved as historical artifacts and are not adapted or\\n"
    "deleted. New reports are written under\\n"
    "`outputs/<observer-model>/dual_register_v8/{qa,summary,data2txt}/`, so the\\n"
    "earlier task reports are not overwritten.",
    "Schema 9 must be recaptured into\\n"
    "`outputs/<observer-model>/shortcut_route_state/{train,test}/`. Older capture\\n"
    "directories are preserved as historical artifacts and are not adapted or\\n"
    "deleted. New reports are written under\\n"
    "`outputs/<observer-model>/shortcut_route_v9/{qa,summary,data2txt}/`, so the\\n"
    "earlier task reports are not overwritten.",
)
Path(path).write_text(text, encoding="utf-8")

path = "experiments/attention_mechanism_audit/METHOD.md"
'''
if text.count(old_write) != 1:
    raise RuntimeError("shortcut refactor README-write block changed")
path.write_text(text.replace(old_write, new_write), encoding="utf-8")
