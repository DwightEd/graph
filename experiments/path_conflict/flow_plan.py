"""Functional head roles discovered from the model, not preselected head IDs."""

from dataclasses import dataclass

DISCOVERY_GROUPS = ("all_context", "evidence", "wrong_source", "history")
SCAN_GROUPS = DISCOVERY_GROUPS + ("condition", "value")


@dataclass(frozen=True)
class HeadTrial:
    layer: int
    head: int
    source_group: str

    @property
    def name(self):
        return f"L{self.layer}H{self.head}_{self.source_group}"


def select_heads(writes, top_k=3, max_heads=12):
    """Select by baseline local effect only; no intervention result is used."""
    mask = writes.source_group.isin(DISCOVERY_GROUPS) & writes["head"].ge(0)
    frame = writes[mask].copy()
    selected = set()
    for _, rows in frame.groupby(["side", "panel", "source_group"], sort=False):
        ranked = rows.assign(size=rows.local_lens_support.abs()).nlargest(top_k, "size")
        selected.update(zip(ranked.layer.astype(int), ranked["head"].astype(int)))
    strength = frame.groupby(["layer", "head"]).local_lens_support.apply(lambda x: x.abs().max())
    ordered = sorted(selected, key=lambda key: -strength.get(key, 0.0))
    return ordered[:max_heads]


def sign_role(value, tolerance=1e-6):
    if value != value:
        return "not_tested"
    if value > tolerance:
        return "supports_correct"
    if value < -tolerance:
        return "supports_wrong"
    return "neutral"


def opposition(left, right):
    if left * right >= 0:
        return 0.0
    return 2 * min(abs(left), abs(right)) / (abs(left) + abs(right))
