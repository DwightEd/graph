"""Public representations and axes, excluding the batch axis (one sequence per call)."""

AXES = {
    "embedding": ("position", "feature"),
    "residual_before": ("position", "feature"),
    "attention_input": ("position", "feature"),
    "query": ("head", "position", "feature"),
    "key": ("head", "position", "feature"),
    "value": ("head", "position", "feature"),
    "attention": ("head", "position", "key"),
    "head_readout": ("position", "head", "feature"),
    "attention_write": ("position", "feature"),
    "residual_mid": ("position", "feature"),
    "mlp_input": ("position", "feature"),
    "mlp_activation": ("position", "feature"),
    "mlp_write": ("position", "feature"),
    "residual_after": ("position", "feature"),
    "final_hidden": ("position", "feature"),
}

GLOBAL_SITES = ("embedding", "final_hidden")
LAYER_SITES = tuple(name for name in AXES if name not in GLOBAL_SITES)
FULL_SEQUENCE_SITES = ("key", "value")
