"""Pre-registered HoloRoute structural ablations."""

from __future__ import annotations

from dataclasses import replace


def ablation_configs(config):
    model = config.model
    return {
        "full": config,
        "no_path": replace(
            config,
            model=replace(model, use_relay=False, use_holonomy=False),
        ),
        "no_depth": replace(
            config,
            model=replace(model, use_depth=False, use_holonomy=False),
        ),
        "no_query_set": replace(
            config,
            model=replace(model, use_query=False),
        ),
        "identity_transport": replace(
            config,
            model=replace(model, use_transport=False),
        ),
        "event_only": replace(
            config,
            model=replace(
                model,
                use_depth=False,
                use_relay=False,
                use_query=False,
                use_holonomy=False,
            ),
        ),
    }
