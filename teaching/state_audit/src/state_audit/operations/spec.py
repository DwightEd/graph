"""Small JSON adapter for operation plans; arrays may be inline or in .npy files."""

from pathlib import Path

import numpy as np

from . import Delete, Inject, Replace, Steer, Target


def load_operation(row: dict, base: Path):
    arguments = dict(row)
    kind = arguments.pop("operation")
    target = Target(**arguments.pop("target"))
    if "value_file" in arguments:
        value = np.load(base / arguments.pop("value_file"), allow_pickle=False)
        field = "direction" if kind == "steer" else "value"
        arguments[field] = value
    constructors = {"delete": Delete, "replace": Replace, "inject": Inject, "steer": Steer}
    return constructors[kind](target, **arguments)
