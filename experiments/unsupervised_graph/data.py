"""Load the historical scalar packs without opening annotation arrays."""

import numpy as np
from state_audit.storage import read_json

INPUTS = ("context", "observations", "source_index", "answer_index", "target",
          "token_id", "unit_index", "development")


def load_inputs(packs, task, split):
    # np.load is lazy: never access the training archive's label arrays here.
    with np.load(packs / f"{task}_{split}.npz", allow_pickle=False) as saved:
        pack = {name: saved[name] for name in INPUTS}
    metadata = read_json(packs / f"{task}_{split}.json")
    generators = sorted({row["generator"] for row in metadata["records"]})
    pack["generator"] = np.empty(len(pack["target"]), dtype=np.int32)
    for row in metadata["records"]:
        region = slice(row["packed_start"], row["packed_stop"])
        pack["generator"][region] = generators.index(row["generator"])
    return pack, metadata
