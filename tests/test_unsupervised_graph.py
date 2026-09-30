"""Packed fixed-baseline input never reads annotation arrays."""

import numpy as np
import json

from experiments.unsupervised_graph.data import INPUTS, load_inputs


def test_input_loader_never_accesses_label_members(tmp_path, monkeypatch):
    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, name):
            assert name in INPUTS, f"Unexpected training input access: {name}"
            return np.zeros(3)

    monkeypatch.setattr(np, "load", lambda *args, **kwargs: Archive())
    metadata = dict(records=[dict(generator="model", packed_start=0, packed_stop=3)])
    (tmp_path / "QA_train.json").write_text(json.dumps(metadata))
    pack, _ = load_inputs(tmp_path, "QA", "train")
    assert not {"labels", "onsets", "firsts", "baselines"} & set(pack)
