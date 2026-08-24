import json

import torch

from experiments.causal_walk_audit.anchors import anchors_for_sample, load_anchor_manifest


def test_manifest_and_uniform_anchors(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            {
                "sample": [
                    {"name": "evidence", "kind": "evidence", "start": 0, "end": 3},
                    {"name": "question", "kind": "question", "start": 3, "end": 5},
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = load_anchor_manifest(path)
    anchors = anchors_for_sample(
        "sample",
        5,
        manifest=manifest,
        max_anchors=4,
        chunk_tokens=2,
        device=torch.device("cpu"),
    )
    assert anchors.names == ("evidence", "question")
    assert anchors.token_anchor.tolist() == [0, 0, 0, 1, 1]

    fallback = anchors_for_sample(
        "missing",
        5,
        manifest=manifest,
        max_anchors=4,
        chunk_tokens=2,
        device=torch.device("cpu"),
    )
    assert fallback.mode == "uniform_chunks"
    assert fallback.count == 3


def test_anchor_permutation_changes_prompt_assignment():
    anchors = anchors_for_sample(
        "missing",
        8,
        manifest={},
        max_anchors=4,
        chunk_tokens=2,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(5)
    shuffled = anchors.permuted(generator)
    assert shuffled.names == anchors.names
    assert sorted(shuffled.token_anchor.tolist()) == sorted(anchors.token_anchor.tolist())
    assert shuffled.token_anchor.tolist() != anchors.token_anchor.tolist()
