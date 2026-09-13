"""Bind finite decisions to FrozenReader's already-published cache records."""

import json
from pathlib import Path

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import FrozenReader, digest


class ReceiptReader(FrozenReader):
    def _record(self, saved, *, cache_hit=False):
        path = self.cache / (digest(saved["request"]) + ".json")
        if json.loads(path.read_text()) != saved:
            raise ValueError("reader cache differs before returning finite prediction")
        self.last_record = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        return super()._record(saved, cache_hit=cache_hit)


def verify_receipt(receipt, identity, instruction, payload, labels, prediction):
    path = Path(receipt["path"])
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError("finite reader cache bytes changed")
    saved = json.loads(path.read_text())
    request = saved["request"]
    if path.name != digest(request) + ".json":
        raise ValueError("finite cache record name differs from its full request identity")
    if (request["model"] != identity or request["instruction"] != instruction
            or request["payload"] != payload or request["labels"] != list(labels)
            or request["max_new_tokens"] != 1 or saved["prediction"] != prediction):
        raise ValueError("finite cache record differs from frozen reader/request/prediction")
