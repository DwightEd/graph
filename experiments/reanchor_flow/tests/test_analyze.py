import json

import pytest

from experiments.reanchor_flow.analyze import open_manifest
from experiments.reanchor_flow.capture import CAPTURE_SCHEMA


def test_old_capture_schema_is_not_silently_resumed(tmp_path):
    manifest = {
        "config": {"capture_schema": 1},
        "analysis_complete": True,
        "samples": [],
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="new --output"):
        open_manifest(tmp_path, {"capture_schema": CAPTURE_SCHEMA})
