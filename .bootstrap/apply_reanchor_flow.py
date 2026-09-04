from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import shutil
import tarfile

root = Path(__file__).resolve().parent
payload = "".join(
    path.read_text(encoding="ascii")
    for path in sorted(root.glob("payload_*.txt"))
)
archive_bytes = base64.b64decode(payload, validate=True)
destination = root.parent / "experiments" / "reanchor_flow"
shutil.rmtree(destination, ignore_errors=True)
destination.mkdir(parents=True)
with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
    archive.extractall(destination)
