"""Portable atomic writes and single-writer leases for derived studies."""

import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

_replace_locks: dict[Path, Lock] = {}
_replace_locks_guard = Lock()

CONTAINER_SCHEMA = 1
SOURCE_ALLOCATION = "source_allocation"
LOOKBACK_TRANSPORT = "lookback_transport"
COUNTERFACTUAL_MEDIATION = "counterfactual_mediation"


class UnsupportedArtifactSchema(ValueError):
    """The directory is not an explicitly identified message-DAG study."""


@dataclass(frozen=True)
class ArtifactDescriptor:
    method_id: str
    method_schema: str


def artifact_header(method_id: str, method_version: int) -> dict[str, object]:
    schema_names = {
        SOURCE_ALLOCATION: "source-allocation",
        LOOKBACK_TRANSPORT: "lookback-tangent",
        COUNTERFACTUAL_MEDIATION: "counterfactual-mediation",
    }
    if method_id not in schema_names:
        raise ValueError(f"unknown message-DAG method: {method_id}")
    return {
        "artifact_kind": "message_dag_study",
        "container_schema": CONTAINER_SCHEMA,
        "method_id": method_id,
        "method_schema": f"message-dag/{schema_names[method_id]}@{method_version}",
    }


def read_descriptor(output: Path) -> tuple[ArtifactDescriptor, dict]:
    index_path = Path(output) / "index.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "message_dag_study":
        raise UnsupportedArtifactSchema(
            f"{index_path}: missing explicit artifact_kind; use the original legacy entry point"
        )
    if manifest.get("container_schema") != CONTAINER_SCHEMA:
        raise UnsupportedArtifactSchema(
            f"{index_path}: unsupported container schema {manifest.get('container_schema')!r}"
        )
    method_id = manifest.get("method_id")
    method_schema = manifest.get("method_schema")
    if method_id not in (
        SOURCE_ALLOCATION,
        LOOKBACK_TRANSPORT,
        COUNTERFACTUAL_MEDIATION,
    ) or not isinstance(method_schema, str):
        raise UnsupportedArtifactSchema(f"{index_path}: invalid method identity")
    return ArtifactDescriptor(method_id, method_schema), manifest


def _replace_lock(path: Path) -> Lock:
    resolved = path.resolve()
    with _replace_locks_guard:
        return _replace_locks.setdefault(resolved, Lock())


class AtomicArtifact:
    """Own one unique temporary file until it is committed or discarded."""

    def __init__(self, destination: Path):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self.destination.parent,
            prefix=f".{self.destination.stem}.",
            suffix=self.destination.suffix,
            delete=False,
        ) as handle:
            self.temporary = Path(handle.name)

    def commit(self) -> None:
        with _replace_lock(self.destination):
            os.replace(self.temporary, self.destination)

    def discard(self) -> None:
        self.temporary.unlink(missing_ok=True)


@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    """Publish one complete file without exposing a partial checkpoint.

    Writers may prepare different temporary files concurrently. Replacement of
    the same destination is serialized because Windows does not reliably allow
    simultaneous ``ReplaceFile`` operations on one path.
    """

    artifact = AtomicArtifact(destination)
    try:
        yield artifact.temporary
        artifact.commit()
    finally:
        artifact.discard()


def _read_owner(handle) -> str:
    handle.seek(1)
    return handle.read().decode("utf-8", errors="replace").strip() or "owner not yet recorded"


def _acquire_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc
        return

    import fcntl

    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def output_writer(output: Path, lock_name: str = ".event_run.lock") -> Iterator[None]:
    """Hold a portable, non-blocking lease for one output directory."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / lock_name
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            _acquire_lock(handle)
        except BlockingIOError as exc:
            owner = _read_owner(handle)
            raise RuntimeError(
                f"output already has an active writer ({owner}): {output}; "
                "finish/stop that run or choose another --output"
            ) from exc
        handle.seek(1)
        handle.truncate()
        handle.write(f"host={socket.gethostname()} pid={os.getpid()}\n".encode())
        handle.flush()
        try:
            yield
        finally:
            _release_lock(handle)
