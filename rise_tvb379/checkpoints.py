"""Crash-safe persistence helpers for experiment results and checkpoints.

Checkpoint blocks are deliberately simple directories::

    checkpoints/<stage>/<block_key>/
        <frame_name>.csv
        metadata.json
        _complete.json

The completion marker is written last and contains hashes for every payload.
Readers therefore ignore both interrupted writes (no marker) and damaged or
stale blocks (a marker whose hashes no longer match).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol, TypeVar


BLOCK_SCHEMA_VERSION = 1
COMPLETE_MARKER_NAME = "_complete.json"
METADATA_NAME = "metadata.json"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FrameT = TypeVar("_FrameT")


class DataFrameLike(Protocol):
    """The small part of the pandas DataFrame interface used here."""

    def to_csv(self, path_or_buf: Any, **kwargs: Any) -> Any:
        """Serialize the frame to CSV."""


@dataclass(frozen=True)
class CompletedBlock:
    """A verified checkpoint block returned by :func:`read_completed_block`."""

    stage: str
    block_key: str
    frames: dict[str, Any]
    metadata: dict[str, Any]
    path: Path
    marker: dict[str, Any]


def _validate_component(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_', and '-' "
            "and must start with a letter or number"
        )
    if value in {".", ".."}:
        raise ValueError(f"{label} may not be {value!r}")
    return value


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync after an atomic replacement."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support fsync on directories.
        pass
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _replace_temporary(temporary: Path, destination: Path) -> Path:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return destination


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write text using a same-directory temporary file and ``os.replace``."""

    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int | None = 2,
) -> Path:
    """Atomically write deterministic, standards-compliant JSON."""

    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
    )
    return atomic_write_text(path, f"{serialized}\n")


def atomic_write_dataframe(
    path: str | os.PathLike[str],
    frame: DataFrameLike,
    *,
    index: bool = False,
    **csv_options: Any,
) -> Path:
    """Atomically serialize a pandas-compatible frame as CSV.

    ``pandas`` is intentionally not imported by this module. Any object with a
    compatible ``to_csv`` method works, while normal project use passes a
    pandas ``DataFrame``.
    """

    destination = Path(path)
    temporary = _temporary_path(destination)
    options: dict[str, Any] = {
        "index": index,
        "encoding": "utf-8",
        "lineterminator": "\n",
    }
    options.update(csv_options)
    try:
        frame.to_csv(temporary, **options)
        _replace_temporary(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


# A descriptive alias for result-table callers.
atomic_write_csv = atomic_write_dataframe


def sha256_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_path(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
    block_key: str,
) -> Path:
    safe_stage = _validate_component(stage, label="stage")
    safe_key = _validate_component(block_key, label="block_key")
    return Path(checkpoint_root) / safe_stage / safe_key


def _load_marker(block_path: Path) -> dict[str, Any] | None:
    marker_path = block_path / COMPLETE_MARKER_NAME
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verified_marker(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
    block_key: str,
) -> tuple[Path, dict[str, Any]] | None:
    block_path = _block_path(checkpoint_root, stage, block_key)
    marker = _load_marker(block_path)
    if marker is None:
        return None
    if marker.get("schema_version") != BLOCK_SCHEMA_VERSION:
        return None
    if marker.get("stage") != stage or marker.get("block_key") != block_key:
        return None

    metadata_record = marker.get("metadata")
    frame_records = marker.get("frames")
    if not isinstance(metadata_record, dict) or not isinstance(frame_records, dict):
        return None
    if metadata_record.get("file") != METADATA_NAME:
        return None
    for frame_name, frame_record in frame_records.items():
        try:
            safe_name = _validate_component(frame_name, label="frame name")
        except ValueError:
            return None
        if (
            not isinstance(frame_record, dict)
            or frame_record.get("file") != f"{safe_name}.csv"
        ):
            return None

    records: list[dict[str, Any]] = [metadata_record]
    records.extend(
        record for record in frame_records.values() if isinstance(record, dict)
    )
    if len(records) != len(frame_records) + 1:
        return None

    for record in records:
        filename = record.get("file")
        expected_hash = record.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            return None
        if Path(filename).name != filename or filename == COMPLETE_MARKER_NAME:
            return None
        payload_path = block_path / filename
        try:
            if not payload_path.is_file() or sha256_file(payload_path) != expected_hash:
                return None
            recorded_size = record.get("size_bytes")
            if (
                not isinstance(recorded_size, int)
                or recorded_size < 0
                or payload_path.stat().st_size != recorded_size
            ):
                return None
        except OSError:
            return None
    return block_path, marker


def write_completed_block(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
    block_key: str,
    frames: Mapping[str, DataFrameLike],
    metadata: Mapping[str, Any] | None = None,
    *,
    completed_at: str | None = None,
    overwrite: bool = False,
    csv_options: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a mapping of named frames and write its marker last.

    A verified existing block is immutable and is returned unchanged unless
    ``overwrite=True``. An incomplete block may be safely rewritten. Frame
    names become CSV filenames, so they use the same conservative component
    rules as stage and block names.
    """

    block_path = _block_path(checkpoint_root, stage, block_key)
    verified = _verified_marker(checkpoint_root, stage, block_key)
    marker_path = block_path / COMPLETE_MARKER_NAME
    if verified is not None and not overwrite:
        return marker_path

    block_path.mkdir(parents=True, exist_ok=True)
    # Removing an old marker first ensures readers never accept a block while
    # its payload is being replaced.
    marker_path.unlink(missing_ok=True)

    frame_records: dict[str, dict[str, Any]] = {}
    options = dict(csv_options or {})
    for frame_name in sorted(frames):
        safe_name = _validate_component(frame_name, label="frame name")
        frame_path = block_path / f"{safe_name}.csv"
        atomic_write_dataframe(frame_path, frames[frame_name], **options)
        frame_records[frame_name] = {
            "file": frame_path.name,
            "sha256": sha256_file(frame_path),
            "size_bytes": frame_path.stat().st_size,
        }

    metadata_path = block_path / METADATA_NAME
    atomic_write_json(metadata_path, dict(metadata or {}))
    marker: dict[str, Any] = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "stage": stage,
        "block_key": block_key,
        "frames": frame_records,
        "metadata": {
            "file": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
        },
    }
    if completed_at is not None:
        marker["completed_at"] = completed_at

    atomic_write_json(marker_path, marker)
    return marker_path


# This name reads naturally at orchestration call sites.
write_block_checkpoint = write_completed_block


def list_completed_blocks(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
) -> list[str]:
    """List verified block keys in deterministic lexical order."""

    safe_stage = _validate_component(stage, label="stage")
    stage_path = Path(checkpoint_root) / safe_stage
    try:
        candidates: Iterable[Path] = stage_path.iterdir()
    except OSError:
        return []

    completed: list[str] = []
    for candidate in sorted(candidates, key=lambda item: item.name):
        if not candidate.is_dir():
            continue
        try:
            _validate_component(candidate.name, label="block_key")
            verified = _verified_marker(
                checkpoint_root,
                safe_stage,
                candidate.name,
            )
        except (OSError, ValueError):
            continue
        if verified is not None:
            completed.append(candidate.name)
    return completed


list_completed_block_keys = list_completed_blocks


def read_completed_block(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
    block_key: str,
    *,
    dataframe_reader: Callable[[Path], _FrameT] | None = None,
) -> CompletedBlock:
    """Read a block only after validating its completion marker and hashes."""

    verified = _verified_marker(checkpoint_root, stage, block_key)
    if verified is None:
        raise RuntimeError(
            f"checkpoint block {stage}/{block_key} is incomplete or failed "
            "integrity validation"
        )
    block_path, marker = verified
    if dataframe_reader is None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency preflight
            raise RuntimeError(
                "pandas is required to read dataframe checkpoints"
            ) from exc
        dataframe_reader = pd.read_csv

    frames: dict[str, Any] = {}
    for frame_name, record in sorted(marker["frames"].items()):
        frames[frame_name] = dataframe_reader(block_path / record["file"])

    metadata_record = marker["metadata"]
    try:
        metadata_value = json.loads(
            (block_path / metadata_record["file"]).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"checkpoint metadata for {stage}/{block_key} is unreadable"
        ) from exc
    if not isinstance(metadata_value, dict):
        raise RuntimeError(
            f"checkpoint metadata for {stage}/{block_key} is not a JSON object"
        )

    return CompletedBlock(
        stage=stage,
        block_key=block_key,
        frames=frames,
        metadata=metadata_value,
        path=block_path,
        marker=marker,
    )


read_block_checkpoint = read_completed_block


def iter_completed_blocks(
    checkpoint_root: str | os.PathLike[str],
    stage: str,
    *,
    dataframe_reader: Callable[[Path], _FrameT] | None = None,
) -> Iterable[CompletedBlock]:
    """Yield verified blocks in the same deterministic order as ``list``."""

    for block_key in list_completed_blocks(checkpoint_root, stage):
        yield read_completed_block(
            checkpoint_root,
            stage,
            block_key,
            dataframe_reader=dataframe_reader,
        )
