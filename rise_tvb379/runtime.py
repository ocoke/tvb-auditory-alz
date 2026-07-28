"""Run-directory, provenance, status, and resume compatibility utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterator

from .checkpoints import atomic_write_json, atomic_write_text, sha256_file


MANIFEST_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
STATUS_FILENAME = "run_status.json"
CONFIG_FILENAME = "resolved_config.json"
ENVIRONMENT_FILENAME = "environment.json"
INPUTS_FILENAME = "inputs.json"
LOG_FILENAME = "run.log"

DEFAULT_RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "tvb-library",
    "tvb-data",
)

RUN_SUBDIRECTORIES = ("inputs", "checkpoints", "results")
RUN_STATES = frozenset({"created", "running", "interrupted", "failed", "completed"})
_STATUS_TRANSITIONS = {
    "created": frozenset({"running", "failed", "interrupted"}),
    "running": frozenset({"running", "interrupted", "failed", "completed"}),
    "interrupted": frozenset({"running", "failed"}),
    "failed": frozenset({"running"}),
    "completed": frozenset(),
}
_SAFE_MODE = re.compile(r"[^A-Za-z0-9_-]+")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp (kept injectable in public APIs)."""

    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime | None = None) -> str:
    """Render a timestamp in stable ISO-8601 UTC form."""

    instant = value or utc_now()
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_for_directory(value: datetime | None = None) -> str:
    instant = value or utc_now()
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonicalize(value: Any) -> Any:
    """Convert common configuration objects into deterministic JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return isoformat_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(
                    f"JSON mapping key {key!r} has unsupported type "
                    f"{type(key).__name__}"
                )
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError(
                    f"mapping contains duplicate canonical key {normalized_key!r}"
                )
            normalized[normalized_key] = canonicalize(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(value.item):
        # NumPy scalar support without importing NumPy during preflight.
        return canonicalize(value.item())
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json(value: Any) -> str:
    """Return compact canonical JSON suitable for hashing."""

    return json.dumps(
        canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint_json(value: Any) -> str:
    """Compute a deterministic SHA-256 digest for JSON-compatible state."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint_config(resolved_config: Any) -> str:
    """Fingerprint the fully resolved experiment configuration."""

    return fingerprint_json(resolved_config)


compute_config_fingerprint = fingerprint_config


def _code_files(
    project_root: Path,
    paths: Sequence[str | os.PathLike[str]] | None,
) -> list[Path]:
    roots = (
        [project_root / "main.py", project_root / "rise_tvb379"]
        if paths is None
        else [
            (
                candidate
                if (candidate := Path(item)).is_absolute()
                else project_root / candidate
            )
            for item in paths
        ]
    )
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.add(root.resolve())
        elif root.is_dir():
            for candidate in root.rglob("*.py"):
                if "__pycache__" not in candidate.parts and candidate.is_file():
                    files.add(candidate.resolve())
    return sorted(
        files,
        key=lambda candidate: candidate.relative_to(project_root).as_posix(),
    )


def fingerprint_code(
    project_root: str | os.PathLike[str],
    *,
    paths: Sequence[str | os.PathLike[str]] | None = None,
) -> str:
    """Fingerprint Python source paths and bytes in deterministic order."""

    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for source_path in _code_files(root, paths):
        try:
            relative_path = source_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"code path {source_path} is outside project root {root}"
            ) from exc
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


compute_code_fingerprint = fingerprint_code


def capture_package_versions(
    packages: Iterable[str] = DEFAULT_RUNTIME_PACKAGES,
) -> dict[str, str | None]:
    """Capture installed distribution versions without importing packages."""

    versions: dict[str, str | None] = {}
    for package_name in sorted(set(packages)):
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def capture_environment(
    packages: Iterable[str] = DEFAULT_RUNTIME_PACKAGES,
) -> dict[str, Any]:
    """Capture exact interpreter, host, and dependency provenance."""

    return {
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": capture_package_versions(packages),
    }


def environment_compatibility_view(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Select only fields that the resume contract requires to remain exact."""

    packages = environment.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("environment is missing a packages mapping")
    return {
        "python_version": environment.get("python_version"),
        "python_implementation": environment.get("python_implementation"),
        "packages": dict(packages),
    }


def fingerprint_environment(environment: Mapping[str, Any] | None = None) -> str:
    """Fingerprint Python and dependency versions used for resume checks."""

    captured = capture_environment() if environment is None else environment
    return fingerprint_json(environment_compatibility_view(captured))


compute_environment_fingerprint = fingerprint_environment


def capture_input_manifest(
    inputs: Mapping[str, str | os.PathLike[str] | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve, hash, and size named input files.

    Mapping-valued inputs must contain ``path`` and may contain an expected
    ``sha256``. Additional source/provenance fields are preserved.
    """

    captured: dict[str, dict[str, Any]] = {}
    for input_name in sorted(inputs):
        raw_record = inputs[input_name]
        if isinstance(raw_record, Mapping):
            if "path" not in raw_record:
                raise ValueError(f"input {input_name!r} is missing path")
            record = dict(canonicalize(raw_record))
            input_path = Path(os.fspath(raw_record["path"])).resolve()
            expected_hash = raw_record.get("sha256")
        else:
            record = {}
            input_path = Path(raw_record).resolve()
            expected_hash = None
        if not input_path.is_file():
            raise FileNotFoundError(f"input {input_name!r} not found: {input_path}")
        actual_hash = sha256_file(input_path)
        if expected_hash is not None and str(expected_hash).lower() != actual_hash:
            raise ValueError(
                f"input fingerprint mismatch for {input_name!r}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        record.update(
            {
                "path": str(input_path),
                "sha256": actual_hash,
                "size_bytes": input_path.stat().st_size,
            }
        )
        captured[str(input_name)] = record
    return captured


def fingerprint_inputs(
    inputs: Mapping[str, str | os.PathLike[str] | Mapping[str, Any]],
) -> str:
    """Fingerprint named input content, independent of machine-local paths."""

    if all(
        isinstance(record, Mapping) and "sha256" in record for record in inputs.values()
    ):
        manifest = inputs
    else:
        manifest = capture_input_manifest(inputs)
    content_identity: dict[str, dict[str, Any]] = {}
    for input_name in sorted(manifest):
        record = manifest[input_name]
        if not isinstance(record, Mapping) or "sha256" not in record:
            raise ValueError(f"input {input_name!r} is missing sha256")
        content_identity[str(input_name)] = {
            "sha256": str(record["sha256"]).lower(),
            "size_bytes": record.get("size_bytes"),
        }
    return fingerprint_json(content_identity)


compute_input_fingerprint = fingerprint_inputs


def create_run_directory(
    output_root: str | os.PathLike[str],
    mode: str,
    config_digest: str | None = None,
    *,
    now: datetime | None = None,
    subdirectories: Sequence[str] = RUN_SUBDIRECTORIES,
    max_collisions: int = 10_000,
) -> Path:
    """Atomically reserve a unique timestamped run directory."""

    clean_mode = _SAFE_MODE.sub("-", mode.strip()).strip("-_")
    if not clean_mode:
        raise ValueError("mode must contain at least one letter or number")
    for subdirectory in subdirectories:
        if Path(subdirectory).name != subdirectory or subdirectory in {".", ".."}:
            raise ValueError(f"unsafe run subdirectory name: {subdirectory!r}")
    digest = config_digest or fingerprint_config({"mode": mode})
    digest_label = re.sub(r"[^0-9a-fA-F]", "", digest)[:8].lower()
    if len(digest_label) < 8:
        digest_label = fingerprint_json(digest)[:8]

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    basename = f"{_timestamp_for_directory(now)}_{clean_mode}_{digest_label}"
    for collision_index in range(max_collisions):
        suffix = "" if collision_index == 0 else f"_{collision_index:02d}"
        run_dir = root / f"{basename}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        for subdirectory in subdirectories:
            (run_dir / subdirectory).mkdir()
        atomic_write_text(run_dir / LOG_FILENAME, "")
        return run_dir
    raise RuntimeError(
        f"could not allocate a unique run directory under {root} "
        f"after {max_collisions} attempts"
    )


create_unique_run_directory = create_run_directory


def _normalize_error(error: BaseException | str | None) -> dict[str, str] | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "RuntimeError", "message": str(error)}


def load_run_status(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and minimally validate ``run_status.json``."""

    path = Path(run_dir).resolve() / STATUS_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run status not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run status is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("status") not in RUN_STATES:
        raise RuntimeError(f"run status is invalid: {path}")
    return value


def initialize_run_status(
    run_dir: str | os.PathLike[str],
    *,
    mode: str | None = None,
    status: str = "running",
    now: datetime | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the initial atomic status record for a run."""

    if status not in {"created", "running"}:
        raise ValueError("initial status must be 'created' or 'running'")
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_path}")
    status_path = run_path / STATUS_FILENAME
    if status_path.exists() and not overwrite:
        raise FileExistsError(f"run status already exists: {status_path}")
    timestamp = isoformat_utc(now)
    record: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "status": status,
        "mode": mode,
        "created_at": timestamp,
        "started_at": timestamp if status == "running" else None,
        "updated_at": timestamp,
        "exit_time": None,
        "completed_at": None,
        "error": None,
        "attempt": 1 if status == "running" else 0,
        "history": [{"status": status, "at": timestamp}],
    }
    atomic_write_json(status_path, record)
    return record


def update_run_status(
    run_dir: str | os.PathLike[str],
    new_status: str,
    *,
    error: BaseException | str | None = None,
    now: datetime | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Transition status atomically, rejecting illegal lifecycle changes."""

    if new_status not in RUN_STATES:
        raise ValueError(f"unknown run status {new_status!r}")
    run_path = Path(run_dir).resolve()
    record = load_run_status(run_path)
    previous_status = record["status"]
    if new_status not in _STATUS_TRANSITIONS[previous_status]:
        raise ValueError(
            f"invalid run status transition: {previous_status} -> {new_status}"
        )
    timestamp = isoformat_utc(now)
    record["status"] = new_status
    record["updated_at"] = timestamp
    record["error"] = _normalize_error(error)
    if details:
        record["details"] = canonicalize(details)

    if new_status == "running":
        record["started_at"] = timestamp
        record["exit_time"] = None
        record["attempt"] = int(record.get("attempt", 0)) + 1
    elif new_status in {"interrupted", "failed"}:
        record["exit_time"] = timestamp
    elif new_status == "completed":
        record["exit_time"] = timestamp
        record["completed_at"] = timestamp

    history = record.setdefault("history", [])
    event: dict[str, Any] = {"status": new_status, "at": timestamp}
    normalized_error = _normalize_error(error)
    if normalized_error is not None:
        event["error"] = normalized_error
    history.append(event)
    atomic_write_json(run_path / STATUS_FILENAME, record)
    return record


def mark_run_interrupted(
    run_dir: str | os.PathLike[str],
    error: BaseException | str | None = None,
) -> dict[str, Any]:
    return update_run_status(run_dir, "interrupted", error=error)


def mark_run_failed(
    run_dir: str | os.PathLike[str],
    error: BaseException | str,
) -> dict[str, Any]:
    return update_run_status(run_dir, "failed", error=error)


def mark_run_completed(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    return update_run_status(run_dir, "completed")


@contextmanager
def run_status_lifecycle(
    run_dir: str | os.PathLike[str],
    *,
    mode: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Maintain running/interrupted/failed/completed status around a run."""

    status_path = Path(run_dir).resolve() / STATUS_FILENAME
    if status_path.exists():
        status = update_run_status(run_dir, "running")
    else:
        status = initialize_run_status(run_dir, mode=mode)
    try:
        yield status
    except KeyboardInterrupt as exc:
        update_run_status(run_dir, "interrupted", error=exc)
        raise
    except BaseException as exc:
        update_run_status(run_dir, "failed", error=exc)
        raise
    else:
        update_run_status(run_dir, "completed")


def write_run_manifest(
    run_dir: str | os.PathLike[str],
    *,
    mode: str,
    resolved_config: Any,
    project_root: str | os.PathLike[str],
    inputs: Mapping[str, str | os.PathLike[str] | Mapping[str, Any]],
    environment: Mapping[str, Any] | None = None,
    code_digest: str | None = None,
    config_digest: str | None = None,
    environment_digest: str | None = None,
    input_digest: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Write resolved provenance files and the aggregate run manifest."""

    run_path = Path(run_dir).resolve()
    project_path = Path(project_root).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_path}")

    normalized_config = canonicalize(resolved_config)
    normalized_environment = canonicalize(
        environment if environment is not None else capture_environment()
    )
    normalized_inputs = capture_input_manifest(inputs)
    fingerprints = {
        "code": code_digest or fingerprint_code(project_path),
        "config": config_digest or fingerprint_config(normalized_config),
        "environment": environment_digest
        or fingerprint_environment(normalized_environment),
        "inputs": input_digest or fingerprint_inputs(normalized_inputs),
    }

    config_path = run_path / CONFIG_FILENAME
    environment_path = run_path / ENVIRONMENT_FILENAME
    inputs_path = run_path / INPUTS_FILENAME
    manifest_path = run_path / MANIFEST_FILENAME
    atomic_write_json(config_path, normalized_config)
    atomic_write_json(environment_path, normalized_environment)
    atomic_write_json(inputs_path, normalized_inputs)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "created_at": isoformat_utc(created_at),
        "fingerprints": fingerprints,
        "paths": {
            "run_dir": str(run_path),
            "project_root": str(project_path),
            "manifest": str(manifest_path),
            "config": str(config_path),
            "environment": str(environment_path),
            "inputs": str(inputs_path),
            "status": str(run_path / STATUS_FILENAME),
            "log": str(run_path / LOG_FILENAME),
            "input_directory": str(run_path / "inputs"),
            "checkpoint_directory": str(run_path / "checkpoints"),
            "results_directory": str(run_path / "results"),
            "input_files": {
                name: record["path"] for name, record in normalized_inputs.items()
            },
        },
        "resolved_config": normalized_config,
        "environment": normalized_environment,
        "inputs": normalized_inputs,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


initialize_run_manifest = write_run_manifest


def load_run_manifest(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the top-level shape of ``run_manifest.json``."""

    path = Path(run_dir).resolve() / MANIFEST_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run manifest not found: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"run manifest is not a JSON object: {path}")
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported run manifest schema: {path}")
    if not isinstance(value.get("fingerprints"), dict):
        raise RuntimeError(f"run manifest fingerprints are missing: {path}")
    if not isinstance(value.get("paths"), dict):
        raise RuntimeError(f"run manifest paths are missing: {path}")
    return value


def _read_json_object(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {source}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {source}")
    return value


def validate_resume(
    run_dir: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    resolved_config: Any | None = None,
    inputs: Mapping[str, str | os.PathLike[str] | Mapping[str, Any]] | None = None,
    environment: Mapping[str, Any] | None = None,
    code_digest: str | None = None,
    config_digest: str | None = None,
    environment_digest: str | None = None,
    input_digest: str | None = None,
) -> dict[str, Any]:
    """Validate an explicit resume request and return its saved manifest.

    When current values are omitted, this function recomputes them from the
    exact project/config/input paths recorded in the manifest and captures the
    currently active Python/dependency environment.
    """

    run_path = Path(run_dir).resolve()
    status = load_run_status(run_path)
    if status["status"] == "completed":
        raise RuntimeError("resume refused: run is already complete")

    manifest = load_run_manifest(run_path)
    paths = manifest["paths"]
    recorded_run_path = Path(paths.get("run_dir", "")).resolve()
    if recorded_run_path != run_path:
        raise RuntimeError(
            "resume refused: run path mismatch "
            f"(expected {recorded_run_path}, got {run_path})"
        )

    expected = manifest["fingerprints"]
    missing = {"code", "config", "environment", "inputs"} - set(expected)
    if missing:
        raise RuntimeError(
            "resume refused: manifest is missing fingerprints "
            + ", ".join(sorted(missing))
        )

    current_project_root = Path(
        project_root if project_root is not None else paths.get("project_root", "")
    ).resolve()
    if code_digest is None:
        code_digest = fingerprint_code(current_project_root)

    if config_digest is None:
        if resolved_config is None:
            resolved_config = _read_json_object(
                paths.get("config", ""),
                label="saved resolved configuration",
            )
        config_digest = fingerprint_config(resolved_config)

    if environment_digest is None:
        if environment is None:
            saved_environment = manifest.get("environment")
            saved_packages = (
                saved_environment.get("packages", {})
                if isinstance(saved_environment, Mapping)
                else {}
            )
            environment = capture_environment(saved_packages.keys())
        environment_digest = fingerprint_environment(environment)

    if input_digest is None:
        if inputs is None:
            saved_inputs = manifest.get("inputs")
            if not isinstance(saved_inputs, Mapping):
                raise RuntimeError("resume refused: saved inputs are missing")
            inputs = saved_inputs
        try:
            current_inputs = capture_input_manifest(inputs)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"resume refused: input fingerprint mismatch ({exc})"
            ) from exc
        input_digest = fingerprint_inputs(current_inputs)

    current = {
        "code": code_digest,
        "config": config_digest,
        "environment": environment_digest,
        "inputs": input_digest,
    }
    for fingerprint_name in ("code", "config", "environment", "inputs"):
        if current[fingerprint_name] != expected[fingerprint_name]:
            raise RuntimeError(
                f"resume refused: {fingerprint_name} fingerprint mismatch "
                f"(expected {expected[fingerprint_name]}, "
                f"got {current[fingerprint_name]})"
            )
    return manifest


validate_resume_compatibility = validate_resume
