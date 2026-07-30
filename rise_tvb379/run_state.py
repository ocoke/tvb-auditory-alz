"""Crash-safe run status, progress logging, and job-level checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
import traceback
from typing import Any
import uuid


STATE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
STATUS_FILENAME = "run_status.json"
PROGRESS_LOG_FILENAME = "progress.log"
CHECKPOINT_ROOT_NAME = ".science_ready_checkpoints"
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "joblib",
    "tvb-library",
    "tvb-data",
)
_SAFE_STAGE = re.compile(r"[^A-Za-z0-9_.-]+")


class RunStateError(RuntimeError):
    """Raised when a run cannot be created, restored, or validated safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            json.dump(
                value,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_run_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir.expanduser().resolve() / STATUS_FILENAME
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunStateError(
            f"No {STATUS_FILENAME} exists in {run_dir}."
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunStateError(
            f"Cannot read valid run status from {path}."
        ) from error
    if not isinstance(status, dict):
        raise RunStateError(f"Run status in {path} is not a JSON object.")
    if status.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RunStateError(
            f"Unsupported run status schema in {path}: "
            f"{status.get('schema_version')!r}."
        )
    return status


def format_run_status(run_dir: Path, status: Mapping[str, Any]) -> str:
    planned = int(status.get("planned_total_tvb_calls", 0))
    completed = int(status.get("completed_tvb_calls", 0))
    percent = 100.0 * completed / planned if planned else 0.0
    lines = [
        f"Run directory: {run_dir.expanduser().resolve()}",
        f"State: {status.get('state', 'unknown')}",
        f"Mode: {status.get('mode', 'unknown')}",
        (
            f"TVB progress: {completed}/{planned} calls "
            f"({percent:.1f}%)"
        ),
        f"Current stage: {status.get('current_stage') or 'none'}",
        f"Attempt: {status.get('attempt', 0)}",
        f"Updated: {status.get('updated_utc', 'unknown')}",
    ]
    restored = int(status.get("restored_tvb_calls_this_attempt", 0))
    executed = int(status.get("executed_tvb_calls_this_attempt", 0))
    lines.append(
        f"This attempt: {restored} restored calls, {executed} executed calls"
    )
    error = status.get("last_error")
    if isinstance(error, dict) and error.get("message"):
        lines.append(
            f"Last error: {error.get('type', 'Error')}: {error['message']}"
        )
    return "\n".join(lines)


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }


def execution_code_sha256(project_root: Path) -> str:
    """Fingerprint all direct-run Python execution files."""

    files = [project_root / "main.py"]
    files.extend(sorted((project_root / "rise_tvb379").glob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_root(run_dir: Path) -> Path:
    return run_dir.parent / CHECKPOINT_ROOT_NAME / run_dir.name


def _status_path(run_dir: Path) -> Path:
    return run_dir / STATUS_FILENAME


class RunController:
    """Own the atomic status file, progress log, and recovery accounting."""

    def __init__(
        self,
        *,
        run_dir: Path,
        status: dict[str, Any],
        notebook_sha256: str,
        code_sha256: str,
        environment: Mapping[str, Any],
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.status = status
        self.notebook_sha256 = notebook_sha256
        self.code_sha256 = code_sha256
        self.environment = dict(environment)
        self.checkpoint_root = Path(status["checkpoint_root"])
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._attempt_started = time.perf_counter()
        self._restored_keys: set[str] = set()
        self._executed_keys: set[str] = set()

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        mode: str,
        notebook_sha256: str,
        code_sha256: str,
        environment: Mapping[str, Any],
        planned_total_tvb_calls: int,
        worker_processes: int,
    ) -> "RunController":
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = _status_path(run_dir)
        if status_path.exists():
            raise RunStateError(
                f"Run status already exists in {run_dir}; use --resume."
            )
        now = utc_now()
        status: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "state": "running",
            "mode": mode,
            "run_dir": str(run_dir),
            "checkpoint_root": str(_checkpoint_root(run_dir)),
            "notebook_sha256": notebook_sha256,
            "execution_code_sha256": code_sha256,
            "environment": dict(environment),
            "planned_total_tvb_calls": int(planned_total_tvb_calls),
            "completed_tvb_calls": 0,
            "completed_work": {},
            "restored_tvb_calls_this_attempt": 0,
            "executed_tvb_calls_this_attempt": 0,
            "configured_worker_processes": int(worker_processes),
            "current_stage": "initializing",
            "completed_source_cells": [],
            "input_hashes": {},
            "attempt": 1,
            "created_utc": now,
            "started_utc": now,
            "updated_utc": now,
            "exit_utc": None,
            "last_error": None,
        }
        controller = cls(
            run_dir=run_dir,
            status=status,
            notebook_sha256=notebook_sha256,
            code_sha256=code_sha256,
            environment=environment,
        )
        controller._write_status()
        controller.log(
            f"Created {mode} run with "
            f"{planned_total_tvb_calls} planned TVB calls and "
            f"{worker_processes} worker process(es)."
        )
        return controller

    @classmethod
    def resume(
        cls,
        *,
        run_dir: Path,
        mode: str,
        notebook_sha256: str,
        code_sha256: str,
        environment: Mapping[str, Any],
        planned_total_tvb_calls: int,
        worker_processes: int,
    ) -> "RunController":
        run_dir = run_dir.resolve()
        status = read_run_status(run_dir)
        mismatches: list[str] = []
        if status.get("state") == "completed":
            raise RunStateError(f"Run is already completed: {run_dir}.")
        if status.get("mode") != mode:
            mismatches.append(
                f"mode {status.get('mode')!r} != {mode!r}"
            )
        if status.get("notebook_sha256") != notebook_sha256:
            mismatches.append("canonical notebook SHA-256")
        if status.get("execution_code_sha256") != code_sha256:
            mismatches.append("Python execution-code SHA-256")
        if status.get("environment") != dict(environment):
            mismatches.append("Python or dependency versions")
        if int(status.get("planned_total_tvb_calls", -1)) != int(
            planned_total_tvb_calls
        ):
            mismatches.append("resolved TVB workload")
        if mismatches:
            raise RunStateError(
                "Resume refused because compatibility checks differ: "
                + ", ".join(mismatches)
                + "."
            )

        now = utc_now()
        status["state"] = "running"
        status["attempt"] = int(status.get("attempt", 0)) + 1
        status["started_utc"] = now
        status["updated_utc"] = now
        status["exit_utc"] = None
        status["current_stage"] = "restoring"
        status["configured_worker_processes"] = int(worker_processes)
        status["restored_tvb_calls_this_attempt"] = 0
        status["executed_tvb_calls_this_attempt"] = 0
        status["last_error"] = None
        controller = cls(
            run_dir=run_dir,
            status=status,
            notebook_sha256=notebook_sha256,
            code_sha256=code_sha256,
            environment=environment,
        )
        controller._write_status()
        controller.log(
            f"Starting resume attempt {status['attempt']} with "
            f"{worker_processes} worker process(es)."
        )
        return controller

    def _write_status(self) -> None:
        self.status["updated_utc"] = utc_now()
        atomic_write_json(_status_path(self.run_dir), self.status)

    def log(self, message: str) -> None:
        timestamp = utc_now()
        line = f"{timestamp} | {message}"
        print(line, flush=True)
        log_path = self.run_dir / PROGRESS_LOG_FILENAME
        with log_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def set_input_hashes(self, hashes: Mapping[str, str]) -> None:
        resolved = {str(key): str(value) for key, value in sorted(hashes.items())}
        existing = self.status.get("input_hashes") or {}
        if existing and existing != resolved:
            raise RunStateError(
                "Resume refused because verified input hashes changed."
            )
        self.status["input_hashes"] = resolved
        self._write_status()
        self.log(f"Verified {len(resolved)} pinned source inputs.")

    def begin_stage(
        self,
        *,
        stage: str,
        total_jobs: int,
        restored_jobs: int,
    ) -> None:
        self.status["current_stage"] = stage
        self._write_status()
        self.log(
            f"{stage}: {total_jobs} work unit(s), "
            f"{restored_jobs} checkpoint(s) available."
        )

    def invalidate_work(self, work_key: str) -> None:
        completed_work = self.status.setdefault("completed_work", {})
        if work_key not in completed_work:
            return
        del completed_work[work_key]
        self.status["completed_tvb_calls"] = sum(
            int(value) for value in completed_work.values()
        )
        self._write_status()

    def record_work(
        self,
        *,
        work_key: str,
        tvb_calls: int,
        restored: bool,
        stage: str,
        stage_completed_jobs: int,
        stage_total_jobs: int,
    ) -> None:
        completed_work = self.status.setdefault("completed_work", {})
        if work_key not in completed_work:
            completed_work[work_key] = int(tvb_calls)
        if restored:
            if work_key not in self._restored_keys:
                self._restored_keys.add(work_key)
                self.status["restored_tvb_calls_this_attempt"] = int(
                    self.status.get("restored_tvb_calls_this_attempt", 0)
                ) + int(tvb_calls)
        elif work_key not in self._executed_keys:
            self._executed_keys.add(work_key)
            self.status["executed_tvb_calls_this_attempt"] = int(
                self.status.get("executed_tvb_calls_this_attempt", 0)
            ) + int(tvb_calls)

        completed_calls = sum(int(value) for value in completed_work.values())
        planned_calls = int(self.status["planned_total_tvb_calls"])
        self.status["completed_tvb_calls"] = completed_calls
        self.status["current_stage"] = stage
        self._write_status()

        percent = 100.0 * completed_calls / planned_calls
        elapsed = max(time.perf_counter() - self._attempt_started, 1e-9)
        executed = int(self.status["executed_tvb_calls_this_attempt"])
        eta_text = ""
        if executed > 0 and completed_calls < planned_calls:
            seconds_per_call = elapsed / executed
            eta_seconds = seconds_per_call * (planned_calls - completed_calls)
            eta_text = f", ETA {_duration(eta_seconds)}"
        action = "restored" if restored else "saved"
        self.log(
            f"{stage}: {action} work unit "
            f"{stage_completed_jobs}/{stage_total_jobs}; overall "
            f"{completed_calls}/{planned_calls} TVB calls "
            f"({percent:.1f}%), elapsed {_duration(elapsed)}{eta_text}."
        )

    def complete_source_cell(self, cell_number: int, total_cells: int) -> None:
        completed = self.status.setdefault("completed_source_cells", [])
        if cell_number not in completed:
            completed.append(cell_number)
            completed.sort()
        self.status["current_stage"] = (
            f"completed source cell {cell_number}/{total_cells}"
        )
        self._write_status()

    def begin_source_cell(self, cell_number: int, total_cells: int) -> None:
        self.status["current_stage"] = (
            f"source cell {cell_number}/{total_cells}"
        )
        self._write_status()

    def mark_completed(self) -> None:
        completed = int(self.status.get("completed_tvb_calls", 0))
        planned = int(self.status["planned_total_tvb_calls"])
        if completed != planned:
            raise RunStateError(
                f"Cannot mark run complete: {completed}/{planned} TVB calls "
                "have durable checkpoints."
            )
        self.status["state"] = "completed"
        self.status["current_stage"] = "completed"
        self.status["exit_utc"] = utc_now()
        self._write_status()
        self.log(
            f"Run completed with {completed}/{planned} TVB calls durably "
            "checkpointed."
        )

    def mark_interrupted(self) -> None:
        self.status["state"] = "interrupted"
        self.status["current_stage"] = "interrupted"
        self.status["exit_utc"] = utc_now()
        self._write_status()
        self.log(
            "Run interrupted. Resume with "
            f"python main.py --resume {self.run_dir}"
        )

    def mark_failed(self, error: BaseException) -> None:
        self.status["state"] = "failed"
        self.status["current_stage"] = "failed"
        self.status["exit_utc"] = utc_now()
        self.status["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(
                    type(error),
                    error,
                    error.__traceback__,
                )
            ),
        }
        self._write_status()
        self.log(f"Run failed: {type(error).__name__}: {error}")


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _safe_stage_name(call_index: int, description: str) -> str:
    cleaned = _SAFE_STAGE.sub("_", description.strip()).strip("._")
    return f"{call_index:03d}_{cleaned or 'work'}"


def _job_weight(worker_name: str, job: Any) -> int:
    if isinstance(job, Mapping) and "probes" in job:
        return 1 + len(tuple(job["probes"]))
    if worker_name == "_run_calibration_block":
        return 2
    return 1


def _execute_tagged_job(
    position: int,
    worker_function: Any,
    job: Any,
    shared_args: Sequence[Any],
) -> tuple[int, Any]:
    return position, worker_function(job, *shared_args)


class CheckpointDispatcher:
    """Drop-in replacement for the notebook's parallel job dispatcher."""

    def __init__(
        self,
        namespace: dict[str, Any],
        controller: RunController,
    ) -> None:
        self.namespace = namespace
        self.controller = controller
        self.call_index = 0

    def __call__(
        self,
        worker_function: Any,
        job_payloads: Sequence[Any],
        shared_args: Sequence[Any],
        description: str,
    ) -> list[Any]:
        jobs = list(job_payloads)
        if not jobs:
            return []
        self.call_index += 1
        stage = _safe_stage_name(self.call_index, str(description))
        worker_name = getattr(
            worker_function,
            "__name__",
            type(worker_function).__name__,
        )
        joblib = self.namespace["joblib"]
        hashes = [
            joblib.hash(
                {"worker": worker_name, "payload": job},
                hash_name="sha1",
            )
            for job in jobs
        ]
        weights = [_job_weight(worker_name, job) for job in jobs]
        stage_dir = self.controller.checkpoint_root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = stage_dir / "manifest.json"
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "notebook_sha256": self.controller.notebook_sha256,
            "execution_code_sha256": self.controller.code_sha256,
            "stage": stage,
            "description": str(description),
            "worker": worker_name,
            "shared_args_hash": joblib.hash(
                tuple(shared_args),
                hash_name="sha1",
            ),
            "job_hashes": hashes,
            "tvb_call_weights": weights,
        }
        if manifest_path.exists():
            try:
                saved_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RunStateError(
                    f"Cannot read checkpoint manifest {manifest_path}."
                ) from error
            if saved_manifest != manifest:
                raise RunStateError(
                    f"Checkpoint manifest mismatch for {stage}."
                )
        else:
            atomic_write_json(manifest_path, manifest)

        outcomes: list[Any | None] = [None] * len(jobs)
        pending: list[tuple[int, Any]] = []
        restored: list[tuple[int, str, int]] = []
        for position, (job, job_hash, weight) in enumerate(
            zip(jobs, hashes, weights, strict=True)
        ):
            checkpoint_path = self._checkpoint_path(
                stage_dir,
                position,
                job_hash,
            )
            outcome = self._load_checkpoint(
                checkpoint_path=checkpoint_path,
                stage=stage,
                job_hash=job_hash,
                joblib=joblib,
            )
            if outcome is None:
                self.controller.invalidate_work(
                    f"{stage}:{job_hash}"
                )
                pending.append((position, job))
                continue
            outcomes[position] = outcome
            restored.append((position, job_hash, weight))

        restored_count = len(restored)
        self.controller.begin_stage(
            stage=stage,
            total_jobs=len(jobs),
            restored_jobs=restored_count,
        )
        for restored_ordinal, (_position, job_hash, weight) in enumerate(
            restored,
            start=1,
        ):
            self.controller.record_work(
                work_key=f"{stage}:{job_hash}",
                tvb_calls=weight,
                restored=True,
                stage=stage,
                stage_completed_jobs=restored_ordinal,
                stage_total_jobs=len(jobs),
            )

        completed_in_stage = restored_count
        if pending and int(self.namespace["PARALLEL_WORKERS"]) == 1:
            for position, job in pending:
                outcome = worker_function(job, *shared_args)
                self._save_and_record(
                    outcomes=outcomes,
                    position=position,
                    outcome=outcome,
                    stage=stage,
                    stage_dir=stage_dir,
                    job_hash=hashes[position],
                    weight=weights[position],
                    completed_in_stage=completed_in_stage + 1,
                    total_jobs=len(jobs),
                    joblib=joblib,
                )
                completed_in_stage += 1
        elif pending:
            memmap_dir = Path(self.namespace["WORK_DIR"]) / ".joblib_memmap"
            memmap_dir.mkdir(parents=True, exist_ok=True)
            parallel_config = self.namespace["parallel_config"]
            Parallel = self.namespace["Parallel"]
            delayed = self.namespace["delayed"]
            with parallel_config(
                backend="loky",
                n_jobs=int(self.namespace["PARALLEL_WORKERS"]),
                inner_max_num_threads=1,
                initializer=self.namespace["initialize_parallel_worker"],
                initargs=(str(self.namespace["WORK_DIR"]),),
                idle_worker_timeout=900,
                temp_folder=str(memmap_dir),
                max_nbytes="512K",
                mmap_mode="r",
            ):
                outcome_stream = Parallel(
                    return_as="generator_unordered",
                    batch_size=1,
                    pre_dispatch="2*n_jobs",
                )(
                    delayed(_execute_tagged_job)(
                        position,
                        worker_function,
                        job,
                        tuple(shared_args),
                    )
                    for position, job in pending
                )
                for position, outcome in outcome_stream:
                    self._save_and_record(
                        outcomes=outcomes,
                        position=position,
                        outcome=outcome,
                        stage=stage,
                        stage_dir=stage_dir,
                        job_hash=hashes[position],
                        weight=weights[position],
                        completed_in_stage=completed_in_stage + 1,
                        total_jobs=len(jobs),
                        joblib=joblib,
                    )
                    completed_in_stage += 1

        if any(outcome is None for outcome in outcomes):
            raise RunStateError(f"Stage {stage} did not produce every outcome.")
        self.controller.log(
            f"{stage}: all {len(jobs)} work unit(s) available."
        )
        return list(outcomes)

    @staticmethod
    def _checkpoint_path(
        stage_dir: Path,
        position: int,
        job_hash: str,
    ) -> Path:
        return stage_dir / f"{position:05d}_{job_hash}.joblib"

    def _load_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        stage: str,
        job_hash: str,
        joblib: Any,
    ) -> Any | None:
        marker_path = self._checkpoint_marker_path(checkpoint_path)
        if not checkpoint_path.is_file() or not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker_valid = (
                isinstance(marker, dict)
                and marker.get("schema_version")
                == CHECKPOINT_SCHEMA_VERSION
                and marker.get("file") == checkpoint_path.name
                and marker.get("stage") == stage
                and marker.get("job_hash") == job_hash
                and marker.get("notebook_sha256")
                == self.controller.notebook_sha256
                and marker.get("execution_code_sha256")
                == self.controller.code_sha256
                and marker.get("size_bytes") == checkpoint_path.stat().st_size
                and marker.get("sha256") == sha256_file(checkpoint_path)
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            marker_valid = False
        if not marker_valid:
            self.controller.log(
                f"{stage}: ignoring incomplete or damaged checkpoint "
                f"{checkpoint_path.name}."
            )
            return None
        try:
            envelope = joblib.load(checkpoint_path)
        except Exception as error:
            self.controller.log(
                f"{stage}: ignoring unreadable partial checkpoint "
                f"{checkpoint_path.name}: {error}"
            )
            return None
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "notebook_sha256": self.controller.notebook_sha256,
            "execution_code_sha256": self.controller.code_sha256,
            "stage": stage,
            "job_hash": job_hash,
        }
        if not isinstance(envelope, dict) or any(
            envelope.get(key) != value for key, value in expected.items()
        ) or "outcome" not in envelope:
            self.controller.log(
                f"{stage}: ignoring incompatible checkpoint "
                f"{checkpoint_path.name}."
            )
            return None
        return envelope["outcome"]

    @staticmethod
    def _checkpoint_marker_path(checkpoint_path: Path) -> Path:
        return checkpoint_path.with_name(
            f"{checkpoint_path.name}.complete.json"
        )

    def _save_and_record(
        self,
        *,
        outcomes: list[Any | None],
        position: int,
        outcome: Any,
        stage: str,
        stage_dir: Path,
        job_hash: str,
        weight: int,
        completed_in_stage: int,
        total_jobs: int,
        joblib: Any,
    ) -> None:
        checkpoint_path = self._checkpoint_path(
            stage_dir,
            position,
            job_hash,
        )
        temporary = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}."
            f"{uuid.uuid4().hex}.partial"
        )
        envelope = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "notebook_sha256": self.controller.notebook_sha256,
            "execution_code_sha256": self.controller.code_sha256,
            "stage": stage,
            "job_hash": job_hash,
            "saved_utc": utc_now(),
            "outcome": outcome,
        }
        marker_path = self._checkpoint_marker_path(checkpoint_path)
        try:
            marker_path.unlink(missing_ok=True)
            joblib.dump(envelope, temporary, compress=0)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint_path)
            _fsync_directory(stage_dir)
            atomic_write_json(
                marker_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "notebook_sha256": self.controller.notebook_sha256,
                    "execution_code_sha256": self.controller.code_sha256,
                    "stage": stage,
                    "job_hash": job_hash,
                    "file": checkpoint_path.name,
                    "size_bytes": checkpoint_path.stat().st_size,
                    "sha256": sha256_file(checkpoint_path),
                    "completed_utc": utc_now(),
                },
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        outcomes[position] = outcome
        self.controller.record_work(
            work_key=f"{stage}:{job_hash}",
            tvb_calls=weight,
            restored=False,
            stage=stage,
            stage_completed_jobs=completed_in_stage,
            stage_total_jobs=total_jobs,
        )
