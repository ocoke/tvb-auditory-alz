"""Spawn-safe parallel execution for independent TVB work units.

This module deliberately imports only the Python standard library.  In
particular, NumPy and TVB must not be imported here: spawned workers need an
opportunity to configure their native thread limits and private TVB runtime
directories before either numerical library is initialized.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import os
from pathlib import Path
import queue
import time
import traceback as traceback_module
from typing import Any


LOGGER = logging.getLogger(__name__)

NATIVE_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CPU_ALLOCATION_ENVIRONMENT_VARIABLES = (
    "SLURM_CPUS_PER_TASK",
    "NSLOTS",
    "PBS_NP",
    "LSB_DJOB_NUMPROC",
)
SUPPORTED_JOB_KINDS = frozenset({"simulation", "calibration"})
WORKER_STARTUP_TIMEOUT_SECONDS = 300.0
QUEUE_POLL_SECONDS = 0.25
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0

_WORKER_CONTEXT: Any | None = None
_WORKER_SIMULATION_MODULE: Any | None = None


def configure_native_thread_limits(threads: int = 1) -> dict[str, str]:
    """Limit native numerical libraries to ``threads`` per worker process.

    Process-level parallelism is the outer layer for this experiment.  Capping
    BLAS, OpenMP, Accelerate, BLIS, and NumExpr prevents every worker from
    creating another full set of native threads and oversubscribing the host.
    Existing environment values are intentionally replaced.
    """

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    value = str(threads)
    configured = {
        variable: value for variable in NATIVE_THREAD_ENVIRONMENT_VARIABLES
    }
    os.environ.update(configured)
    return configured


def available_cpu_count() -> int:
    """Return the number of CPUs currently available to this process."""

    sources = cpu_allocation_sources()
    return min(sources.values())


def cpu_allocation_sources() -> dict[str, int]:
    """Return every valid host, affinity, and scheduler CPU limit."""

    sources = {"os_cpu_count": max(1, int(os.cpu_count() or 1))}
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
        except (OSError, TypeError, ValueError):
            count = 0
        if count > 0:
            sources["process_affinity"] = count
    for variable in CPU_ALLOCATION_ENVIRONMENT_VARIABLES:
        raw_value = os.environ.get(variable)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{variable} must be a positive integer, got {raw_value!r}."
            ) from error
        if value < 1:
            raise ValueError(
                f"{variable} must be a positive integer, got {value}."
            )
        sources[variable] = value
    return sources


def _resolve_worker_count(worker_count: int | None) -> int:
    if worker_count is None:
        return available_cpu_count()
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ValueError("worker_count must be a positive integer or None")
    return worker_count


def execution_details(worker_count: int | None = None) -> dict[str, Any]:
    """Describe the resolved process-level parallel execution settings."""

    available = available_cpu_count()
    resolved = _resolve_worker_count(worker_count)
    return {
        "execution_mode": (
            "single_process" if resolved == 1 else "process_parallel"
        ),
        "parallel_enabled": resolved > 1,
        "multiprocessing_start_method": (
            None if resolved == 1 else "spawn"
        ),
        "cpu_allocation_sources": cpu_allocation_sources(),
        "available_cpu_count": available,
        "requested_worker_count": worker_count,
        "worker_count": resolved,
        "native_threads_per_worker": 1,
        "native_thread_environment": {
            variable: "1"
            for variable in NATIVE_THREAD_ENVIRONMENT_VARIABLES
        },
        "oversubscribed": resolved > available,
    }


@dataclass(frozen=True, slots=True)
class WorkerJob:
    """One uniquely ordered, independently checkpointable worker request."""

    ordinal: int
    kind: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        if self.kind not in SUPPORTED_JOB_KINDS:
            choices = ", ".join(sorted(SUPPORTED_JOB_KINDS))
            raise ValueError(f"kind must be one of: {choices}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        # A plain dict is stable and spawn-picklable; MappingProxyType and
        # custom mapping implementations often are not.
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """A completed worker result, tagged for deterministic parent ordering."""

    ordinal: int
    kind: str
    result: Any
    worker_pid: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    """Pickle-safe details for an exception raised in a spawned worker."""

    phase: str
    worker_pid: int
    exception_module: str
    exception_type: str
    message: str
    traceback: str


@dataclass(frozen=True, slots=True)
class _WorkerMessage:
    """One control or result message sent from a worker to its parent."""

    kind: str
    worker_index: int
    worker_pid: int
    ordinal: int | None = None
    payload: Any = None


def _capture_worker_failure(
    phase: str,
    error: BaseException,
) -> _WorkerFailure:
    return _WorkerFailure(
        phase=phase,
        worker_pid=os.getpid(),
        exception_module=type(error).__module__,
        exception_type=type(error).__name__,
        message=str(error),
        traceback="".join(
            traceback_module.format_exception(
                type(error),
                error,
                error.__traceback__,
            )
        ),
    )


def _worker_failure_error(failure: _WorkerFailure) -> RuntimeError:
    detail = (
        f"{failure.exception_module}.{failure.exception_type}: "
        f"{failure.message}"
    ).rstrip()
    return RuntimeError(
        f"parallel worker {failure.worker_pid} failed during "
        f"{failure.phase}: {detail}\n{failure.traceback}"
    )


def _plain_context_payload(context: Any) -> dict[str, Any]:
    """Remove a dataclass type identity before crossing the spawn boundary.

    ``SimulationContext`` lives in the module that imports TVB.  Pickling that
    dataclass directly would import TVB while a child is still unpickling its
    initializer arguments, before the initializer can set ``TVB_USER_HOME``.
    Sending only its fields avoids that premature import.
    """

    if isinstance(context, Mapping):
        return dict(context)
    if is_dataclass(context) and not isinstance(context, type):
        return {
            field.name: getattr(context, field.name)
            for field in fields(context)
        }
    raise TypeError("context must be a mapping or dataclass instance")


def _worker_initializer(
    context_payload: Mapping[str, Any],
    run_dir: str,
    log_queue: Any,
    log_level: int,
) -> None:
    """Configure a spawned worker before importing the TVB simulation module."""

    global _WORKER_CONTEXT, _WORKER_SIMULATION_MODULE

    worker_root = (
        Path(run_dir) / ".runtime" / "workers" / str(os.getpid())
    )
    tvb_home = worker_root / "tvb"
    matplotlib_home = worker_root / "matplotlib"
    tvb_home.mkdir(parents=True, exist_ok=True)
    matplotlib_home.mkdir(parents=True, exist_ok=True)
    os.environ["TVB_USER_HOME"] = str(tvb_home)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_home)

    # These values are also set in the parent before spawn so they are already
    # present if unpickling imports NumPy.  Reassert them for clarity and for
    # worker initializers invoked directly by tests or other callers.
    configure_native_thread_limits(1)

    # Import only after the process-private TVB paths have been installed.
    from rise_tvb379 import simulation

    simulation.initialize_tvb_runtime()
    _WORKER_SIMULATION_MODULE = simulation
    _WORKER_CONTEXT = simulation.SimulationContext(**dict(context_payload))

    # TVB configures logging during import, so replace handlers afterwards.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))
    root_logger.setLevel(log_level)


def _execute_worker_job(job: WorkerJob) -> WorkerOutcome:
    """Dispatch one job inside an initialized worker process."""

    if _WORKER_CONTEXT is None or _WORKER_SIMULATION_MODULE is None:
        raise RuntimeError("parallel worker was not initialized")

    worker_logger = logging.getLogger("rise_tvb379.worker")
    worker_logger.info(
        "Worker %d starting %s job ordinal=%d",
        os.getpid(),
        job.kind,
        job.ordinal,
    )
    started = time.perf_counter()
    if job.kind == "simulation":
        result = _WORKER_SIMULATION_MODULE.run_condition_seed_block(
            _WORKER_CONTEXT,
            **dict(job.payload),
        )
    elif job.kind == "calibration":
        result = _WORKER_SIMULATION_MODULE.run_calibration_block(
            _WORKER_CONTEXT,
            **dict(job.payload),
        )
    else:  # WorkerJob validation makes this defensive branch unreachable.
        raise RuntimeError(f"unsupported worker job kind: {job.kind}")
    elapsed = time.perf_counter() - started
    worker_logger.info(
        "Worker %d finished %s job ordinal=%d in %.2f seconds",
        os.getpid(),
        job.kind,
        job.ordinal,
        elapsed,
    )
    return WorkerOutcome(
        ordinal=job.ordinal,
        kind=job.kind,
        result=result,
        worker_pid=os.getpid(),
        elapsed_seconds=elapsed,
    )


def _worker_main(
    worker_index: int,
    context_payload: Mapping[str, Any],
    run_dir: str,
    log_queue: Any,
    log_level: int,
    job_queue: Any,
    result_queue: Any,
) -> None:
    """Initialize once, then execute jobs until the parent sends a sentinel."""

    try:
        _worker_initializer(
            context_payload,
            run_dir,
            log_queue,
            log_level,
        )
    except BaseException as error:
        result_queue.put(
            _WorkerMessage(
                kind="init_error",
                worker_index=worker_index,
                worker_pid=os.getpid(),
                payload=_capture_worker_failure("initialization", error),
            )
        )
        return

    result_queue.put(
        _WorkerMessage(
            kind="ready",
            worker_index=worker_index,
            worker_pid=os.getpid(),
        )
    )
    while True:
        try:
            job = job_queue.get()
        except (EOFError, OSError):
            return
        if job is None:
            return
        if not isinstance(job, WorkerJob):
            error = TypeError("worker received an invalid job")
            result_queue.put(
                _WorkerMessage(
                    kind="job_error",
                    worker_index=worker_index,
                    worker_pid=os.getpid(),
                    payload=_capture_worker_failure("job dispatch", error),
                )
            )
            return
        try:
            outcome = _execute_worker_job(job)
        except BaseException as error:
            result_queue.put(
                _WorkerMessage(
                    kind="job_error",
                    worker_index=worker_index,
                    worker_pid=os.getpid(),
                    ordinal=job.ordinal,
                    payload=_capture_worker_failure(
                        f"{job.kind} job ordinal={job.ordinal}",
                        error,
                    ),
                )
            )
            return
        result_queue.put(
            _WorkerMessage(
                kind="outcome",
                worker_index=worker_index,
                worker_pid=os.getpid(),
                ordinal=job.ordinal,
                payload=outcome,
            )
        )


class SerialRunner:
    """Execute the same checkpointable jobs in the current process."""

    worker_count = 1

    def __init__(
        self,
        context: Any,
        run_dir: str | os.PathLike[str],
    ) -> None:
        self._context_payload = _plain_context_payload(context)
        self.run_dir = Path(run_dir).resolve()
        self._context: Any | None = None
        self._simulation: Any | None = None
        self._entered = False
        self._active_execution = False

    def __enter__(self) -> SerialRunner:
        if self._entered:
            raise RuntimeError("SerialRunner cannot be entered more than once")
        configure_native_thread_limits(1)
        runtime_root = self.run_dir / ".runtime" / "serial"
        tvb_home = runtime_root / "tvb"
        matplotlib_home = runtime_root / "matplotlib"
        tvb_home.mkdir(parents=True, exist_ok=True)
        matplotlib_home.mkdir(parents=True, exist_ok=True)
        os.environ["TVB_USER_HOME"] = str(tvb_home)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_home)

        from rise_tvb379 import simulation

        simulation.initialize_tvb_runtime()
        self._simulation = simulation
        self._context = simulation.SimulationContext(
            **dict(self._context_payload)
        )
        self._entered = True
        LOGGER.info("Using single-process execution")
        return self

    def execute(self, jobs: Iterable[WorkerJob]) -> Iterator[WorkerOutcome]:
        if not self._entered:
            raise RuntimeError("SerialRunner must be entered before execute")
        if self._active_execution:
            raise RuntimeError("SerialRunner already has an active execution")
        job_list = list(jobs)
        if not all(isinstance(job, WorkerJob) for job in job_list):
            raise TypeError("jobs must contain only WorkerJob instances")
        ordinals = [job.ordinal for job in job_list]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("job ordinals must be unique within one execution")
        if not job_list:
            return iter(())
        self._active_execution = True
        return self._iterate_outcomes(job_list)

    def _iterate_outcomes(
        self,
        jobs: list[WorkerJob],
    ) -> Iterator[WorkerOutcome]:
        try:
            if self._context is None or self._simulation is None:
                raise RuntimeError("SerialRunner was not initialized")
            for job in jobs:
                started = time.perf_counter()
                if job.kind == "simulation":
                    result = self._simulation.run_condition_seed_block(
                        self._context,
                        **dict(job.payload),
                    )
                elif job.kind == "calibration":
                    result = self._simulation.run_calibration_block(
                        self._context,
                        **dict(job.payload),
                    )
                else:
                    raise RuntimeError(
                        f"unsupported serial job kind: {job.kind}"
                    )
                yield WorkerOutcome(
                    ordinal=job.ordinal,
                    kind=job.kind,
                    result=result,
                    worker_pid=os.getpid(),
                    elapsed_seconds=time.perf_counter() - started,
                )
        finally:
            self._active_execution = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool:
        self._active_execution = False
        self._entered = False
        self._context = None
        self._simulation = None
        return False


class ParallelRunner:
    """Reusable persistent spawn pool for multiple experiment stages.

    ``execute`` yields outcomes in completion order.  Callers should checkpoint
    each yielded outcome immediately and store it by ``outcome.ordinal`` before
    deterministic final aggregation.
    """

    def __init__(
        self,
        context: Any,
        run_dir: str | os.PathLike[str],
        worker_count: int | None = None,
        *,
        handlers: Sequence[logging.Handler] | None = None,
        log_level: int | None = None,
    ) -> None:
        self._context_payload = _plain_context_payload(context)
        self.run_dir = Path(run_dir).resolve()
        self.worker_count = _resolve_worker_count(worker_count)
        self._handlers = tuple(handlers) if handlers is not None else None
        self._log_level = (
            int(log_level)
            if log_level is not None
            else logging.getLogger().getEffectiveLevel()
        )
        self._mp_context: Any | None = None
        self._log_queue: Any | None = None
        self._job_queue: Any | None = None
        self._result_queue: Any | None = None
        self._queue_listener: QueueListener | None = None
        self._workers: list[Any] = []
        self._entered = False
        self._active_execution = False
        self._broken = False

    def __enter__(self) -> ParallelRunner:
        if self._entered:
            raise RuntimeError("ParallelRunner cannot be entered more than once")

        configure_native_thread_limits(1)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._mp_context = multiprocessing.get_context("spawn")
        self._log_queue = self._mp_context.Queue()
        self._job_queue = self._mp_context.Queue()
        self._result_queue = self._mp_context.Queue()

        handlers = self._handlers
        if handlers is None:
            handlers = tuple(
                handler
                for handler in logging.getLogger().handlers
                if not isinstance(handler, QueueHandler)
            )
        if not handlers:
            handlers = (logging.StreamHandler(),)

        self._queue_listener = QueueListener(
            self._log_queue,
            *handlers,
            respect_handler_level=True,
        )
        self._queue_listener.start()
        try:
            for worker_index in range(self.worker_count):
                worker = self._mp_context.Process(
                    target=_worker_main,
                    args=(
                        worker_index,
                        self._context_payload,
                        str(self.run_dir),
                        self._log_queue,
                        self._log_level,
                        self._job_queue,
                        self._result_queue,
                    ),
                    name=f"TVBWorker-{worker_index + 1}",
                )
                worker.start()
                self._workers.append(worker)
            self._wait_for_worker_readiness()
        except BaseException:
            self._broken = True
            self._terminate_workers_safely()
            self._stop_logging_and_queues(aborted=True)
            raise

        self._entered = True
        LOGGER.info(
            "Started persistent spawn pool with %d worker processes",
            self.worker_count,
        )
        return self

    def execute(self, jobs: Iterable[WorkerJob]) -> Iterator[WorkerOutcome]:
        """Yield completed outcomes in worker completion order."""

        if not self._entered:
            raise RuntimeError("ParallelRunner must be entered before execute")
        if self._broken:
            raise RuntimeError("ParallelRunner is unusable after a worker failure")
        if self._active_execution:
            raise RuntimeError("ParallelRunner already has an active execution")
        self._raise_if_worker_died()

        job_list = list(jobs)
        if not all(isinstance(job, WorkerJob) for job in job_list):
            raise TypeError("jobs must contain only WorkerJob instances")
        ordinals = [job.ordinal for job in job_list]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("job ordinals must be unique within one execution")
        if not job_list:
            return iter(())

        assert self._job_queue is not None
        for job in job_list:
            self._job_queue.put(job)
        self._active_execution = True
        return self._iterate_outcomes(job_list)

    def _wait_for_worker_readiness(self) -> None:
        assert self._result_queue is not None
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        ready_indices: set[int] = set()
        while len(ready_indices) < self.worker_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                missing = sorted(
                    set(range(self.worker_count)) - ready_indices
                )
                raise RuntimeError(
                    "parallel worker initialization timed out; "
                    f"workers not ready: {missing}"
                )
            try:
                message = self._result_queue.get(
                    timeout=min(QUEUE_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                self._raise_if_worker_died(starting=True)
                continue
            if not isinstance(message, _WorkerMessage):
                raise RuntimeError(
                    "parallel worker sent an invalid initialization message"
                )
            if message.kind == "init_error":
                if not isinstance(message.payload, _WorkerFailure):
                    raise RuntimeError(
                        "parallel worker initialization failed without details"
                    )
                raise _worker_failure_error(message.payload)
            if message.kind != "ready":
                raise RuntimeError(
                    "parallel worker sent a result before initialization "
                    f"completed: {message.kind}"
                )
            if (
                message.worker_index < 0
                or message.worker_index >= self.worker_count
                or message.worker_index in ready_indices
            ):
                raise RuntimeError(
                    "parallel worker sent an invalid or duplicate ready message"
                )
            ready_indices.add(message.worker_index)

    def _iterate_outcomes(
        self,
        jobs: list[WorkerJob],
    ) -> Iterator[WorkerOutcome]:
        pending = {job.ordinal: job for job in jobs}
        try:
            assert self._result_queue is not None
            while pending:
                try:
                    message = self._result_queue.get(
                        timeout=QUEUE_POLL_SECONDS
                    )
                except queue.Empty:
                    self._raise_if_worker_died()
                    continue
                if not isinstance(message, _WorkerMessage):
                    raise RuntimeError(
                        "parallel worker sent an invalid result message"
                    )
                if message.kind == "job_error":
                    if not isinstance(message.payload, _WorkerFailure):
                        raise RuntimeError(
                            "parallel worker job failed without details"
                        )
                    raise _worker_failure_error(message.payload)
                if message.kind != "outcome":
                    raise RuntimeError(
                        "parallel worker sent an unexpected result message: "
                        f"{message.kind}"
                    )
                if (
                    message.ordinal is None
                    or message.ordinal not in pending
                ):
                    raise RuntimeError(
                        "parallel worker returned an unknown or duplicate "
                        f"job ordinal: {message.ordinal}"
                    )
                outcome = message.payload
                if not isinstance(outcome, WorkerOutcome):
                    raise RuntimeError(
                        "parallel worker returned an invalid outcome"
                    )
                if (
                    outcome.ordinal != message.ordinal
                    or outcome.kind != pending[message.ordinal].kind
                ):
                    raise RuntimeError(
                        "parallel worker outcome identity does not match "
                        f"job ordinal={message.ordinal}"
                    )
                del pending[message.ordinal]
                yield outcome
        except BaseException:
            self._broken = True
            self._terminate_workers_safely()
            raise
        finally:
            self._active_execution = False

    def _raise_if_worker_died(self, *, starting: bool = False) -> None:
        for worker_index, worker in enumerate(self._workers):
            exitcode = worker.exitcode
            if exitcode is not None:
                phase = (
                    "during initialization"
                    if starting
                    else "while jobs were pending"
                )
                raise RuntimeError(
                    f"parallel worker {worker_index} (pid={worker.pid}) "
                    f"exited unexpectedly with code {exitcode} {phase}"
                )

    def _terminate_workers_safely(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            try:
                if worker.is_alive():
                    worker.terminate()
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_SECONDS
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                worker.join(timeout=remaining)
            except (OSError, ValueError):
                continue
        for worker in workers:
            try:
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            except (AttributeError, OSError, ValueError):
                pass

    def _close_workers_safely(self) -> None:
        workers, self._workers = self._workers, []
        if not workers:
            return
        assert self._job_queue is not None
        for worker in workers:
            if worker.is_alive():
                self._job_queue.put(None)
        deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_SECONDS
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            worker.join(timeout=remaining)
        stragglers = [worker for worker in workers if worker.is_alive()]
        for worker in stragglers:
            worker.terminate()
        for worker in stragglers:
            worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            if worker.is_alive():
                worker.kill()
                worker.join()

    def _stop_logging_and_queues(self, *, aborted: bool) -> None:
        listener, self._queue_listener = self._queue_listener, None
        if listener is not None:
            listener.stop()
        queues = (
            ("job", self._job_queue),
            ("result", self._result_queue),
            ("log", self._log_queue),
        )
        self._job_queue = None
        self._result_queue = None
        self._log_queue = None
        for queue_name, managed_queue in queues:
            if managed_queue is None:
                continue
            if aborted and queue_name == "job":
                cancel_join = getattr(
                    managed_queue,
                    "cancel_join_thread",
                    None,
                )
                if cancel_join is not None:
                    cancel_join()
            close = getattr(managed_queue, "close", None)
            if close is not None:
                close()
            join_thread = getattr(managed_queue, "join_thread", None)
            if join_thread is not None:
                join_thread()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool:
        aborted = exc_type is not None or self._active_execution or self._broken
        try:
            if aborted:
                self._terminate_workers_safely()
            else:
                self._close_workers_safely()
        finally:
            self._active_execution = False
            self._entered = False
            self._stop_logging_and_queues(aborted=aborted)
        return False


__all__ = [
    "CPU_ALLOCATION_ENVIRONMENT_VARIABLES",
    "NATIVE_THREAD_ENVIRONMENT_VARIABLES",
    "ParallelRunner",
    "SerialRunner",
    "WorkerJob",
    "WorkerOutcome",
    "available_cpu_count",
    "configure_native_thread_limits",
    "cpu_allocation_sources",
    "execution_details",
]
