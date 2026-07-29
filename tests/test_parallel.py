from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from rise_tvb379 import parallel


class FakeQueue:
    def __init__(self) -> None:
        self.items: deque[Any] = deque()
        self.closed = False
        self.joined = False
        self.join_cancelled = False

    def put(self, value: Any) -> None:
        self.items.append(value)

    def get(self, timeout: float | None = None) -> Any:
        if self.items:
            return self.items.popleft()
        raise parallel.queue.Empty

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True

    def cancel_join_thread(self) -> None:
        self.join_cancelled = True


class FakeListener:
    instances: list["FakeListener"] = []

    def __init__(self, queue: Any, *handlers: Any, **options: Any) -> None:
        self.queue = queue
        self.handlers = handlers
        self.options = options
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeProcess:
    def __init__(
        self,
        context: "FakeMultiprocessingContext",
        *,
        target: Any,
        args: tuple[Any, ...],
        name: str,
    ) -> None:
        self.context = context
        self.target = target
        self.args = args
        self.name = name
        self.worker_index = int(args[0])
        self.pid = 10_000 + self.worker_index
        self.exitcode: int | None = None
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_count = 0

    def start(self) -> None:
        self.started = True
        result_queue = self.args[-1]
        failure = self.context.initialization_failures.get(
            self.worker_index
        )
        if failure is not None:
            self.exitcode = 1
            result_queue.put(
                parallel._WorkerMessage(
                    kind="init_error",
                    worker_index=self.worker_index,
                    worker_pid=self.pid,
                    payload=failure,
                )
            )
            return
        result_queue.put(
            parallel._WorkerMessage(
                kind="ready",
                worker_index=self.worker_index,
                worker_pid=self.pid,
            )
        )

    def is_alive(self) -> bool:
        return self.started and self.exitcode is None

    def join(self, timeout: float | None = None) -> None:
        self.join_count += 1
        job_queue = self.args[-2]
        if any(item is None for item in job_queue.items):
            self.exitcode = 0

    def terminate(self) -> None:
        self.terminated = True
        self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.exitcode = -9

    def die(self, exitcode: int = -9) -> None:
        self.exitcode = exitcode


class FakeMultiprocessingContext:
    def __init__(
        self,
        *,
        initialization_failures: dict[int, parallel._WorkerFailure]
        | None = None,
    ) -> None:
        self.initialization_failures = initialization_failures or {}
        self.queues: list[FakeQueue] = []
        self.processes: list[FakeProcess] = []

    @property
    def log_queue(self) -> FakeQueue:
        return self.queues[0]

    @property
    def job_queue(self) -> FakeQueue:
        return self.queues[1]

    @property
    def result_queue(self) -> FakeQueue:
        return self.queues[2]

    def Queue(self) -> FakeQueue:
        created = FakeQueue()
        self.queues.append(created)
        return created

    def Process(self, **options: Any) -> FakeProcess:
        process = FakeProcess(self, **options)
        self.processes.append(process)
        return process


@dataclass
class TinyContext:
    n_regions: int
    label: str


def install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialization_failures: dict[int, parallel._WorkerFailure]
    | None = None,
) -> FakeMultiprocessingContext:
    fake_context = FakeMultiprocessingContext(
        initialization_failures=initialization_failures
    )
    FakeListener.instances.clear()
    monkeypatch.setattr(
        parallel.multiprocessing,
        "get_context",
        lambda method: fake_context if method == "spawn" else None,
    )
    monkeypatch.setattr(parallel, "QueueListener", FakeListener)
    return fake_context


def worker_failure(
    message: str,
    *,
    phase: str = "test",
    pid: int = 10_000,
) -> parallel._WorkerFailure:
    return parallel._WorkerFailure(
        phase=phase,
        worker_pid=pid,
        exception_module="builtins",
        exception_type="RuntimeError",
        message=message,
        traceback=f"RuntimeError: {message}\n",
    )


def test_configure_native_thread_limits_overrides_all_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in parallel.NATIVE_THREAD_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(variable, "99")

    configured = parallel.configure_native_thread_limits(3)

    assert configured == {
        variable: "3"
        for variable in parallel.NATIVE_THREAD_ENVIRONMENT_VARIABLES
    }
    assert all(os.environ[name] == "3" for name in configured)
    with pytest.raises(ValueError, match="positive integer"):
        parallel.configure_native_thread_limits(0)
    with pytest.raises(ValueError, match="positive integer"):
        parallel.configure_native_thread_limits(True)


def test_available_cpu_count_prefers_affinity_and_has_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parallel.os,
        "sched_getaffinity",
        lambda process_id: {0, 2, 4},
        raising=False,
    )
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 99)
    assert parallel.available_cpu_count() == 3

    def unavailable_affinity(process_id: int) -> set[int]:
        raise OSError("unsupported")

    monkeypatch.setattr(
        parallel.os,
        "sched_getaffinity",
        unavailable_affinity,
        raising=False,
    )
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)
    assert parallel.available_cpu_count() == 1


def test_available_cpu_count_honors_scheduler_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parallel.os,
        "sched_getaffinity",
        lambda process_id: set(range(12)),
        raising=False,
    )
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")

    assert parallel.available_cpu_count() == 4
    assert parallel.cpu_allocation_sources()["SLURM_CPUS_PER_TASK"] == 4


def test_execution_details_defaults_to_every_available_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "available_cpu_count", lambda: 8)

    automatic = parallel.execution_details()
    explicit = parallel.execution_details(3)

    assert automatic["worker_count"] == 8
    assert automatic["requested_worker_count"] is None
    assert automatic["multiprocessing_start_method"] == "spawn"
    assert automatic["native_threads_per_worker"] == 1
    assert automatic["oversubscribed"] is False
    assert explicit["worker_count"] == 3
    serial = parallel.execution_details(1)
    assert serial["execution_mode"] == "single_process"
    assert serial["parallel_enabled"] is False
    assert serial["multiprocessing_start_method"] is None
    with pytest.raises(ValueError, match="worker_count"):
        parallel.execution_details(-1)


def test_serial_runner_executes_in_current_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rise_tvb379 import simulation

    initialized: list[bool] = []
    monkeypatch.setattr(
        simulation,
        "initialize_tvb_runtime",
        lambda: initialized.append(True),
    )
    monkeypatch.setattr(
        simulation,
        "SimulationContext",
        lambda **values: values,
    )
    monkeypatch.setattr(
        simulation,
        "run_condition_seed_block",
        lambda context, **payload: {
            "context": context,
            "payload": payload,
        },
    )
    job = parallel.WorkerJob(
        0,
        "simulation",
        {"scope": "minimal", "seed": 11},
    )

    with parallel.SerialRunner(
        TinyContext(379, "serial"),
        tmp_path,
    ) as runner:
        outcomes = list(runner.execute([job]))

    assert initialized == [True]
    assert outcomes[0].worker_pid == os.getpid()
    assert outcomes[0].result == {
        "context": {"n_regions": 379, "label": "serial"},
        "payload": {"scope": "minimal", "seed": 11},
    }
    assert (tmp_path / ".runtime" / "serial" / "tvb").is_dir()


def test_worker_job_validates_and_plainly_copies_payload() -> None:
    payload = {"seed": 11}
    job = parallel.WorkerJob(2, "simulation", payload)
    payload["seed"] = 23

    assert job.payload == {"seed": 11}
    with pytest.raises(ValueError, match="ordinal"):
        parallel.WorkerJob(-1, "simulation", {})
    with pytest.raises(ValueError, match="kind"):
        parallel.WorkerJob(0, "unknown", {})
    with pytest.raises(TypeError, match="payload"):
        parallel.WorkerJob(0, "simulation", [])  # type: ignore[arg-type]


def test_parallel_runner_reuses_workers_and_yields_completion_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = install_fake_runtime(monkeypatch)
    outcomes = [
        parallel.WorkerOutcome(2, "simulation", "second", 102, 2.0),
        parallel.WorkerOutcome(1, "simulation", "first", 101, 1.0),
    ]
    jobs = [
        parallel.WorkerJob(1, "simulation", {"seed": 11}),
        parallel.WorkerJob(2, "simulation", {"seed": 23}),
    ]

    with parallel.ParallelRunner(
        TinyContext(379, "test"),
        tmp_path,
        worker_count=2,
        handlers=(logging.NullHandler(),),
    ) as runner:
        for outcome in outcomes:
            fake_context.result_queue.put(
                parallel._WorkerMessage(
                    kind="outcome",
                    worker_index=0,
                    worker_pid=outcome.worker_pid,
                    ordinal=outcome.ordinal,
                    payload=outcome,
                )
            )
        assert list(runner.execute(jobs)) == outcomes
        assert list(runner.execute([])) == []

    assert len(fake_context.processes) == 2
    assert all(
        process.target is parallel._worker_main
        for process in fake_context.processes
    )
    assert fake_context.processes[0].args[1] == {
        "n_regions": 379,
        "label": "test",
    }
    submitted = [
        item
        for item in fake_context.job_queue.items
        if isinstance(item, parallel.WorkerJob)
    ]
    assert submitted == jobs
    assert all(process.exitcode == 0 for process in fake_context.processes)
    assert all(process.terminated is False for process in fake_context.processes)
    assert all(managed.closed for managed in fake_context.queues)
    assert all(managed.joined for managed in fake_context.queues)
    assert FakeListener.instances[0].started is True
    assert FakeListener.instances[0].stopped is True


def test_initializer_failure_surfaces_and_terminates_other_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = worker_failure(
        "TVB import failed",
        phase="initialization",
    )
    fake_context = install_fake_runtime(
        monkeypatch,
        initialization_failures={0: failure},
    )

    with pytest.raises(RuntimeError, match="TVB import failed"):
        with parallel.ParallelRunner(
            {"n_regions": 379},
            tmp_path,
            worker_count=2,
            handlers=(logging.NullHandler(),),
        ):
            pass

    assert fake_context.processes[0].exitcode == 1
    assert fake_context.processes[1].terminated is True
    assert all(managed.closed for managed in fake_context.queues)


def test_job_failure_surfaces_and_terminates_all_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = install_fake_runtime(monkeypatch)
    job = parallel.WorkerJob(0, "calibration", {"candidate_g": 60.0})

    with pytest.raises(RuntimeError, match="worker failed"):
        with parallel.ParallelRunner(
            {"n_regions": 379},
            tmp_path,
            worker_count=2,
            handlers=(logging.NullHandler(),),
        ) as runner:
            fake_context.result_queue.put(
                parallel._WorkerMessage(
                    kind="job_error",
                    worker_index=0,
                    worker_pid=10_000,
                    ordinal=0,
                    payload=worker_failure("worker failed"),
                )
            )
            list(runner.execute([job]))

    assert all(process.terminated for process in fake_context.processes)
    assert fake_context.job_queue.join_cancelled is True


def test_abrupt_worker_death_is_detected_while_job_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = install_fake_runtime(monkeypatch)
    job = parallel.WorkerJob(0, "simulation", {"seed": 11})

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        with parallel.ParallelRunner(
            {"n_regions": 379},
            tmp_path,
            worker_count=2,
            handlers=(logging.NullHandler(),),
        ) as runner:
            outcomes = runner.execute([job])
            fake_context.processes[0].die()
            list(outcomes)

    assert fake_context.processes[1].terminated is True


def test_parallel_runner_terminates_on_body_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = install_fake_runtime(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        with parallel.ParallelRunner(
            {"n_regions": 379},
            tmp_path,
            worker_count=1,
            handlers=(logging.NullHandler(),),
        ):
            raise KeyboardInterrupt

    assert fake_context.processes[0].terminated is True


def test_runner_rejects_duplicate_ordinals_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_context = install_fake_runtime(monkeypatch)
    jobs = [
        parallel.WorkerJob(4, "simulation", {}),
        parallel.WorkerJob(4, "calibration", {}),
    ]

    with parallel.ParallelRunner(
        {"n_regions": 379},
        tmp_path,
        worker_count=1,
        handlers=(logging.NullHandler(),),
    ) as runner:
        with pytest.raises(ValueError, match="ordinals"):
            runner.execute(jobs)

    assert not any(
        isinstance(item, parallel.WorkerJob)
        for item in fake_context.job_queue.items
    )
