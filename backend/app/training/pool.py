"""Runs model-training jobs in parallel worker processes with a hard time limit per job.

A model that exceeds its limit is killed (its process is replaced) instead of stalling the
whole run - essential when trying dozens of models whose cost we cannot predict exactly.
If a worker dies (e.g. out of memory), only the job it was running fails: the pool is rebuilt
and the other in-flight jobs are retried.
"""
from __future__ import annotations

import os
import traceback
from concurrent.futures import FIRST_COMPLETED, TimeoutError, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable

from pebble import ProcessPool
from pebble.common import ProcessExpired

from app.config import get_settings

# Parallelism comes from processes; multi-threaded BLAS inside each one only multiplies memory
# use and oversubscribes cores. Children inherit these at spawn time.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

GB = 1024**3
MEMORY_PER_WORKER = 0.6 * GB  # single-threaded BLAS keeps a worker well under this
MAX_POOL_RESTARTS = 3


@dataclass
class Job:
    job_id: str
    fn: Callable
    args: tuple
    timeout: float


def default_workers() -> int:
    configured = get_settings().worker_processes
    if configured > 0:
        return configured
    by_cpu = max(1, (os.cpu_count() or 2) // 2)
    available = _available_memory()
    if available is None:
        return by_cpu
    return max(1, min(by_cpu, int((available - 1.0 * GB) // MEMORY_PER_WORKER)))


def _available_memory() -> int | None:
    """Physical memory available, capped on Windows by the free commit charge (RAM + page file),
    which is what actually runs out when the page file cannot grow."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        return None
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            available = min(available, stat.ullAvailPageFile)
    return available


def run_jobs(
    jobs: list[Job],
    on_result: Callable[[str, dict | None, str | None, str], None],
    should_cancel: Callable[[], bool] = lambda: False,
    workers: int | None = None,
) -> bool:
    """Calls on_result(job_id, result, error, status) as each job finishes.
    status is 'completed' | 'failed' | 'timeout'. Returns False if cancelled."""
    queue = list(jobs)
    restarts = 0
    while queue:
        n = max(1, min(workers or default_workers(), len(queue)))
        retry, cancelled = _run_batch(queue, on_result, should_cancel, n)
        if cancelled:
            return False
        if retry:
            restarts += 1
            if restarts > MAX_POOL_RESTARTS:
                for job in retry:
                    on_result(job.job_id, None, "Training workers kept crashing (the server may be low on memory).", "failed")
                return True
            workers = max(1, n // 2)  # be gentler after a crash
        queue = retry
    return True


def _run_batch(jobs, on_result, should_cancel, workers) -> tuple[list[Job], bool]:
    pool = ProcessPool(max_workers=workers)
    pending: dict[Any, Job] = {}
    retry: list[Job] = []
    try:
        for job in jobs:
            pending[pool.schedule(job.fn, args=job.args, timeout=job.timeout)] = job
        while pending:
            done, _ = wait(list(pending), timeout=2.0, return_when=FIRST_COMPLETED)
            if should_cancel():
                for f in pending:
                    f.cancel()
                return [], True
            for fut in done:
                job = pending.pop(fut)
                result, error, status = None, None, "completed"
                try:
                    result = fut.result()
                except TimeoutError:
                    error, status = "Exceeded the time limit for one model.", "timeout"
                except ProcessExpired as e:
                    error, status = f"The worker process died while training this model (likely out of memory): {e}", "failed"
                except BrokenProcessPool:
                    retry.append(job)  # collateral damage from another job's crash: try again
                    continue
                except Exception as e:  # the model raised inside the worker
                    tb = getattr(e, "traceback", None) or traceback.format_exc()
                    error, status = f"{type(e).__name__}: {e}\n{_tail(tb)}", "failed"
                on_result(job.job_id, result, error, status)
        return retry, False
    finally:
        pool.stop()
        pool.join(timeout=10)


def _tail(tb: str, lines: int = 6) -> str:
    return "\n".join(str(tb).strip().splitlines()[-lines:])
