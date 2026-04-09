"""
main_worker.py
──────────────
Orchestrates all worker processes:
  • One camera_worker per configured camera source.
  • One gpu_worker (single process; batches internally).
  • One db_worker.

Uses multiprocessing so each worker is isolated and restartable.
A supervisor loop monitors child processes and restarts them on failure.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

RESTART_DELAY = 3   # seconds before restarting a failed worker
MAX_RESTARTS = 20   # per worker before giving up


# ── Worker entry-points ────────────────────────────────────────────────────────

def _run_camera(camera_id: str, source: str) -> None:
    from workers.camera_worker import run
    run(camera_id, source)


def _run_gpu() -> None:
    from workers.gpu_worker import run
    run()


def _run_db() -> None:
    from workers.db_worker import run
    run()


# ── Process descriptor ─────────────────────────────────────────────────────────

class WorkerSpec:
    def __init__(self, name: str, target: Callable, args: tuple = ()):
        self.name = name
        self.target = target
        self.args = args
        self.process: mp.Process | None = None
        self.restarts = 0

    def start(self) -> None:
        self.process = mp.Process(
            target=self.target,
            args=self.args,
            name=self.name,
            daemon=True,
        )
        self.process.start()
        log.info("worker_started", name=self.name, pid=self.process.pid)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def restart(self) -> bool:
        if self.restarts >= MAX_RESTARTS:
            log.error("worker_max_restarts", name=self.name)
            return False
        self.restarts += 1
        log.warning("worker_restarting", name=self.name, attempt=self.restarts)
        time.sleep(RESTART_DELAY)
        self.start()
        return True

    def terminate(self) -> None:
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            log.info("worker_terminated", name=self.name)


# ── Main supervisor ────────────────────────────────────────────────────────────

_workers: List[WorkerSpec] = []
_shutdown_flag = False


def _on_signal(sig, frame):
    global _shutdown_flag
    log.info("supervisor_shutdown_signal")
    _shutdown_flag = True


def build_worker_specs() -> List[WorkerSpec]:
    specs: List[WorkerSpec] = []

    # One camera worker per source
    for idx, source in enumerate(settings.camera_source_list):
        cam_id = f"cam_{idx:02d}"
        specs.append(WorkerSpec(f"camera:{cam_id}", _run_camera, (cam_id, source)))

    # GPU worker
    specs.append(WorkerSpec("gpu_worker", _run_gpu))

    # DB worker
    specs.append(WorkerSpec("db_worker", _run_db))

    return specs


def run_supervisor() -> None:
    configure_logging(settings.worker_log_level)
    log.info("supervisor_start", cameras=settings.camera_source_list)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    global _workers
    _workers = build_worker_specs()
    for w in _workers:
        w.start()

    while not _shutdown_flag:
        for w in _workers:
            if not w.is_alive():
                log.warning("worker_died", name=w.name)
                if not w.restart():
                    log.error("worker_disabled", name=w.name)
        time.sleep(2)

    log.info("supervisor_stopping_all")
    for w in _workers:
        w.terminate()
    log.info("supervisor_done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_supervisor()
