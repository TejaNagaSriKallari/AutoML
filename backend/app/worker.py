"""Job worker: claims queued CPU runs from the database and executes them.

Run standalone with `python -m app.worker` (scale out by starting more workers), or inside the
API process when RUN_WORKER_IN_API=true. The database row is the queue: a run is claimed with an
atomic compare-and-set on its status, so several workers never pick the same run.
"""
import logging
import os
import socket
import threading
import time
from datetime import timedelta, timezone

from sqlalchemy import update

from app.db import SessionLocal, init_db
from app.models import Run, utcnow

log = logging.getLogger("worker")
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
HEARTBEAT_SECONDS = 20
STALE_AFTER_SECONDS = 180


def claim_next() -> str | None:
    with SessionLocal() as db:
        candidate = (db.query(Run.id).filter(Run.status == "queued", Run.backend == "cpu")
                     .order_by(Run.created_at).first())
        if not candidate:
            return None
        res = db.execute(
            update(Run).where(Run.id == candidate.id, Run.status == "queued")
            .values(status="running", worker_id=WORKER_ID, started_at=utcnow(), heartbeat_at=utcnow(),
                    progress={"stage": "starting", "message": "Worker picked up the run"})
        )
        db.commit()
        return candidate.id if res.rowcount == 1 else None


def recover_stale() -> None:
    """Runs whose worker stopped sending heartbeats (crash, restart) are marked failed."""
    cutoff = utcnow() - timedelta(seconds=STALE_AFTER_SECONDS)
    with SessionLocal() as db:
        stale = db.query(Run).filter(Run.status.in_(["running", "cancelling"]), Run.backend == "cpu").all()
        for r in stale:
            beat = r.heartbeat_at or r.started_at
            if beat is not None and beat.tzinfo is None:
                beat = beat.replace(tzinfo=timezone.utc)
            if beat is None or beat < cutoff:
                r.status = "failed"
                r.error = "The training worker stopped while this run was in progress. Please start it again."
                r.finished_at = utcnow()
        db.commit()


def _heartbeat(run_id: str, done: threading.Event) -> None:
    while not done.wait(HEARTBEAT_SECONDS):
        try:
            with SessionLocal() as db:
                db.execute(update(Run).where(Run.id == run_id).values(heartbeat_at=utcnow()))
                db.commit()
        except Exception:
            log.exception("Heartbeat failed")


def loop(stop: threading.Event | None = None, poll_seconds: float = 2.0) -> None:
    from app.training.runner import execute_run

    log.info("Worker %s started", WORKER_ID)
    while not (stop and stop.is_set()):
        run_id = None
        try:
            run_id = claim_next()
        except Exception:
            log.exception("Could not poll for runs")
        if run_id:
            log.info("Executing run %s", run_id)
            done = threading.Event()
            threading.Thread(target=_heartbeat, args=(run_id, done), daemon=True).start()
            try:
                execute_run(run_id)
            finally:
                done.set()
        else:
            try:
                recover_stale()
            except Exception:
                log.exception("Stale-run check failed")
            time.sleep(poll_seconds)


def start_in_background() -> threading.Event:
    stop = threading.Event()
    threading.Thread(target=loop, args=(stop,), name="training-worker", daemon=True).start()
    return stop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    recover_stale()
    loop()
