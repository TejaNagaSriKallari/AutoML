"""Executes one CPU run end to end and finalises the leaderboard."""
from __future__ import annotations

import logging
import os
import traceback

from app.db import SessionLocal
from app.models import ModelResult, Project, Run, utcnow
from app.storage import run_dir
from app.training.context import RunContext
from app.training.evaluate import rank_results
from app.understanding import explain_results

log = logging.getLogger(__name__)


def execute_run(run_id: str) -> None:
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        project = db.get(Project, run.project_id)
        ctx = RunContext(run, project.dataset_kind, project.dataset_path, project.profile, run_dir(project.id, run.id))
        model_keys = list(run.config.get("model_keys", []))

    task = ctx.spec["task"]
    try:
        if task in ("classification", "regression"):
            from app.training.tabular import run_supervised
            run_supervised(ctx, model_keys)
        elif task == "clustering":
            from app.training.tabular import run_clustering
            run_clustering(ctx, model_keys)
        elif task == "forecasting":
            from app.training.forecasting import run_forecasting
            run_forecasting(ctx, model_keys)
        elif task == "text_classification":
            from app.training.text import run_text
            run_text(ctx, model_keys)
        elif task == "image_classification":
            from app.training.image import run_image
            run_image(ctx, model_keys)
        else:
            raise ValueError(f"Task '{task}' cannot run on CPU; use the Colab GPU option.")
    except Exception as e:
        log.exception("Run %s failed", run_id)
        _finish(run_id, "failed", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
        return
    finalize(run_id, ctx.model_paths, cancelled=ctx.cancelled())


def finalize(run_id: str, model_paths: dict[str, str] | None = None, cancelled: bool = False,
             status: str | None = None) -> None:
    """Ranks results, marks the best model, keeps only its file, writes the summary."""
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        metric = run.spec["metric"]
        rows = db.query(ModelResult).filter(ModelResult.run_id == run_id).all()
        dicts = [{"id": r.id, "status": r.status, "primary_score": r.primary_score, "cv_mean": r.cv_mean} for r in rows]
        ranked = rank_results(dicts, metric)
        rank_by_id = {d["id"]: d["rank"] for d in ranked}
        best_row = None
        for r in rows:
            r.rank = rank_by_id.get(r.id)
            r.is_best = r.rank == 1
            if r.is_best:
                best_row = r
        if model_paths is not None:
            best_path = model_paths.get(best_row.model_key) if best_row else None
            for key, path in model_paths.items():
                if path != best_path and os.path.exists(path):
                    os.remove(path)
            run.best_model_path = best_path
        leaderboard = [{"name": r.name, "score": r.primary_score, "cv_mean": r.cv_mean,
                        "metrics": r.metrics, "train_seconds": r.train_seconds}
                       for r in sorted((r for r in rows if r.rank), key=lambda r: r.rank)]
        run.summary = explain_results(run.spec["task"], metric, leaderboard, run.insights or {})
        failed = sum(1 for r in rows if r.status != "completed")
        run.status = status or ("cancelled" if cancelled else "completed" if leaderboard else "failed")
        if not leaderboard and not cancelled:
            run.error = run.error or "No model finished successfully."
        run.progress = {"stage": "done", "message": f"{len(leaderboard)} models ranked, {failed} failed or skipped",
                        "done": len(rows), "total": len(rows)}
        run.finished_at = utcnow()
        db.commit()


def _finish(run_id: str, status: str, error: str | None = None, tb: str | None = None) -> None:
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        run.status = status
        run.error = error
        run.finished_at = utcnow()
        run.progress = {**(run.progress or {}), "stage": "done", "message": error or status}
        if tb:
            run.insights = {**(run.insights or {}), "traceback": tb[-4000:]}
        db.commit()
