"""Business logic shared by the API routes."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import ModelResult, Project, Run, utcnow
from app.profiling import DatasetError, analyze_upload
from app.registry import TASKS, all_models, candidates
from app.schemas import ColabResultIn, ModelResultOut, ProjectOut, RunOut, TaskSpec
from app.storage import UnsafeArchive, project_dir
from app.training.context import _clean
from app.training.evaluate import higher_is_better, rank_results
from app.understanding import infer_spec

log = logging.getLogger(__name__)
_HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*(/[\w.-]+)?$")


def analyze_project(project_id: str) -> None:
    """Background task: profile the upload and infer the task spec."""
    with SessionLocal() as db:
        p = db.get(Project, project_id)
        if not p:
            return
        try:
            kind, data_path, profile = analyze_upload(Path(p.upload_path), project_dir(p.id))
            spec = infer_spec(p.requirement, kind, profile)
            p.dataset_kind, p.dataset_path, p.profile = kind, str(data_path), profile
            p.spec, p.status, p.error = spec.model_dump(), "ready", None
        except (DatasetError, UnsafeArchive) as e:
            p.status, p.error = "failed", str(e)
        except Exception as e:
            log.exception("Analysis failed for %s", project_id)
            p.status, p.error = "failed", f"Could not analyse the dataset: {type(e).__name__}: {e}"
        db.commit()


def project_out(p: Project, include_profile: bool = True) -> ProjectOut:
    latest = p.runs[0] if p.runs else None
    return ProjectOut(
        id=p.id, name=p.name, requirement=p.requirement, dataset_filename=p.dataset_filename,
        dataset_kind=p.dataset_kind, dataset_size_bytes=p.dataset_size_bytes, status=p.status, error=p.error,
        profile=p.profile if include_profile else None, spec=p.spec, spec_confirmed=p.spec_confirmed,
        created_at=p.created_at,
        latest_run={"id": latest.id, "status": latest.status, "backend": latest.backend,
                    "created_at": latest.created_at.isoformat()} if latest else None,
    )


def run_out(run: Run) -> RunOut:
    metric = (run.spec or {}).get("metric", "accuracy")
    rows = list(run.results)
    live_rank = {d["id"]: d["rank"] for d in rank_results(
        [{"id": r.id, "status": r.status, "primary_score": r.primary_score, "cv_mean": r.cv_mean} for r in rows], metric)}
    results = []
    for r in rows:
        rank = r.rank if r.rank is not None else live_rank.get(r.id)
        results.append(ModelResultOut(
            id=r.id, model_key=r.model_key, name=r.name, family=r.family, status=r.status, metrics=r.metrics or {},
            primary_score=r.primary_score, cv_mean=r.cv_mean, cv_std=r.cv_std, train_seconds=r.train_seconds,
            params=r.params or {}, details=r.details or {}, error=r.error, rank=rank, is_best=r.is_best or rank == 1,
        ))
    results.sort(key=lambda x: (x.rank is None, x.rank or 0, x.name))
    return RunOut(
        id=run.id, project_id=run.project_id, backend=run.backend, status=run.status, config=run.config or {},
        spec=run.spec or {}, progress=run.progress or {}, insights=run.insights or {}, summary=run.summary,
        error=run.error, has_model=bool(run.best_model_path) or run.spec.get("task") in ("forecasting", "clustering") and run.status == "completed",
        created_at=run.created_at, started_at=run.started_at, finished_at=run.finished_at, results=results,
    )


def candidate_lists(p: Project) -> dict:
    spec = p.spec or {}
    cands = candidates(spec, p.profile or {})
    by_backend: dict[str, list[dict]] = {"cpu": [], "colab": []}
    for c in cands:
        by_backend[c["backend"]].append(c)
    return {"task": spec.get("task"), "task_info": TASKS.get(spec.get("task"), {}), "backends": by_backend}


def validate_model_selection(task: str, backend: str, keys: list[str], custom: list[str]) -> list[str]:
    models = all_models()
    if backend not in TASKS[task]["backends"]:
        raise ValueError(f"{TASKS[task]['label']} cannot run on {'CPU' if backend == 'cpu' else 'Colab'}.")
    clean = []
    for k in dict.fromkeys(keys):
        m = models.get(k)
        if not m or m.task != task or m.backend != backend:
            raise ValueError(f"Model '{k}' does not apply to this task/backend.")
        clean.append(k)
    for c in custom:
        c = c.strip()
        if not c:
            continue
        if task == "image_classification":
            try:
                import timm
            except ImportError:  # lite install without PyTorch: the Colab notebook validates the name
                timm = None
            if timm and c not in timm.list_models(pretrained=True):
                raise ValueError(f"'{c}' is not a pretrained timm model name (see timm.list_models(pretrained=True)).")
            if backend == "cpu":
                clean.append(f"image_classification/cpu/backbone:{c}")
        elif task == "text_classification":
            if not _HF_ID.match(c):
                raise ValueError(f"'{c}' is not a valid Hugging Face model id.")
            if backend == "cpu":
                clean.append(f"text_classification/cpu/enc:{c}")
        else:
            raise ValueError("Custom models are supported for image and text classification.")
    if backend == "cpu" and task in ("image_classification", "text_classification"):
        has_backbone = any("/backbone:" in k or "/enc:" in k for k in clean)
        has_head = any("/head:" in k for k in clean)
        if not (has_backbone and has_head):
            raise ValueError("Pick at least one feature extractor and one classifier head.")
    if not clean and not custom:
        raise ValueError("Select at least one model.")
    return clean


def upsert_colab_result(db: Session, run: Run, res: ColabResultIn) -> None:
    metric = run.spec["metric"]
    score = res.metrics.get(metric)
    if score is None and res.status == "completed":
        score = res.metrics.get("accuracy") if higher_is_better(metric) else None
    row = (db.query(ModelResult).filter(ModelResult.run_id == run.id, ModelResult.model_key == res.model_key).first())
    if row is None:
        row = ModelResult(run_id=run.id, model_key=res.model_key)
        db.add(row)
    row.name = res.name
    row.family = "finetune" if run.spec["task"] != "object_detection" else "detector"
    row.status = res.status
    row.metrics = _clean(res.metrics)
    row.primary_score = _clean(score)
    row.train_seconds = res.train_seconds
    row.params = _clean(res.params)
    row.details = _clean(res.details)
    row.error = res.error
    if run.status == "awaiting_colab":
        run.status = "running"
        run.started_at = run.started_at or utcnow()


def spec_from_update(data: dict) -> TaskSpec:
    return TaskSpec(**data, source="user", confidence=1.0)
