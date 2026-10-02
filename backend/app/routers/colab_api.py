"""Endpoints called by Colab notebooks, authenticated with the run-scoped token (not the user session)."""
import hmac

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Run, utcnow
from app.schemas import ColabResultIn, ColabStatusIn
from app.security import hash_run_token
from app.services import upsert_colab_result
from app.training.runner import finalize

router = APIRouter(prefix="/api/colab", tags=["colab"])


def token_run(run_id: str, x_run_token: str = Header(...), db: Session = Depends(get_db)) -> Run:
    run = db.get(Run, run_id)
    if not run or run.backend != "colab" or not run.colab_token_hash:
        raise HTTPException(404, "Run not found")
    if not hmac.compare_digest(run.colab_token_hash, hash_run_token(x_run_token)):
        raise HTTPException(401, "Invalid run token. Download a fresh notebook from the app.")
    expires = run.colab_token_expires
    if expires is not None and expires.tzinfo is None:
        from datetime import timezone

        expires = expires.replace(tzinfo=timezone.utc)
    if expires and expires < utcnow():
        raise HTTPException(401, "This notebook's token expired. Download a fresh notebook from the app.")
    if run.status in ("cancelled",):
        raise HTTPException(409, "This run was cancelled in the app.")
    return run


@router.get("/runs/{run_id}/dataset")
def dataset(run: Run = Depends(token_run)):
    p = run.project
    return FileResponse(p.upload_path, filename=p.dataset_filename)


@router.post("/runs/{run_id}/status")
def status(body: ColabStatusIn, run: Run = Depends(token_run), db: Session = Depends(get_db)):
    if body.status == "completed":
        db.commit()
        finalize(run.id, status="completed")
        return {"ok": True}
    if body.status == "failed":
        run.status, run.error, run.finished_at = "failed", body.message, utcnow()
    else:
        if run.status in ("awaiting_colab", "completed", "failed"):
            run.status = "running"
            run.started_at = run.started_at or utcnow()
            run.finished_at = None
        run.progress = {"stage": "colab", "message": body.message, "done": body.done, "total": body.total}
    db.commit()
    return {"ok": True}


@router.post("/runs/{run_id}/results")
def result(body: ColabResultIn, run: Run = Depends(token_run), db: Session = Depends(get_db)):
    allowed = set(run.config.get("model_keys", [])) | {
        f"{run.spec['task']}/colab/{c}" for c in run.config.get("custom_models", [])}
    if body.model_key not in allowed:
        raise HTTPException(400, "This model was not part of the run.")
    upsert_colab_result(db, run, body)
    db.commit()
    return {"ok": True}
