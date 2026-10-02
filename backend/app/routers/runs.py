import io
import json
import shutil
import tempfile
import zipfile
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from app import colab
from app.config import BACKEND_DIR, get_settings
from app.db import get_db
from app.models import Project, Run, User, utcnow
from app.routers.projects import owned_project, safe_filename
from app.schemas import ColabResultIn, RunCreate, RunOut
from app.security import get_current_user, new_run_token
from app.services import run_out, upsert_colab_result, validate_model_selection
from app.shared_data import IMAGE_EXTS, load_table
from app.storage import run_dir, safe_extract_zip
from app.training.runner import finalize

router = APIRouter(prefix="/api", tags=["runs"])
ACTIVE = ("queued", "running", "cancelling", "awaiting_colab")


def owned_run(run_id: str, user: User, db: Session) -> Run:
    run = db.get(Run, run_id)
    if not run or run.project.user_id != user.id:
        raise HTTPException(404, "Run not found")
    return run


@router.post("/projects/{project_id}/runs", response_model=RunOut)
def create_run(project_id: str, body: RunCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = owned_project(project_id, user, db)
    if p.status != "ready" or not p.spec:
        raise HTTPException(409, "The dataset is not analysed yet.")
    settings = get_settings()
    active = (db.query(Run).join(Project).filter(Project.user_id == user.id, Run.status.in_(ACTIVE[:3])).count())
    if body.backend == "cpu" and active >= settings.max_active_runs_per_user:
        raise HTTPException(429, f"You already have {active} trainings running. Wait for one to finish or cancel it.")
    task = p.spec["task"]
    try:
        keys = validate_model_selection(task, body.backend, body.model_keys, body.custom_models)
    except ValueError as e:
        raise HTTPException(422, str(e))
    config = body.model_dump()
    config["model_keys"] = keys
    config["max_rows"] = settings.max_tabular_rows
    config["max_images"] = body.max_images or settings.max_cpu_images
    run = Run(project_id=p.id, backend=body.backend, config=config, spec=p.spec,
              status="queued" if body.backend == "cpu" else "awaiting_colab",
              progress={"stage": "queued", "message": "Waiting for a training worker" if body.backend == "cpu"
                        else "Download the notebook and run it in Google Colab"})
    db.add(run)
    db.commit()
    return run_out(run)


@router.get("/projects/{project_id}/runs", response_model=list[RunOut])
def list_runs(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return [run_out(r) for r in owned_project(project_id, user, db).runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return run_out(owned_run(run_id, user, db))


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = owned_run(run_id, user, db)
    if run.status == "queued" or run.backend == "colab" and run.status in ACTIVE:
        run.status, run.finished_at = "cancelled", utcnow()
    elif run.status == "running":
        run.status = "cancelling"
        run.progress = {**(run.progress or {}), "message": "Stopping after the models in progress…"}
    db.commit()
    return run_out(run)


@router.get("/runs/{run_id}/notebook")
def download_notebook(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Each download issues a fresh run token (older notebook copies stop being able to report)."""
    run = owned_run(run_id, user, db)
    if run.backend != "colab":
        raise HTTPException(400, "This run trains on the server, not in Colab.")
    settings = get_settings()
    token, token_hash = new_run_token()
    run.colab_token_hash = token_hash
    run.colab_token_expires = utcnow() + timedelta(days=settings.colab_token_days)
    db.commit()
    p, c, spec = run.project, run.config, run.spec
    common = dict(run_id=run.id, token=token, api_base=settings.public_api_url, dataset_filename=p.dataset_filename,
                  model_keys=c["model_keys"], epochs=c.get("epochs", 10),
                  time_limit_min=max(1, int(c.get("time_limit_per_model", 1800)) // 60))
    if spec["task"] == "image_classification":
        nb = colab.image_classification_notebook(custom=c.get("custom_models", []), **common)
    elif spec["task"] == "text_classification":
        nb = colab.text_classification_notebook(custom=c.get("custom_models", []), text_column=spec["text_column"],
                                                target=spec["target"], **common)
    elif spec["task"] == "object_detection":
        nb = colab.object_detection_notebook(**common)
    else:
        raise HTTPException(400, "This task runs on CPU.")
    name = f"{safe_filename(p.name)}_{spec['task']}_colab.ipynb"
    return Response(json.dumps(nb, indent=1), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/runs/{run_id}/import-results", response_model=RunOut)
def import_results(run_id: str, file: UploadFile = File(...), user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Manual mode: upload the results.json the Colab notebook produced."""
    run = owned_run(run_id, user, db)
    if run.backend != "colab":
        raise HTTPException(400, "Only Colab runs accept imported results.")
    try:
        data = json.loads(file.file.read(5 * 1024 * 1024))
        if data.get("run_id") not in (None, run.id):
            raise HTTPException(400, "This results.json belongs to a different run.")
        items = [ColabResultIn(**r) for r in data.get("results", [])]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Not a valid results.json: {e}")
    allowed = set(run.config.get("model_keys", [])) | {
        f"{run.spec['task']}/colab/{c}" for c in run.config.get("custom_models", [])}
    for item in items:
        if item.model_key in allowed:
            upsert_colab_result(db, run, item)
    db.commit()
    finalize(run.id, status="completed")
    db.expire_all()
    return run_out(run)


@router.post("/runs/{run_id}/finish", response_model=RunOut)
def finish_colab(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = owned_run(run_id, user, db)
    if run.backend != "colab":
        raise HTTPException(400, "Only Colab runs can be finished manually.")
    finalize(run.id, status="completed")
    db.expire_all()
    return run_out(run)


# --------------------------------------------------------------------------------------
# Deliverables
# --------------------------------------------------------------------------------------

_PREDICT_EXAMPLE = '''"""Use the trained model.

pip install scikit-learn pandas joblib {extra}
Keep automl_runtime.py next to this script.
"""
import joblib
import pandas as pd

model = joblib.load("model.joblib")
{usage}
'''


@router.get("/runs/{run_id}/download")
def download(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = owned_run(run_id, user, db)
    task = run.spec["task"]
    best = next((r for r in run.results if r.is_best), None)
    if run.status != "completed" or not best:
        raise HTTPException(409, "No finished model to download.")
    rd = run_dir(run.project_id, run.id)
    if task == "forecasting":
        dates = (run.insights or {}).get("series", {}).get("future_dates", [])
        preds = best.details.get("future_pred", [])
        df = pd.DataFrame({"date": dates[:len(preds)], "forecast": preds, "model": best.name})
        return Response(df.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="forecast.csv"'})
    if task == "clustering":
        from app.training.tabular import export_clusters

        out = export_clusters(run.project.dataset_path, rd, run.best_model_path, run.config.get("max_rows", 200_000))
        return FileResponse(out, filename="clusters.csv", media_type="text/csv")
    if not run.best_model_path or not Path(run.best_model_path).exists():
        raise HTTPException(404, "The model file is not available (Colab models stay in your Colab session).")

    extra = {"image_classification": "torch timm pillow", "text_classification": "sentence-transformers"}.get(task, "")
    if task == "image_classification":
        usage = 'print(model.predict(["photo1.jpg", "photo2.jpg"]))'
    elif task == "text_classification":
        usage = f'df = pd.DataFrame({{{run.spec["text_column"]!r}: ["some text to classify"]}})\nprint(model.predict(df))'
    else:
        usage = 'df = pd.read_csv("new_data.csv")  # same columns as the training data (target not needed)\nprint(model.predict(df))'
    meta = {"task": task, "model": best.name, "model_key": best.model_key, "metrics": best.metrics,
            "primary_metric": run.spec["metric"], "spec": run.spec, "trained_at": str(run.finished_at),
            "features": (run.insights or {}).get("data", {}).get("features")}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run.best_model_path, "model.joblib")
        z.write(BACKEND_DIR / "automl_runtime.py", "automl_runtime.py")
        z.writestr("predict_example.py", _PREDICT_EXAMPLE.format(extra=extra, usage=usage))
        z.writestr("metadata.json", json.dumps(meta, indent=2, default=str))
    buf.seek(0)
    name = f"{safe_filename(run.project.name)}_best_model.zip"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/runs/{run_id}/predict")
def predict(run_id: str, file: UploadFile = File(...), user: User = Depends(get_current_user),
            db: Session = Depends(get_db)):
    """Upload new data, get predictions from the best model as CSV."""
    run = owned_run(run_id, user, db)
    task = run.spec["task"]
    if task not in ("classification", "regression", "text_classification", "image_classification"):
        raise HTTPException(400, "Predictions on new data are available for classification and regression tasks.")
    if not run.best_model_path or not Path(run.best_model_path).exists():
        raise HTTPException(409, "No trained model on the server for this run.")
    model = joblib.load(run.best_model_path)
    suffix = Path(file.filename or "x").suffix.lower()
    with tempfile.TemporaryDirectory(dir=get_settings().data_dir) as tmp:
        src = Path(tmp) / f"input{suffix}"
        with open(src, "wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
        try:
            if task == "image_classification":
                if suffix == ".zip":
                    root = safe_extract_zip(src, Path(tmp) / "x")
                    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
                elif suffix in IMAGE_EXTS:
                    paths = [src]
                else:
                    raise HTTPException(400, "Upload an image or a ZIP of images.")
                if not paths:
                    raise HTTPException(400, "No images found.")
                preds = model.predict([str(p) for p in paths[:5000]])
                df = pd.DataFrame({"file": [p.name if suffix != ".zip" else p.relative_to(root).as_posix() for p in paths[:5000]],
                                   "prediction": preds})
            else:
                df = load_table(src)
                df["prediction"] = model.predict(df)
        except HTTPException:
            raise
        except KeyError as e:
            raise HTTPException(422, f"The file is missing a column the model needs: {e}")
        except Exception as e:
            raise HTTPException(422, f"Prediction failed: {type(e).__name__}: {e}")
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="predictions.csv"'})
