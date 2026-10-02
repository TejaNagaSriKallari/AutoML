import re
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Project, User
from app.registry import catalog
from app.schemas import ProjectOut, SpecUpdate
from app.security import get_current_user
from app.services import analyze_project, candidate_lists, project_out, spec_from_update
from app.shared_data import TABLE_EXTS
from app.storage import delete_project_files, project_dir
from app.understanding import validate_spec

router = APIRouter(prefix="/api", tags=["projects"])
_ALLOWED = TABLE_EXTS | {".zip"}


def owned_project(project_id: str, user: User, db: Session) -> Project:
    p = db.get(Project, project_id)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "Project not found")
    return p


@router.get("/catalog")
def get_catalog():
    return catalog()


@router.post("/projects", response_model=ProjectOut)
def create_project(
    background: BackgroundTasks,
    name: str = Form(..., max_length=200),
    requirement: str = Form(..., min_length=5, max_length=5000),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filename = Path(file.filename or "dataset").name
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Upload CSV, Excel, Parquet, JSON or a ZIP.")
    project = Project(user_id=user.id, name=name.strip() or filename, requirement=requirement.strip(),
                      dataset_filename=filename, upload_path="", dataset_kind="unknown", status="analyzing")
    db.add(project)
    db.flush()
    dest = project_dir(project.id) / ("upload" + ext)
    limit = get_settings().max_upload_mb * 1024 * 1024
    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                delete_project_files(project.id)
                db.rollback()
                raise HTTPException(413, f"File is larger than {get_settings().max_upload_mb} MB.")
            out.write(chunk)
    project.upload_path = str(dest)
    project.dataset_size_bytes = size
    db.commit()
    background.add_task(analyze_project, project.id)
    return project_out(project)


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Project).filter(Project.user_id == user.id).order_by(Project.created_at.desc()).all()
    return [project_out(p, include_profile=False) for p in rows]


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return project_out(owned_project(project_id, user, db))


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = owned_project(project_id, user, db)
    if any(r.status in ("queued", "running", "cancelling") for r in p.runs):
        raise HTTPException(409, "Cancel the running training before deleting this project.")
    db.delete(p)
    db.commit()
    delete_project_files(project_id)
    return {"ok": True}


@router.post("/projects/{project_id}/reanalyze", response_model=ProjectOut)
def reanalyze(project_id: str, background: BackgroundTasks, user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    p = owned_project(project_id, user, db)
    extracted = project_dir(p.id) / "extracted"
    shutil.rmtree(extracted, ignore_errors=True)
    p.status, p.error, p.spec_confirmed = "analyzing", None, False
    db.commit()
    background.add_task(analyze_project, p.id)
    return project_out(p)


@router.put("/projects/{project_id}/spec", response_model=ProjectOut)
def update_spec(project_id: str, body: SpecUpdate, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    p = owned_project(project_id, user, db)
    if p.status != "ready":
        raise HTTPException(409, "The dataset is not analysed yet.")
    previous = p.spec or {}
    spec = spec_from_update(body.model_dump())
    spec.rationale = previous.get("rationale", "") if previous.get("task") == spec.task else "Set by you."
    spec.warnings = previous.get("warnings", [])
    try:
        spec = validate_spec(spec, p.dataset_kind, p.profile or {})
    except ValueError as e:
        raise HTTPException(422, str(e))
    p.spec = spec.model_dump()
    p.spec_confirmed = True
    db.commit()
    return project_out(p)


@router.get("/projects/{project_id}/candidates")
def get_candidates(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = owned_project(project_id, user, db)
    if p.status != "ready":
        raise HTTPException(409, "The dataset is not analysed yet.")
    return candidate_lists(p)


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)[:100] or "file"
