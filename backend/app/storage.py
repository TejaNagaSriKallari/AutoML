import shutil
import zipfile
from pathlib import Path, PurePosixPath

from app.config import get_settings


class UnsafeArchive(ValueError):
    pass


def project_dir(project_id: str) -> Path:
    d = get_settings().data_dir / "projects" / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(project_id: str, run_id: str) -> Path:
    d = project_dir(project_id) / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def delete_project_files(project_id: str) -> None:
    shutil.rmtree(get_settings().data_dir / "projects" / project_id, ignore_errors=True)


def safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extracts a user-supplied zip, defending against zip-slip and zip bombs."""
    settings = get_settings()
    max_bytes = int(settings.max_extracted_gb * 1024**3)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > settings.max_zip_members:
            raise UnsafeArchive(f"Archive has {len(members)} files; the limit is {settings.max_zip_members}.")
        total = sum(m.file_size for m in members)
        if total > max_bytes:
            raise UnsafeArchive(f"Archive expands to {total / 1024**3:.1f} GB; the limit is {settings.max_extracted_gb} GB.")
        for m in members:
            name = PurePosixPath(m.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts:
                raise UnsafeArchive(f"Unsafe path in archive: {m.filename}")
            if any(p.startswith("__MACOSX") or p.startswith(".") for p in name.parts):
                continue
            target = (dest / Path(*name.parts)).resolve()
            if not target.is_relative_to(dest_resolved):
                raise UnsafeArchive(f"Unsafe path in archive: {m.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    return dest
