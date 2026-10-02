"""Looks at an uploaded dataset and describes it: modality, columns, classes, quality issues."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.shared_data import (
    IMAGE_EXTS,
    TABLE_EXTS,
    discover_image_items,
    find_yolo_dataset,
    load_table,
    unwrap_root,
)
from app.storage import safe_extract_zip
from automl_runtime import coerce_numeric

_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|^unnamed: ?0$)", re.I)


class DatasetError(ValueError):
    pass


def infer_column_kind(s: pd.Series) -> str:
    n = len(s)
    nn = s.dropna()
    if len(nn) == 0:
        return "empty"
    nu = nn.nunique()
    if nu <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        integer_valued = bool(np.all(np.mod(nn.to_numpy(dtype=float), 1) == 0))
        if integer_valued and nu == len(nn) and n > 50 and (_ID_NAME.search(str(s.name)) or nn.is_monotonic_increasing):
            return "id"
        return "numeric"
    sample = nn.astype(str).head(2000)
    if coerce_numeric(sample).notna().mean() >= 0.95:
        return "numeric"
    avg_len = sample.str.len().mean()
    if sample.str.contains(r"\d").mean() > 0.9 and 6 <= avg_len <= 40:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return "datetime"
    unique_ratio = nu / len(nn)
    avg_words = sample.str.count(r"\s+").mean() + 1
    if avg_len >= 30 and (unique_ratio >= 0.3 or (avg_words >= 4 and nu >= 30)):
        return "text"
    if (unique_ratio >= 0.95 and n > 50) or (_ID_NAME.search(str(s.name)) and unique_ratio > 0.9):
        return "id"
    return "categorical"


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v if isinstance(v, (int, float, bool)) else str(v)[:120]


SEASON_BY_FREQ = {"H": 24, "D": 7, "B": 5, "W": 52, "M": 12, "MS": 12, "ME": 12, "Q": 4, "QS": 4, "QE": 4}


def infer_frequency(dates: pd.Series) -> str | None:
    d = pd.Series(pd.to_datetime(dates, errors="coerce", format="mixed")).dropna().drop_duplicates().sort_values()
    if len(d) < 3:
        return None
    freq = pd.infer_freq(d.tail(500)) if len(d) >= 3 else None
    if freq is None:
        step = d.diff().median()
        days = step / pd.Timedelta(days=1)
        for f, lo, hi in (("h", 0.03, 0.05), ("D", 0.9, 1.1), ("W", 6.5, 7.5), ("MS", 27, 32), ("QS", 85, 95), ("YS", 360, 370)):
            if lo <= days <= hi:
                return f
        return None
    return freq


def season_length_for(freq: str | None) -> int | None:
    if not freq:
        return None
    base = freq.split("-")[0].upper().lstrip("0123456789")
    return SEASON_BY_FREQ.get(base)


def _datetime_info(nn: pd.Series) -> dict:
    parsed = pd.to_datetime(nn, errors="coerce", format="mixed").dropna()
    freq = infer_frequency(parsed)
    return {
        "min_date": parsed.min().isoformat() if len(parsed) else None,
        "max_date": parsed.max().isoformat() if len(parsed) else None,
        "n_unique_dates": int(parsed.nunique()),
        "freq": freq,
        "season_length": season_length_for(freq),
    }


def profile_table(df: pd.DataFrame) -> dict:
    n = len(df)
    cols = []
    for name in df.columns:
        s = df[name]
        kind = infer_column_kind(s)
        nn = s.dropna()
        info = {
            "name": str(name),
            "dtype": str(s.dtype),
            "kind": kind,
            "missing_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
            "n_unique": int(nn.nunique()),
            "unique_ratio": round(float(nn.nunique() / max(len(nn), 1)), 4),
            "samples": [_jsonable(v) for v in nn.drop_duplicates().head(5).tolist()],
        }
        if kind == "numeric":
            num = coerce_numeric(nn)
            info["stats"] = {k: _jsonable(getattr(num, k)()) for k in ("min", "max", "mean", "std")}
            info["integer_valued"] = bool(num.dropna().mod(1).eq(0).all())
        if kind in {"categorical", "constant"} or (kind == "numeric" and info["n_unique"] <= 20):
            info["top_values"] = [{"value": _jsonable(k), "count": int(v)} for k, v in nn.value_counts().head(8).items()]
        if kind == "text":
            info["avg_length"] = round(float(nn.astype(str).str.len().mean()), 1)
        if kind == "datetime":
            info.update(_datetime_info(nn))
        cols.append(info)
    return {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "n_duplicate_rows": int(df.duplicated().sum()) if n <= 500_000 else None,
        "columns": cols,
    }


def profile_images(root: Path) -> dict:
    found = discover_image_items(root)
    items = found["items"]
    if not items:
        raise DatasetError(
            "No labelled images found. Use class folders (dataset/cat/*.jpg, dataset/dog/*.jpg), "
            "train/val/test folders with class subfolders, or images plus a labels.csv."
        )
    counts = Counter(it["label"] for it in items)
    split_counts = Counter(it["split"] or "unsplit" for it in items)
    sizes, unreadable = [], 0
    for it in items[:: max(1, len(items) // 60)][:60]:
        try:
            with Image.open(Path(found["root"]) / it["path"]) as im:
                sizes.append(im.size)
        except Exception:
            unreadable += 1
    notes = list(found["notes"])
    small = [c for c, k in counts.items() if k < 5]
    if small:
        notes.append(f"{len(small)} class(es) have fewer than 5 images: {', '.join(small[:5])}.")
    if counts and max(counts.values()) / max(min(counts.values()), 1) >= 5:
        notes.append("Classes are strongly imbalanced; macro-F1 will be used alongside accuracy.")
    if unreadable:
        notes.append(f"{unreadable} sampled images could not be opened (corrupt or unsupported).")
    return {
        "root": found["root"],
        "layout": found["layout"],
        "n_images": len(items),
        "n_classes": len(counts),
        "classes": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "splits": dict(split_counts),
        "median_size": [int(np.median([s[0] for s in sizes])), int(np.median([s[1] for s in sizes]))] if sizes else None,
        "notes": notes,
    }


def profile_yolo(info: dict) -> dict:
    import yaml

    with open(info["yaml"], encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return {
        "root": info["root"],
        "yaml": info["yaml"],
        "n_images": info["n_images"],
        "n_label_files": info["n_label_files"],
        "classes": names,
        "n_classes": len(names),
        "splits": {k: str(cfg[k]) for k in ("train", "val", "test") if k in cfg},
        "notes": [],
    }


def analyze_upload(upload_path: Path, work_dir: Path) -> tuple[str, Path, dict]:
    """Returns (dataset_kind, data_path, profile). dataset_kind: table | image_folder | yolo_detection."""
    ext = upload_path.suffix.lower()
    if ext in TABLE_EXTS:
        return "table", upload_path, _profile_table_file(upload_path)
    if ext != ".zip":
        raise DatasetError(f"Unsupported file type '{ext}'. Upload CSV, Excel, Parquet, JSON, or a ZIP of images.")

    extracted = safe_extract_zip(upload_path, work_dir / "extracted")
    root = unwrap_root(extracted)
    yolo = find_yolo_dataset(root)
    if yolo:
        return "yolo_detection", Path(yolo["root"]), profile_yolo(yolo)
    has_images = any(p.suffix.lower() in IMAGE_EXTS for p in root.rglob("*"))
    if has_images:
        prof = profile_images(root)
        return "image_folder", Path(prof["root"]), prof
    tables = sorted(p for p in root.rglob("*") if p.suffix.lower() in TABLE_EXTS)
    if len(tables) >= 1:
        return "table", tables[0], _profile_table_file(tables[0])
    raise DatasetError("The ZIP contains no images or tables I can use.")


def _profile_table_file(path: Path) -> dict:
    try:
        df = load_table(path)
    except Exception as e:  # pandas raises many types for malformed files
        raise DatasetError(f"Could not read the table: {e}") from e
    if df.empty or df.shape[1] == 0:
        raise DatasetError("The table is empty.")
    prof = profile_table(df)
    prof["file"] = str(path)
    return prof
