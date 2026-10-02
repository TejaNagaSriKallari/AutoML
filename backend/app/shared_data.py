"""Dataset discovery and train/test splitting.

This module is copied verbatim into generated Colab notebooks, so the CPU path and the
Colab GPU path discover exactly the same files and use exactly the same test split.
Keep it self-contained: standard library + numpy + pandas + scikit-learn only.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
TABLE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "dev": "val", "test": "test", "testing": "test"}
SEED = 42
TEST_SIZE = 0.2


def _visible(p: Path) -> bool:
    return not any(part.startswith(".") or part.startswith("__MACOSX") for part in p.parts)


def unwrap_root(root: Path) -> Path:
    """Zips often wrap everything in one folder; descend through single-folder levels."""
    root = Path(root)
    while True:
        children = [c for c in root.iterdir() if _visible(Path(c.name))]
        dirs = [c for c in children if c.is_dir()]
        if len(dirs) == 1 and len(children) == 1:
            root = dirs[0]
        else:
            return root


def _images_under(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and _visible(p.relative_to(d)))


def load_table(path, nrows=None) -> pd.DataFrame:
    df = _read_table(Path(path), nrows)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_table(path: Path, nrows=None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows)
    if ext == ".json":
        df = pd.read_json(path)
        return df.head(nrows) if nrows else df
    raw = path.open("rb").read(64 * 1024)
    encoding = "utf-8"
    try:
        sample = raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = "latin-1"
        sample = raw.decode("latin-1")
    try:
        sep = csv.Sniffer().sniff("\n".join(sample.splitlines()[:20]), delimiters=",;\t|").delimiter
    except csv.Error:
        sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(path, sep=sep, encoding=encoding, nrows=nrows, low_memory=False,
                       encoding_errors="replace")


def discover_image_items(root) -> dict:
    """Finds labelled images. Supported layouts:
    1. root/<class>/*.jpg
    2. root/<train|val|test>/<class>/*.jpg
    3. images anywhere + a CSV with a filename column and a label column
    Returns {"root", "layout", "items": [{"path", "label", "split"}], "notes": [...]}
    """
    root = unwrap_root(Path(root))
    notes = []
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and _visible(Path(d.name)))
    split_dirs = [d for d in subdirs if d.name.lower() in SPLIT_NAMES]
    items = []

    if split_dirs:
        layout = "split_folders"
        for sd in split_dirs:
            split = SPLIT_NAMES[sd.name.lower()]
            for cd in sorted(d for d in sd.iterdir() if d.is_dir() and _visible(Path(d.name))):
                for img in _images_under(cd):
                    items.append({"path": img.relative_to(root).as_posix(), "label": cd.name, "split": split})
        ignored = [d.name for d in subdirs if d not in split_dirs]
        if ignored:
            notes.append(f"Ignored folders outside train/val/test: {', '.join(ignored[:5])}")
    else:
        csvs = sorted(p for p in root.rglob("*.csv") if _visible(p.relative_to(root)))
        class_dirs = [d for d in subdirs if _images_under(d)]
        if csvs and not (len(class_dirs) >= 2 and not any(d.name.lower() in {"images", "imgs", "img", "data"} for d in class_dirs)):
            layout = "csv_labels"
            df = load_table(csvs[0])
            cols = {c.lower(): c for c in df.columns}
            fcol = next((cols[c] for c in ("filename", "file", "image", "image_id", "path", "filepath", "img", "id", "name") if c in cols), df.columns[0])
            lcol = next((cols[c] for c in ("label", "labels", "class", "category", "target", "y") if c in cols), df.columns[-1])
            all_imgs = _images_under(root)
            by_rel = {p.relative_to(root).as_posix(): p for p in all_imgs}
            by_name = {p.name: p for p in all_imgs}
            by_stem = {p.stem: p for p in all_imgs}
            missing = 0
            for f, lab in zip(df[fcol].astype(str), df[lcol]):
                if pd.isna(lab):
                    continue
                f = f.replace("\\", "/").strip()
                p = by_rel.get(f) or by_name.get(Path(f).name) or by_stem.get(Path(f).stem)
                if p is None:
                    missing += 1
                    continue
                items.append({"path": p.relative_to(root).as_posix(), "label": str(lab), "split": None})
            notes.append(f"Labels read from {csvs[0].name} (file column '{fcol}', label column '{lcol}').")
            if missing:
                notes.append(f"{missing} rows in the CSV did not match any image file.")
        else:
            layout = "class_folders"
            for cd in class_dirs:
                for img in _images_under(cd):
                    items.append({"path": img.relative_to(root).as_posix(), "label": cd.name, "split": None})
            loose = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if loose:
                notes.append(f"{len(loose)} images directly in the top folder have no class folder and were ignored.")

    items.sort(key=lambda it: it["path"])
    return {"root": str(root), "layout": layout, "items": items, "notes": notes}


def prepare_image_items(root, min_per_class=3, verify=True):
    """discover_image_items + drop unreadable files + drop classes too small to learn/evaluate.
    Returns (root, items, notes). Used identically by the CPU trainer and Colab notebooks."""
    found = discover_image_items(root)
    items, notes = found["items"], list(found["notes"])
    if verify:
        from PIL import Image

        good = []
        for it in items:
            try:
                with Image.open(Path(found["root"]) / it["path"]) as im:
                    im.verify()
                good.append(it)
            except Exception:
                pass
        if len(good) < len(items):
            notes.append(f"Skipped {len(items) - len(good)} unreadable image(s).")
        items = good
    counts = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    small = sorted(c for c, n in counts.items() if n < min_per_class)
    if small:
        notes.append(f"Removed {len(small)} class(es) with fewer than {min_per_class} images: {', '.join(small[:5])}.")
        items = [it for it in items if it["label"] not in small]
    return found["root"], items, notes


def make_image_split(items, test_size=TEST_SIZE, seed=SEED):
    """Deterministic split shared by CPU and Colab. Returns (train_idx, test_idx)."""
    splits = [it.get("split") for it in items]
    if "test" in splits:
        test = [i for i, s in enumerate(splits) if s == "test"]
        train = [i for i, s in enumerate(splits) if s != "test"]
        return train, test
    if "val" in splits:
        test = [i for i, s in enumerate(splits) if s == "val"]
        train = [i for i, s in enumerate(splits) if s != "val"]
        return train, test
    return make_row_split([it["label"] for it in items], test_size=test_size, seed=seed, stratify=True)


def make_row_split(labels, test_size=TEST_SIZE, seed=SEED, stratify=True):
    """Stratified when every class has >= 2 members; rare classes are kept in train."""
    n = len(labels)
    idx = np.arange(n)
    if not stratify:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
        return sorted(tr.tolist()), sorted(te.tolist())
    labels = np.asarray([str(x) for x in labels], dtype=object)
    values, counts = np.unique(labels, return_counts=True)
    rare = set(values[counts < 2])
    common_idx = np.array([i for i in idx if labels[i] not in rare], dtype=int)
    rare_idx = [int(i) for i in idx if labels[i] in rare]
    n_classes = len(values) - len(rare)
    n_test = int(round(len(common_idx) * test_size))
    if n_classes >= 2 and n_test >= n_classes and len(common_idx) - n_test >= n_classes:
        tr, te = train_test_split(common_idx, test_size=test_size, random_state=seed, stratify=labels[common_idx])
    else:
        tr, te = train_test_split(common_idx if len(common_idx) else idx, test_size=test_size, random_state=seed)
    return sorted(tr.tolist() + rare_idx), sorted(te.tolist())


def find_yolo_dataset(root):
    """Returns info about a YOLO-format detection dataset, or None."""
    root = unwrap_root(Path(root))
    yamls = sorted(p for p in root.rglob("*.y*ml") if _visible(p.relative_to(root)))
    label_txts = [p for p in root.rglob("*.txt") if "labels" in [q.lower() for q in p.parts]]
    if not yamls or not label_txts:
        return None
    return {"root": str(root), "yaml": str(yamls[0]), "n_label_files": len(label_txts),
            "n_images": len(_images_under(root))}

