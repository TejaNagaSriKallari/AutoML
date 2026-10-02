"""Image classification on CPU (Option 1): frozen pretrained backbones turn every image into a
feature vector once, then many fast classifiers are trained and compared on those features."""
from __future__ import annotations

import hashlib
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.registry import HEADS, IMAGE_BACKBONES, build_head
from app.shared_data import make_image_split, prepare_image_items
from app.training.context import RunContext
from app.training.evaluate import cv_folds_for, evaluate_supervised
from app.training.pool import Job, run_jobs
from automl_runtime import LabelDecoder, TimmImageEmbedder

_BUNDLES: dict[str, dict] = {}
_BACKBONE_NAMES = {name: display for name, display, _, _ in IMAGE_BACKBONES}


def _bundle(path: str) -> dict:
    if path not in _BUNDLES:
        _BUNDLES.clear()
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def fit_image_head(bundle_path: str, backbone: str, head: str, emb_path: str, out_path: str, time_limit: float) -> dict:
    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    X = np.load(emb_path)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", build_head(head))])
    res = evaluate_supervised(pipe, X, b["y"], b["train_pos"], b["test_pos"], "classification", b["metric"],
                              n_classes=len(b["classes"]), cv_folds=b["cv_folds"],
                              deadline=time.monotonic() + time_limit * 0.6)
    fitted = res.pop("model")
    final = LabelDecoder(Pipeline([("embed", TimmImageEmbedder(backbone))] + list(fitted.steps)), b["classes"])
    joblib.dump(final, out_path)
    return res


def run_image(ctx: RunContext, model_keys: list[str]) -> None:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    ctx.progress("preparing", "Scanning and verifying images")
    root, items, notes = prepare_image_items(ctx.data_path)
    for n in notes:
        ctx.warn(n)
    labels = sorted({it["label"] for it in items})
    if len(labels) < 2:
        raise ValueError("Need at least two classes with 3+ images each.")
    train_idx, test_idx = make_image_split(items)

    max_images = int(ctx.config.get("max_images") or 8000)
    if len(train_idx) > max_images:
        rng = np.random.default_rng(42)
        by_class: dict[str, list[int]] = {}
        for i in train_idx:
            by_class.setdefault(items[i]["label"], []).append(i)
        frac = max_images / len(train_idx)
        train_idx = sorted(j for idx in by_class.values()
                           for j in rng.choice(idx, max(2, int(round(len(idx) * frac))), replace=False).tolist())
        ctx.warn(f"CPU mode used a stratified sample of {len(train_idx):,} training images (limit {max_images:,}). "
                 "Use the Colab GPU option to train on everything.")

    used = sorted(set(train_idx) | set(test_idx))
    pos = {i: p for p, i in enumerate(used)}
    enc = LabelEncoder().fit(labels)
    y = enc.transform([items[i]["label"] for i in used])
    paths = [str(Path(root) / items[i]["path"]) for i in used]
    bundle = {"y": y, "train_pos": np.array([pos[i] for i in train_idx]), "test_pos": np.array([pos[i] for i in test_idx]),
              "classes": enc.classes_.tolist(), "metric": ctx.spec["metric"],
              "cv_folds": min(cv_folds_for(len(train_idx)), 5)}
    bpath = ctx.run_dir / "bundle.joblib"
    joblib.dump(bundle, bpath)
    ctx.set_insight("data", {"images_used": len(used), "train_images": len(train_idx), "test_images": len(test_idx),
                             "classes": bundle["classes"][:200],
                             "split": "given test/val folder" if any(it.get("split") for it in items) else "stratified 80/20"})

    backbones = [k.split("backbone:", 1)[1] for k in model_keys if "/cpu/backbone:" in k]
    heads = [k.split("head:", 1)[1] for k in model_keys if "/cpu/head:" in k]
    if not backbones or not heads:
        raise ValueError("Select at least one backbone and one classifier head.")

    emb_paths = {}
    for i, bb in enumerate(backbones):
        label = _BACKBONE_NAMES.get(bb, bb)

        def report(done, total, _label=label, _i=i):
            ctx.progress("embedding", f"Extracting features with {_label} (backbone {_i + 1}/{len(backbones)})", done, total)

        report(0, len(paths))
        try:
            X = TimmImageEmbedder(bb).transform(paths, progress=report)
        except Exception as e:
            ctx.warn(f"Backbone {label} failed to load or run: {e}")
            continue
        p = ctx.run_dir / f"emb_{hashlib.sha1(bb.encode()).hexdigest()[:10]}.npy"
        np.save(p, X.astype(np.float32))
        emb_paths[bb] = str(p)
        if ctx.cancelled():
            return

    models_dir = ctx.run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    jobs, meta = [], {}
    for bb, emb in emb_paths.items():
        for h in heads:
            key = f"image_classification/cpu/{bb}+{h}"
            out = models_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.joblib"
            meta[key] = (f"{_BACKBONE_NAMES.get(bb, bb)} → {HEADS[h][0]}", HEADS[h][1], str(out), {"backbone": bb, "head": h})
            jobs.append(Job(key, fit_image_head, (str(bpath), bb, h, emb, str(out), ctx.time_limit()),
                            timeout=ctx.time_limit()))

    total, done = len(jobs), 0
    ctx.progress("training", f"Training {total} backbone × classifier combinations", 0, total)

    def on_result(key, res, err, status):
        nonlocal done
        done += 1
        name, fam, out, params = meta[key]
        if res:
            ctx.add_result(model_key=key, name=name, family=fam, status="completed", metrics=res["metrics"],
                           primary_score=res["primary_score"], cv_mean=res["cv_mean"], cv_std=res["cv_std"],
                           train_seconds=res["train_seconds"], params=params, details=res["details"], model_path=out)
        else:
            ctx.add_result(model_key=key, name=name, family=fam, status=status, error=err, params=params)
        ctx.progress("training", f"Finished {name}", done, total)

    run_jobs(jobs, on_result, ctx.cancelled)
