"""Text classification on CPU: (TF-IDF or frozen pretrained sentence embeddings) x classifier heads."""
from __future__ import annotations

import hashlib
import warnings

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.registry import HEADS, TEXT_ENCODERS, build_head
from app.shared_data import load_table, make_row_split
from app.training.context import RunContext
from app.training.evaluate import cv_folds_for, evaluate_supervised
from app.training.pool import Job, run_jobs
from automl_runtime import LabelDecoder, SentenceEmbedder, TextColumn

MIN_CLASS_COUNT = 3
_BUNDLES: dict[str, dict] = {}


def _bundle(path: str) -> dict:
    if path not in _BUNDLES:
        _BUNDLES.clear()
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def _tfidf(kind: str) -> TfidfVectorizer:
    if kind == "tfidf_char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000, sublinear_tf=True)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, strip_accents="unicode")


def fit_text(bundle_path: str, encoder: str, head: str, emb_path: str | None, out_path: str, time_limit: float) -> dict:
    import time

    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    if emb_path:  # dense: precomputed sentence embeddings
        X = np.load(emb_path)
        pipe = Pipeline([("clf", build_head(head))])
    else:
        X = b["texts"]
        pipe = Pipeline([("tfidf", _tfidf(encoder)), ("clf", build_head(head))])
    res = evaluate_supervised(pipe, X, b["y"], b["train_idx"], b["test_idx"], "classification", b["metric"],
                              n_classes=len(b["classes"]), cv_folds=b["cv_folds"],
                              deadline=time.monotonic() + time_limit * 0.6)
    fitted = res.pop("model")
    steps = [("text", TextColumn(b["text_column"]))]
    steps += [("embed", SentenceEmbedder(encoder))] if emb_path else []
    final = LabelDecoder(Pipeline(steps + list(fitted.steps)), b["classes"])
    joblib.dump(final, out_path)
    return res


def run_text(ctx: RunContext, model_keys: list[str]) -> None:
    spec = ctx.spec
    ctx.progress("preparing", "Loading texts and labels")
    df = load_table(ctx.data_path)
    df = df[df[spec["target"]].notna()]
    labels = df[spec["target"]].astype(str)
    counts = labels.value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare:
        ctx.warn(f"Removed {len(rare)} class(es) with fewer than {MIN_CLASS_COUNT} examples.")
        keep = ~labels.isin(rare)
        df, labels = df[keep], labels[keep]
    max_rows = ctx.config.get("max_rows") or 200_000
    if len(df) > max_rows:
        ctx.warn(f"Used a random sample of {max_rows:,} of {len(df):,} rows.")
        df = df.sample(max_rows, random_state=42)
        labels = labels.loc[df.index]
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    texts = df[spec["text_column"]].fillna("").astype(str).to_numpy()
    train_idx, test_idx = make_row_split(labels.to_numpy())  # same call as the Colab notebook
    bundle = {"texts": texts, "y": y, "train_idx": np.asarray(train_idx), "test_idx": np.asarray(test_idx),
              "classes": enc.classes_.tolist(), "metric": spec["metric"], "text_column": spec["text_column"],
              "cv_folds": min(cv_folds_for(len(train_idx)), 5)}
    path = ctx.run_dir / "bundle.joblib"
    joblib.dump(bundle, path)
    ctx.set_insight("data", {"rows_used": int(len(texts)), "train_rows": len(train_idx), "test_rows": len(test_idx),
                             "classes": bundle["classes"][:100], "avg_chars": float(np.mean([len(t) for t in texts]))})

    encoders = [k.split("enc:", 1)[1] for k in model_keys if "/cpu/enc:" in k]
    heads = [k.split("head:", 1)[1] for k in model_keys if "/cpu/head:" in k]
    if not encoders or not heads:
        raise ValueError("Select at least one text representation and one classifier.")

    emb_paths: dict[str, str | None] = {}
    for i, e in enumerate(encoders):
        if TEXT_ENCODERS.get(e, ("", "dense"))[1] == "sparse":
            emb_paths[e] = None
            continue
        ctx.progress("embedding", f"Encoding texts with {e} ({i + 1}/{len(encoders)})")
        try:
            X = SentenceEmbedder(e).transform(texts)
        except Exception as ex:
            ctx.warn(f"Could not load encoder '{e}': {ex}")
            continue
        p = ctx.run_dir / f"emb_{hashlib.sha1(e.encode()).hexdigest()[:10]}.npy"
        np.save(p, X.astype(np.float32))
        emb_paths[e] = str(p)
        if ctx.cancelled():
            return

    models_dir = ctx.run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    jobs, meta = [], {}
    for e, emb in emb_paths.items():
        kind = "dense" if emb else "sparse"
        for h in heads:
            if kind not in HEADS[h][2]:
                continue
            key = f"text_classification/cpu/{e}+{h}"
            out = models_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.joblib"
            enc_name = TEXT_ENCODERS.get(e, (e,))[0]
            meta[key] = (f"{enc_name} → {HEADS[h][0]}", HEADS[h][1], str(out), {"encoder": e, "head": h})
            jobs.append(Job(key, fit_text, (str(path), e, h, emb, str(out), ctx.time_limit()), timeout=ctx.time_limit()))

    total, done = len(jobs), 0
    ctx.progress("training", f"Training {total} encoder × classifier combinations", 0, total)

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
