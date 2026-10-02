"""Tabular classification, regression and clustering on CPU."""
from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.registry import get_model
from app.shared_data import load_table, make_row_split
from app.training.context import RunContext
from app.training.evaluate import cv_folds_for, evaluate_supervised, rank_results
from app.training.pool import Job, run_jobs
from app.training.tuning import is_tunable, tune_and_evaluate
from automl_runtime import LabelDecoder, build_tabular_preprocessor, coerce_numeric

MIN_CLASS_COUNT = 3
_BUNDLES: dict[str, dict] = {}


def _bundle(path: str) -> dict:
    """Worker-process cache: each worker loads the prepared data once."""
    if path not in _BUNDLES:
        _BUNDLES.clear()
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def feature_columns(profile: dict, spec: dict) -> dict[str, list[str]]:
    skip = {spec.get("target"), spec.get("date_column") if spec["task"] == "forecasting" else None,
            *spec.get("drop_columns", [])}
    groups = {"numeric": [], "categorical": [], "datetime": [], "text": []}
    for c in profile["columns"]:
        if c["name"] in skip or c["kind"] not in groups:
            continue
        groups[c["kind"]].append(c["name"])
    return groups


# --------------------------------------------------------------------------------------
# Worker-side functions (run inside pool processes)
# --------------------------------------------------------------------------------------

def make_pipeline(spec, columns: dict) -> Pipeline:
    est = spec.build()
    if spec.scale_target:
        est = TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    return Pipeline([("prep", build_tabular_preprocessor(columns, scaler=spec.scaler)), ("model", est)])


def fit_supervised(bundle_path: str, model_key: str, out_path: str, time_limit: float) -> dict:
    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    spec = get_model(model_key)
    pipe = make_pipeline(spec, b["columns"])
    res = evaluate_supervised(pipe, b["X"], b["y"], b["train_idx"], b["test_idx"], b["task"], b["metric"],
                              n_classes=b["n_classes"], cv_folds=b["cv_folds"],
                              deadline=time.monotonic() + time_limit * 0.6)
    joblib.dump(_final(res.pop("model"), b), out_path)
    return res


def _final(model, b: dict):
    """Classification models are saved so they predict the original class names."""
    return LabelDecoder(model, b["classes"]) if b["task"] == "classification" else model


def fit_tuned(bundle_path: str, model_key: str, out_path: str, time_limit: float) -> dict:
    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    spec = get_model(model_key)
    pipe = make_pipeline(spec, b["columns"])
    res = tune_and_evaluate(pipe, model_key, b, time_budget=time_limit * 0.6)
    joblib.dump(_final(res.pop("model"), b), out_path)
    return res


def check_leakage(bundle_path: str) -> list[dict]:
    """Flags single columns that predict the target almost perfectly on their own."""
    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    X, y = b["X"], b["y"]
    idx = np.asarray(b["train_idx"])
    if len(idx) > 5000:
        idx = np.random.default_rng(0).choice(idx, 5000, replace=False)
    is_clf = b["task"] == "classification"
    scoring = "balanced_accuracy" if is_clf else "r2"
    flagged = []
    for group, cols in b["columns"].items():
        for col in cols[:80]:
            single = {k: ([col] if k == group else []) for k in b["columns"]}
            est = DecisionTreeClassifier(max_depth=4, random_state=0) if is_clf else DecisionTreeRegressor(max_depth=4, random_state=0)
            pipe = Pipeline([("prep", build_tabular_preprocessor(single)), ("m", est)])
            try:
                score = cross_val_score(pipe, X.iloc[idx], y[idx], cv=3, scoring=scoring).mean()
            except Exception:
                continue
            if score >= 0.98:
                flagged.append({"column": col, "score": round(float(score), 4), "metric": scoring})
    return flagged


def permutation_importance_for(bundle_path: str, model_path: str) -> list[dict]:
    from sklearn.inspection import permutation_importance

    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    model = joblib.load(model_path)
    model = getattr(model, "pipeline", model)  # unwrap LabelDecoder: scoring uses encoded labels
    test_idx = np.asarray(b["test_idx"])
    if len(test_idx) > 2000:
        test_idx = np.random.default_rng(0).choice(test_idx, 2000, replace=False)
    X_test, y_test = b["X"].iloc[test_idx], b["y"][test_idx]
    scoring = {"classification": "balanced_accuracy", "regression": "r2"}[b["task"]]
    r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0, scoring=scoring, n_jobs=1)
    out = [{"feature": c, "importance": float(m), "std": float(s)}
           for c, m, s in zip(X_test.columns, r.importances_mean, r.importances_std)]
    return sorted(out, key=lambda d: -d["importance"])[:30]


# --------------------------------------------------------------------------------------
# Orchestration (runs in the job worker)
# --------------------------------------------------------------------------------------

def prepare_supervised(ctx: RunContext) -> str:
    spec, profile = ctx.spec, ctx.profile
    task, target = spec["task"], spec["target"]
    df = load_table(ctx.data_path)
    max_rows = ctx.config.get("max_rows") or 200_000

    if task == "classification":
        y_raw = df[target]
        keep = y_raw.notna()
        dropped = int((~keep).sum())
        df, y_raw = df[keep], y_raw[keep].astype(str)
        counts = y_raw.value_counts()
        rare = counts[counts < MIN_CLASS_COUNT].index.tolist()
        if rare:
            ctx.warn(f"Removed {len(rare)} class(es) with fewer than {MIN_CLASS_COUNT} rows: {', '.join(map(str, rare[:5]))}.")
            keep = ~y_raw.isin(rare)
            df, y_raw = df[keep], y_raw[keep]
        if y_raw.nunique() < 2:
            raise ValueError("The target has fewer than two classes with enough rows to learn from.")
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)
        classes = enc.classes_.tolist()
    else:
        y_num = coerce_numeric(df[target])
        keep = y_num.notna()
        dropped = int((~keep).sum())
        df, y = df[keep], y_num[keep].to_numpy(dtype=float)
        classes = []
    if dropped:
        ctx.warn(f"{dropped} rows with a missing target were skipped.")

    if len(df) > max_rows:
        ctx.warn(f"The dataset has {len(df):,} rows; training used a random sample of {max_rows:,} rows.")
        rng = np.random.default_rng(42)
        pick = np.sort(rng.choice(len(df), max_rows, replace=False))
        df, y = df.iloc[pick], y[pick]

    df = df.reset_index(drop=True)
    columns = feature_columns(profile, spec)
    X = df[[c for cols in columns.values() for c in cols]]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing the target and dropped columns.")
    train_idx, test_idx = make_row_split(y, stratify=task == "classification")
    bundle = {
        "X": X, "y": y, "train_idx": np.asarray(train_idx), "test_idx": np.asarray(test_idx),
        "columns": columns, "task": task, "metric": spec["metric"], "n_classes": len(classes),
        "classes": classes, "cv_folds": cv_folds_for(len(train_idx)),
    }
    path = ctx.run_dir / "bundle.joblib"
    joblib.dump(bundle, path)
    ctx.set_insight("data", {
        "rows_used": int(len(X)), "train_rows": len(train_idx), "test_rows": len(test_idx),
        "features": {k: v for k, v in columns.items() if v}, "classes": classes[:100],
        "cv_folds": bundle["cv_folds"],
        "class_balance": {classes[i]: int(c) for i, c in enumerate(np.bincount(y))} if classes else None,
    })
    return str(path)


def run_supervised(ctx: RunContext, model_keys: list[str]) -> None:
    ctx.progress("preparing", "Loading and splitting the data")
    bundle_path = prepare_supervised(ctx)
    models_dir = ctx.run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    ctx.progress("leakage", "Checking for columns that leak the answer")
    leak: list = []
    run_jobs([Job("leak", check_leakage, (bundle_path,), timeout=max(60, ctx.time_limit()))],
             lambda _id, res, err, st: leak.extend(res or []), ctx.cancelled)
    if leak:
        ctx.set_insight("leakage", leak)
        names = ", ".join(f"'{d['column']}'" for d in leak[:5])
        ctx.warn(f"Possible data leakage: {names} alone predict(s) the target almost perfectly. "
                 "If these are only known after the outcome, drop them and retrain.")

    _train_many(ctx, bundle_path, model_keys, fit_supervised, "training", "Training")

    tune_k = int(ctx.config.get("tune_top_k", 0))
    if tune_k and not ctx.cancelled():
        ranked = rank_results(_completed_results(ctx), ctx.spec["metric"])
        to_tune = [r["model_key"] for r in ranked if is_tunable(r["model_key"]) and not r["model_key"].endswith("+tuned")][:tune_k]
        if to_tune:
            _train_many(ctx, bundle_path, to_tune, fit_tuned, "tuning", "Tuning", suffix="+tuned",
                        timeout_factor=3)

    best = _best_completed(ctx)
    if best and best["model_key"] in ctx.model_paths and not ctx.cancelled():
        ctx.progress("explaining", "Computing feature importance for the best model")
        out: list = []
        run_jobs([Job("imp", permutation_importance_for, (bundle_path, ctx.model_paths[best["model_key"]]),
                      timeout=max(120, ctx.time_limit()))],
                 lambda _id, res, err, st: out.extend(res or []), ctx.cancelled)
        if out:
            ctx.set_insight("feature_importance", out)


def model_file(models_dir: Path, key: str) -> Path:
    return models_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.joblib"


def _train_many(ctx: RunContext, bundle_path: str, model_keys: list[str], fn, stage: str, verb: str,
                suffix: str = "", timeout_factor: float = 1.0) -> None:
    models_dir = ctx.run_dir / "models"
    total, done = len(model_keys), 0
    jobs = []
    for key in model_keys:
        out = model_file(models_dir, key + suffix)
        jobs.append(Job(key, fn, (bundle_path, key, str(out), ctx.time_limit()), timeout=ctx.time_limit() * timeout_factor))
    ctx.progress(stage, f"{verb} {total} models in parallel", 0, total)

    def on_result(key, res, err, status):
        nonlocal done
        done += 1
        spec = get_model(key)
        result_key = key + suffix
        if res is not None:
            ctx.add_result(model_key=result_key, name=spec.name + (" (tuned)" if suffix else ""), family=spec.family,
                           status="completed", metrics=res["metrics"], primary_score=res["primary_score"],
                           cv_mean=res["cv_mean"], cv_std=res["cv_std"], train_seconds=res["train_seconds"],
                           params=res.get("best_params") or spec.params, details=res["details"],
                           model_path=str(model_file(models_dir, result_key)))
        else:
            ctx.add_result(model_key=result_key, name=spec.name, family=spec.family, status=status, error=err)
        ctx.progress(stage, f"{verb}: finished {spec.name}", done, total)

    run_jobs(jobs, on_result, ctx.cancelled)


def _completed_results(ctx: RunContext) -> list[dict]:
    from app.db import SessionLocal
    from app.models import ModelResult

    with SessionLocal() as db:
        rows = db.query(ModelResult).filter(ModelResult.run_id == ctx.run_id).all()
        return [{"model_key": r.model_key, "status": r.status, "primary_score": r.primary_score,
                 "cv_mean": r.cv_mean, "name": r.name} for r in rows]


def _best_completed(ctx: RunContext) -> dict | None:
    ranked = rank_results(_completed_results(ctx), ctx.spec["metric"])
    return ranked[0] if ranked else None


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------

def fit_cluster(bundle_path: str, model_key: str, out_path: str, time_limit: float) -> dict:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.neighbors import KNeighborsClassifier

    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    X = b["Xt"]
    spec = get_model(model_key)
    rng = np.random.default_rng(42)
    fit_idx = np.arange(len(X))
    if spec.max_rows and len(X) > spec.max_rows:
        fit_idx = np.sort(rng.choice(len(X), spec.max_rows, replace=False))
    eval_idx = fit_idx if len(fit_idx) <= 5000 else np.sort(rng.choice(fit_idx, 5000, replace=False))

    t0 = time.perf_counter()
    est0 = spec.build()
    param = "n_clusters" if "n_clusters" in est0.get_params() else "n_components" if "n_components" in est0.get_params() else None
    fixed_k = b.get("n_clusters")
    ks = [fixed_k] if (param and fixed_k) else (list(range(2, min(10, len(fit_idx) - 1) + 1)) if param else [None])
    best = None
    deadline = time.monotonic() + time_limit * 0.8
    for k in ks:
        if best is not None and time.monotonic() > deadline:
            break
        est = clone(est0)
        if k is not None:
            est.set_params(**{param: k})
        labels = est.fit_predict(X[fit_idx]) if hasattr(est, "fit_predict") else est.fit(X[fit_idx]).predict(X[fit_idx])
        pos = np.searchsorted(fit_idx, eval_idx)
        lab_eval = labels[pos]
        n_found = len(set(lab_eval) - {-1})
        mask = lab_eval != -1
        if n_found < 2 or mask.sum() < 3 or len(set(lab_eval[mask])) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(X[eval_idx][mask], lab_eval[mask]))
        if best is None or score > best[0]:
            best = (score, k, est, labels)
    score, k, est, labels = best
    # label every row (models fitted on a sample assign the rest by nearest neighbour)
    if len(fit_idx) < len(X):
        knn = KNeighborsClassifier(n_neighbors=1).fit(X[fit_idx], labels)
        all_labels = knn.predict(X)
    else:
        all_labels = labels
    mask = all_labels != -1
    sample = eval_idx
    m = {"silhouette": score}
    lab_s = all_labels[sample]
    ms = lab_s != -1
    if len(set(lab_s[ms])) >= 2:
        m["calinski_harabasz"] = float(calinski_harabasz_score(X[sample][ms], lab_s[ms]))
        m["davies_bouldin"] = float(davies_bouldin_score(X[sample][ms], lab_s[ms]))
    sizes = pd.Series(all_labels).value_counts().sort_index()
    m["n_clusters"] = int(len(set(all_labels) - {-1}))
    m["noise_pct"] = float((~mask).mean() * 100)
    joblib.dump({"labels": all_labels, "model": est}, out_path)
    return {
        "metrics": m, "primary_score": m.get(b["metric"], score), "cv_mean": None, "cv_std": None,
        "train_seconds": round(time.perf_counter() - t0, 3),
        "details": {"cluster_sizes": {str(k_): int(v) for k_, v in sizes.items()}, "chosen_k": k,
                    "fitted_on_rows": int(len(fit_idx))},
        "best_params": {**spec.params, **({param: k} if k else {})},
    }


def run_clustering(ctx: RunContext, model_keys: list[str]) -> None:
    ctx.progress("preparing", "Loading and encoding the data")
    df = load_table(ctx.data_path)
    max_rows = ctx.config.get("max_rows") or 200_000
    if len(df) > max_rows:
        ctx.warn(f"The dataset has {len(df):,} rows; clustering used a random sample of {max_rows:,}.")
        df = df.sample(max_rows, random_state=42)
    df = df.reset_index(drop=True)
    columns = feature_columns(ctx.profile, ctx.spec)
    X = df[[c for cols in columns.values() for c in cols]]
    prep = build_tabular_preprocessor(columns)
    Xt = prep.fit_transform(X).astype(np.float32)
    bundle = {"Xt": Xt, "metric": ctx.spec["metric"], "n_clusters": ctx.spec.get("n_clusters")}
    path = ctx.run_dir / "bundle.joblib"
    joblib.dump(bundle, path)
    joblib.dump(prep, ctx.run_dir / "preprocessor.joblib")
    ctx.set_insight("data", {"rows_used": int(len(X)), "encoded_features": int(Xt.shape[1]),
                             "features": {k: v for k, v in columns.items() if v}})
    (ctx.run_dir / "models").mkdir(exist_ok=True)
    _train_many(ctx, str(path), model_keys, fit_cluster, "training", "Clustering with")


def export_clusters(data_path: str, run_dir: Path, model_path: str, max_rows: int) -> Path:
    """Original rows + assigned cluster, for download."""
    df = load_table(data_path)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    df = df.reset_index(drop=True)
    labels = joblib.load(model_path)["labels"]
    df["cluster"] = labels
    out = run_dir / "clusters.csv"
    df.to_csv(out, index=False)
    return out
