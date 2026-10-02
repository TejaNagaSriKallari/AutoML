"""Shared evaluation protocol: every model of a run is scored the same way on the same split."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from app.registry import METRICS


def higher_is_better(metric: str) -> bool:
    return METRICS.get(metric, {}).get("higher_is_better", True)


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass
    return None


def classification_metrics(y_true, y_pred, scores, n_classes: int) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            s = np.asarray(scores)
            if n_classes == 2:
                pos = s[:, 1] if s.ndim == 2 else s
                m["roc_auc"] = roc_auc_score(y_true, pos)
            elif s.ndim == 2 and s.shape[1] == n_classes:
                probs = s if np.allclose(s.sum(axis=1), 1, atol=1e-3) else _softmax(s)
                m["roc_auc"] = roc_auc_score(y_true, probs, multi_class="ovr", labels=list(range(n_classes)))
            if s.ndim == 2 and np.all(s >= 0) and np.allclose(s.sum(axis=1), 1, atol=1e-3):
                m["log_loss"] = log_loss(y_true, np.clip(s, 1e-15, 1), labels=list(range(n_classes)))
        except ValueError:
            pass
    return {k: float(v) for k, v in m.items()}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y_true) > 1e-9
    m = {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    if nonzero.mean() > 0.9:
        m["mape"] = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {k: float(v) for k, v in m.items()}


def primary_from(metrics: dict, metric: str):
    v = metrics.get(metric)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def cv_folds_for(n_train: int) -> int:
    if n_train > 200_000:
        return 0
    if n_train > 20_000:
        return 3
    return 5


def evaluate_supervised(estimator, X, y, train_idx, test_idx, task: str, metric: str, n_classes: int = 0,
                        cv_folds: int = 5, deadline: float | None = None) -> dict:
    """K-fold CV on the training part (for ranking stability) + fit on all training data and score
    on the untouched test part. Returns metrics, CV stats, fitted model and test details."""
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    X_train, y_train = _take(X, train_idx), y[train_idx]
    X_test, y_test = _take(X, test_idx), y[test_idx]
    is_clf = task != "regression"

    cv_scores = []
    if cv_folds >= 2:
        if is_clf:
            min_count = np.bincount(y_train).min() if len(y_train) else 0
            folds = min(cv_folds, int(min_count)) if min_count >= 2 else 0
            splitter = StratifiedKFold(folds, shuffle=True, random_state=42) if folds >= 2 else None
        else:
            splitter = KFold(cv_folds, shuffle=True, random_state=42)
        if splitter is not None:
            for tr, va in splitter.split(np.zeros(len(y_train)), y_train):
                if deadline and time.monotonic() > deadline:
                    break
                m = clone(estimator)
                m.fit(_take(X_train, tr), y_train[tr])
                pred = m.predict(_take(X_train, va))
                if is_clf:
                    fold = classification_metrics(y_train[va], pred, _scores(m, _take(X_train, va)), n_classes)
                else:
                    fold = regression_metrics(y_train[va], pred)
                s = fold.get(metric)
                if s is not None:
                    cv_scores.append(s)

    model = clone(estimator)
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    pred = model.predict(X_test)
    details = {}
    if is_clf:
        metrics = classification_metrics(y_test, pred, _scores(model, X_test), n_classes)
        if n_classes <= 30:
            details["confusion_matrix"] = confusion_matrix(y_test, pred, labels=list(range(n_classes))).tolist()
    else:
        metrics = regression_metrics(y_test, pred)
        k = min(len(y_test), 400)
        details["pred_vs_actual"] = [[float(a), float(b)] for a, b in zip(np.asarray(y_test)[:k], np.asarray(pred)[:k])]
    return {
        "metrics": metrics,
        "primary_score": primary_from(metrics, metric),
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_std": float(np.std(cv_scores)) if len(cv_scores) > 1 else None,
        "train_seconds": round(fit_seconds, 3),
        "details": details,
        "model": model,
    }


def _take(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def rank_results(results: list[dict], metric: str) -> list[dict]:
    """Ranks completed results by CV mean when available (more stable), else by test score."""
    hib = higher_is_better(metric)
    done = [r for r in results if r.get("status") == "completed" and r.get("primary_score") is not None]

    def key(r):
        v = r["cv_mean"] if r.get("cv_mean") is not None else r["primary_score"]
        return -v if hib else v

    done.sort(key=key)
    for i, r in enumerate(done, 1):
        r["rank"] = i
    return done
