"""Optuna hyperparameter search for the best-performing tabular models."""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from app.training.evaluate import classification_metrics, evaluate_supervised, higher_is_better, regression_metrics


def _rf(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 16, 24]),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
            "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0])}


def _hgb(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-6, 10, log=True)}


def _gb(t):
    return {"n_estimators": t.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "subsample": t.suggest_float("subsample", 0.5, 1.0)}


def _xgb(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def _lgbm(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 256, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 100),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def _cat(t):
    return {"iterations": t.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": t.suggest_int("depth", 4, 10),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}


def _logreg(t):
    return {"C": t.suggest_float("C", 1e-3, 100, log=True)}


def _svm(t):
    return {"C": t.suggest_float("C", 1e-2, 100, log=True), "gamma": t.suggest_float("gamma", 1e-4, 1, log=True)}


def _knn(t):
    return {"n_neighbors": t.suggest_int("n_neighbors", 1, 50), "weights": t.suggest_categorical("weights", ["uniform", "distance"])}


def _ridge(t):
    return {"alpha": t.suggest_float("alpha", 1e-3, 100, log=True)}


def _enet(t):
    return {"alpha": t.suggest_float("alpha", 1e-4, 10, log=True), "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0)}


def _mlp(t):
    return {"hidden_layer_sizes": t.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (256,), (128, 64), (256, 128)]),
            "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}


def _tree(t):
    return {"max_depth": t.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20]),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30)}


# keyed by the last segment of the model key (e.g. "classification/cpu/xgboost" -> "xgboost")
SEARCH_SPACES = {
    "sk.RandomForestClassifier": _rf, "sk.RandomForestRegressor": _rf,
    "sk.ExtraTreesClassifier": _rf, "sk.ExtraTreesRegressor": _rf,
    "sk.HistGradientBoostingClassifier": _hgb, "sk.HistGradientBoostingRegressor": _hgb,
    "sk.GradientBoostingClassifier": _gb, "sk.GradientBoostingRegressor": _gb,
    "xgboost": _xgb, "lightgbm": _lgbm, "catboost": _cat,
    "sk.LogisticRegression": _logreg, "sk.SVC": _svm, "sk.SVR": _svm,
    "sk.KNeighborsClassifier": _knn, "sk.KNeighborsRegressor": _knn,
    "sk.Ridge": _ridge, "sk.RidgeClassifier": _ridge, "sk.Lasso": _ridge, "sk.ElasticNet": _enet,
    "sk.MLPClassifier": _mlp, "sk.MLPRegressor": _mlp,
    "sk.DecisionTreeClassifier": _tree, "sk.DecisionTreeRegressor": _tree,
}


def is_tunable(model_key: str) -> bool:
    return model_key.rsplit("/", 1)[-1] in SEARCH_SPACES


def tune_and_evaluate(pipe, model_key: str, b: dict, time_budget: float, max_trials: int = 40) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = SEARCH_SPACES[model_key.rsplit("/", 1)[-1]]
    prefix = "model__regressor__" if hasattr(pipe.named_steps["model"], "regressor") else "model__"
    task, metric, n_classes = b["task"], b["metric"], b["n_classes"]
    X, y = b["X"], b["y"]
    train_idx = np.asarray(b["train_idx"])
    if len(train_idx) > 50_000:  # tune on a sample; final fit uses all training rows
        train_idx = np.random.default_rng(0).choice(train_idx, 50_000, replace=False)
    Xt, yt = X.iloc[train_idx], y[train_idx]
    is_clf = task == "classification"
    splitter = StratifiedKFold(3, shuffle=True, random_state=7) if is_clf else KFold(3, shuffle=True, random_state=7)
    folds = list(splitter.split(np.zeros(len(yt)), yt))
    sign = 1 if higher_is_better(metric) else -1

    def objective(trial):
        params = {f"{prefix}{k}": v for k, v in space(trial).items()}
        scores = []
        for tr, va in folds:
            m = clone(pipe).set_params(**params)
            m.fit(Xt.iloc[tr], yt[tr])
            pred = m.predict(Xt.iloc[va])
            if is_clf:
                s = classification_metrics(yt[va], pred, None, n_classes).get(metric)
                if s is None:  # roc_auc/log_loss need scores; fall back to balanced accuracy
                    s = classification_metrics(yt[va], pred, None, n_classes)["balanced_accuracy"]
            else:
                s = regression_metrics(yt[va], pred)[metric]
            scores.append(s)
        return sign * float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t_end = time.monotonic() + time_budget
    study.optimize(objective, n_trials=max_trials, timeout=time_budget, catch=(Exception,))
    best_params = study.best_params if study.best_trial else {}
    tuned = clone(pipe).set_params(**{f"{prefix}{k}": v for k, v in best_params.items()})
    res = evaluate_supervised(tuned, X, y, b["train_idx"], b["test_idx"], task, metric, n_classes=n_classes,
                              cv_folds=b["cv_folds"], deadline=max(t_end, time.monotonic() + 30))
    res["best_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}
    res["details"]["n_trials"] = len(study.trials)
    return res
