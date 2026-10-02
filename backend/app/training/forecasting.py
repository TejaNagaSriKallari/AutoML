"""Single-series time-series forecasting. Every model forecasts the same final `horizon` points
(never seen during fitting), then is refitted on the full series to forecast the future."""
from __future__ import annotations

import itertools
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from app.profiling import infer_frequency, season_length_for
from app.registry import get_model
from app.shared_data import load_table
from app.training.context import RunContext
from app.training.pool import Job, run_jobs
from automl_runtime import coerce_numeric

_BUNDLES: dict[str, dict] = {}


def _bundle(path: str) -> dict:
    if path not in _BUNDLES:
        _BUNDLES.clear()
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def forecast_metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = a - p
    m = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}
    nz = np.abs(a) > 1e-9
    if nz.mean() > 0.9:
        m["mape"] = float(np.mean(np.abs(err[nz] / a[nz])) * 100)
    denom = np.abs(a) + np.abs(p)
    ok = denom > 1e-9
    m["smape"] = float(np.mean(2 * np.abs(err[ok]) / denom[ok]) * 100) if ok.any() else 0.0
    return m


# --------------------------------------------------------------------------------------
# Models: each is fn(y_train, h, ctx) -> forecast of length h
# --------------------------------------------------------------------------------------

def _naive(y, h, c):
    return np.repeat(y[-1], h)


def _seasonal_naive(y, h, c):
    s = c["season"]
    last = y[-s:]
    return np.array([last[i % s] for i in range(h)])


def _drift(y, h, c):
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return y[-1] + slope * np.arange(1, h + 1)


def _moving_average(y, h, c):
    w = min(len(y), c["season"] or 7)
    return np.repeat(y[-w:].mean(), h)


def _ses(y, h, c):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    return SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(h)


def _holt(y, h, c):
    from statsmodels.tsa.holtwinters import Holt

    return Holt(y, damped_trend=True, initialization_method="estimated").fit().forecast(h)


def _holt_winters(y, h, c):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    s = c["season"]
    if len(y) < 2 * s + 2:
        raise ValueError(f"Holt-Winters needs at least two full seasons ({2 * s} points).")
    return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=s,
                                initialization_method="estimated").fit().forecast(h)


def _theta(y, h, c):
    from statsmodels.tsa.forecasting.theta import ThetaModel

    s = c["season"]
    use_season = bool(s and len(y) >= 2 * s)
    model = ThetaModel(y, period=s if use_season else 1, deseasonalize=use_season)
    return np.asarray(model.fit().forecast(h))


def _arima(y, h, c):
    from statsmodels.tsa.arima.model import ARIMA

    best = None
    for p, d, q in itertools.product((0, 1, 2), (0, 1), (0, 1, 2)):
        if time.monotonic() > c["deadline"] and best is not None:
            break
        try:
            fit = ARIMA(y, order=(p, d, q)).fit()
        except Exception:
            continue
        if best is None or fit.aic < best[0]:
            best = (fit.aic, (p, d, q), fit)
    if best is None:
        raise ValueError("No ARIMA order could be fitted.")
    c["chosen"] = {"order": list(best[1])}
    return np.asarray(best[2].forecast(h))


def _lag_regressor(name):
    def build():
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        if name == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                                     n_jobs=1, verbose=-1, random_state=42)
        if name == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, n_jobs=1, random_state=42)
        return {
            "linear": lambda: LinearRegression(),
            "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "lasso": lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000)),
            "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=42),
            "extra_trees": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=42),
            "hist_gb": lambda: HistGradientBoostingRegressor(random_state=42),
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
            "svr": lambda: make_pipeline(StandardScaler(), SVR(C=10.0)),
        }[name]()
    return build


def _lag_features(history: np.ndarray, n_lags: int, season: int | None, t_index: int, dates) -> list[float]:
    lags = history[-n_lags:][::-1].tolist()
    feats = lags + [history[-3:].mean(), history[-n_lags:].mean(), history[-n_lags:].std()]
    if season and len(history) >= season:
        feats.append(history[-season])
    feats.append(float(t_index))
    if dates is not None:
        d = dates[t_index]
        feats += [d.month, d.dayofweek, d.day, d.quarter]
    return feats


def _lag_model(name):
    def fit_forecast(y, h, c):
        season = c["season"]
        n_lags = int(min(max(season or 0, 12), max(3, len(y) // 3)))
        dates = c["dates"]
        offset = c.get("offset", 0)  # index of y[0] within the full date index
        X, T = [], []
        for t in range(n_lags + (season or 0) if season and season > n_lags else n_lags, len(y)):
            X.append(_lag_features(y[:t], n_lags, season, offset + t, dates))
            T.append(y[t])
        if len(X) < 10:
            raise ValueError("Series too short for lag-feature models.")
        # model the change from the last value: lets tree models extrapolate trends
        X, T = np.array(X), np.array(T)
        base = X[:, 0]
        model = _lag_regressor(name)()
        model.fit(X, T - base)
        hist = list(y)
        out = []
        for i in range(h):
            f = _lag_features(np.array(hist), n_lags, season, offset + len(y) + i, dates)
            nxt = float(model.predict(np.array([f]))[0] + f[0])
            out.append(nxt)
            hist.append(nxt)
        return np.array(out)
    return fit_forecast


MODELS = {
    "naive": _naive, "seasonal_naive": _seasonal_naive, "drift": _drift, "moving_average": _moving_average,
    "ses": _ses, "holt": _holt, "holt_winters": _holt_winters, "theta": _theta, "arima": _arima,
    **{f"lag.{n}": _lag_model(n) for n in ("linear", "ridge", "random_forest", "extra_trees", "hist_gb",
                                           "lightgbm", "xgboost", "knn", "svr", "lasso")},
}


def fit_forecaster(bundle_path: str, model_key: str, time_limit: float) -> dict:
    warnings.filterwarnings("ignore")
    b = _bundle(bundle_path)
    name = model_key.split("/", 2)[2]
    fn = MODELS[name]
    y, h = b["y"], b["horizon"]
    y_train, y_test = y[:-h], y[-h:]
    c = {"season": b["season"], "dates": b["all_dates"], "deadline": time.monotonic() + time_limit * 0.4}
    t0 = time.perf_counter()
    pred = np.asarray(fn(y_train, h, c), dtype=float)
    secs = time.perf_counter() - t0
    if not np.all(np.isfinite(pred)):
        raise ValueError("The model produced non-finite forecasts.")
    metrics = forecast_metrics(y_test, pred)
    c2 = {"season": b["season"], "dates": b["all_dates"], "deadline": time.monotonic() + time_limit * 0.4}
    future = np.asarray(fn(y, h, c2), dtype=float)
    return {
        "metrics": metrics, "primary_score": metrics.get(b["metric"], metrics["mae"]), "cv_mean": None, "cv_std": None,
        "train_seconds": round(secs, 3),
        "details": {"test_pred": pred.tolist(), "future_pred": future.tolist()},
        "best_params": c.get("chosen", {}),
    }


def run_forecasting(ctx: RunContext, model_keys: list[str]) -> None:
    spec = ctx.spec
    ctx.progress("preparing", "Loading and ordering the series")
    df = load_table(ctx.data_path)
    dates = pd.to_datetime(df[spec["date_column"]], errors="coerce", format="mixed")
    values = coerce_numeric(df[spec["target"]])
    s = pd.DataFrame({"date": dates, "y": values}).dropna()
    if s["date"].duplicated().any():
        ctx.warn("Several rows share the same date; they were summed into one value per date. "
                 "Forecasting each sub-series separately (e.g. per store) is not supported yet.")
        s = s.groupby("date", as_index=False)["y"].sum()
    s = s.sort_values("date").set_index("date")["y"]
    freq = infer_frequency(s.index.to_series())
    if freq:
        try:
            full = s.asfreq(freq)
            missing = int(full.isna().sum())
            if missing:
                ctx.warn(f"{missing} missing time steps were filled by interpolation.")
                full = full.interpolate(limit_direction="both")
            s = full
        except ValueError:
            freq = None
    if len(s) < 12:
        raise ValueError(f"Only {len(s)} time points after cleaning; forecasting needs at least 12.")

    season = spec.get("season_length") or season_length_for(freq)
    if season and len(s) < 2 * season + 4:
        season = None
    horizon = int(spec.get("horizon") or 12)
    if horizon > len(s) // 4:
        new_h = max(1, len(s) // 4)
        ctx.warn(f"Horizon {horizon} is too long for {len(s)} points; evaluated on the last {new_h} points instead.")
        horizon = new_h

    future_dates = (pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:] if freq
                    else pd.RangeIndex(len(s), len(s) + horizon))
    all_dates = s.index.append(future_dates) if freq else None
    bundle = {"y": s.to_numpy(dtype=float), "horizon": horizon, "season": season, "metric": spec["metric"],
              "all_dates": all_dates}
    path = ctx.run_dir / "bundle.joblib"
    joblib.dump(bundle, path)
    tail = min(len(s), max(120, horizon * 6))
    ctx.set_insight("data", {
        "n_points": int(len(s)), "frequency": freq, "season_length": season, "horizon": horizon,
        "start": str(s.index[0]), "end": str(s.index[-1]),
    })
    ctx.set_insight("series", {
        "history_dates": [str(d) for d in s.index[-tail:]], "history": s.to_numpy()[-tail:].tolist(),
        "test_dates": [str(d) for d in s.index[-horizon:]],
        "future_dates": [str(d) for d in future_dates],
    })

    keys = [k for k in model_keys if k.split("/", 2)[2] in MODELS]
    if not season:
        dropped = [k for k in keys if k.endswith(("seasonal_naive", "holt_winters"))]
        keys = [k for k in keys if k not in dropped]
        for k in dropped:
            spec_ = get_model(k)
            ctx.add_result(model_key=k, name=spec_.name, family=spec_.family, status="skipped",
                           error="No seasonality detected in this series.")
    total, done = len(keys), 0
    ctx.progress("training", f"Training {total} forecasting models", 0, total)

    def on_result(key, res, err, status):
        nonlocal done
        done += 1
        m = get_model(key)
        if res:
            ctx.add_result(model_key=key, name=m.name, family=m.family, status="completed", metrics=res["metrics"],
                           primary_score=res["primary_score"], train_seconds=res["train_seconds"],
                           params=res["best_params"], details=res["details"])
        else:
            ctx.add_result(model_key=key, name=m.name, family=m.family, status=status, error=err)
        ctx.progress("training", f"Finished {m.name}", done, total)

    run_jobs([Job(k, fit_forecaster, (str(path), k, ctx.time_limit()), timeout=ctx.time_limit()) for k in keys],
             on_result, ctx.cancelled)
