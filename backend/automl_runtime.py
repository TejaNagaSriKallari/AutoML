"""Runtime components that live inside every saved model pipeline.

This module is intentionally standalone (no imports from `app`) because it is shipped
inside every model download: a user can `joblib.load("model.joblib")` anywhere as long as
this file sits next to their script.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

DATE_PARTS = ("year", "month", "day", "dayofweek", "hour")
_NUM_CLEAN = re.compile(r"[,\s$€£₹%]")


def coerce_numeric(series: pd.Series) -> pd.Series:
    """'1,200', '$3.5', ' 45% ' -> floats. Unparseable values become NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype("string").str.replace(_NUM_CLEAN, "", regex=True), errors="coerce")


class DataCleaner(BaseEstimator, TransformerMixin):
    """Applies the column decisions made at profiling time, identically at train and predict time."""

    def __init__(self, numeric=(), categorical=(), datetime=(), text=()):
        self.numeric = numeric
        self.categorical = categorical
        self.datetime = datetime
        self.text = text

    def fit(self, X, y=None):
        return self

    def output_columns(self) -> dict[str, list[str]]:
        date_cols = [f"{c}__{p}" for c in self.datetime for p in DATE_PARTS]
        return {"numeric": list(self.numeric) + date_cols, "categorical": list(self.categorical), "text": list(self.text)}

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        for c in self.numeric:
            out[c] = coerce_numeric(X[c]) if c in X else np.nan
        for c in self.datetime:
            d = pd.to_datetime(X[c], errors="coerce", format="mixed") if c in X else pd.Series(pd.NaT, index=X.index)
            for p in DATE_PARTS:
                out[f"{c}__{p}"] = getattr(d.dt, p).astype(float)
        for c in self.categorical:
            col = X[c] if c in X else pd.Series(np.nan, index=X.index)
            out[c] = col.astype("string").fillna("__missing__").astype(object)
        for c in self.text:
            col = X[c] if c in X else pd.Series("", index=X.index)
            out[c] = col.fillna("").astype(str)
        return out


def build_tabular_preprocessor(columns: dict[str, list[str]], scaler: str = "standard") -> Pipeline:
    """columns: {'numeric': [...], 'categorical': [...], 'datetime': [...], 'text': [...]}"""
    cleaner = DataCleaner(
        numeric=columns.get("numeric", []),
        categorical=columns.get("categorical", []),
        datetime=columns.get("datetime", []),
        text=columns.get("text", []),
    )
    out = cleaner.output_columns()
    scale = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}[scaler]
    transformers = []
    if out["numeric"]:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                      ("scale", scale)]),
            out["numeric"],
        ))
    if out["categorical"]:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, min_frequency=2,
                          sparse_output=False),
            out["categorical"],
        ))
    for c in out["text"]:
        transformers.append((f"txt_{c}", TfidfVectorizer(max_features=100, sublinear_tf=True), c))
    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline([("clean", cleaner), ("encode", ct)])


# --------------------------------------------------------------------------------------
# Embedding transformers (Option 1: frozen pretrained models on CPU)
# --------------------------------------------------------------------------------------

class TimmImageEmbedder(BaseEstimator, TransformerMixin):
    """Turns image paths (or PIL images) into feature vectors with a frozen pretrained timm backbone."""

    def __init__(self, model_name: str = "resnet50", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._transform = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_transform"] = None
        return state

    def _load(self):
        if self._model is None:
            import timm
            import torch

            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            model.eval()
            cfg = timm.data.resolve_data_config({}, model=model)
            self._transform = timm.data.create_transform(**cfg)
            self._model = model
            torch.set_grad_enabled(False)
        return self._model, self._transform

    def fit(self, X, y=None):
        return self

    def transform(self, X, progress=None):
        import torch
        from PIL import Image

        model, tf = self._load()
        items = list(X)
        feats = []
        for start in range(0, len(items), self.batch_size):
            batch = []
            for item in items[start:start + self.batch_size]:
                img = item if isinstance(item, Image.Image) else Image.open(Path(item))
                batch.append(tf(img.convert("RGB")))
            with torch.inference_mode():
                feats.append(model(torch.stack(batch)).float().numpy())
            if progress:
                progress(min(start + self.batch_size, len(items)), len(items))
        return np.concatenate(feats) if feats else np.zeros((0, model.num_features), dtype=np.float32)


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Turns raw texts into sentence embeddings with a frozen pretrained sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        texts = [("" if t is None else str(t)) for t in X]
        return self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)


class TextColumn(BaseEstimator, TransformerMixin):
    """Selects one text column from a DataFrame (so text pipelines accept the original CSV)."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.column]
        return pd.Series(X).fillna("").astype(str).to_numpy()


class LabelDecoder:
    """Wraps a fitted pipeline so predictions come back as the original class names."""

    def __init__(self, pipeline, classes):
        self.pipeline = pipeline
        self.classes_ = np.asarray(classes, dtype=object)

    def predict(self, X):
        return self.classes_[np.asarray(self.pipeline.predict(X)).astype(int)]

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
