from __future__ import annotations

import math
import time
from pathlib import Path

from app.db import SessionLocal
from app.models import ModelResult, Run


def _clean(v):
    """JSON columns reject NaN/inf; convert them to None recursively."""
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item"):  # numpy scalar
        return _clean(v.item())
    return v


class RunContext:
    """What a trainer needs from the platform: inputs, progress reporting, result storage."""

    def __init__(self, run: Run, dataset_kind: str, data_path: str, profile: dict, run_dir: Path):
        self.run_id = run.id
        self.project_id = run.project_id
        self.spec = run.spec
        self.config = run.config
        self.dataset_kind = dataset_kind
        self.data_path = data_path
        self.profile = profile
        self.run_dir = run_dir
        self.model_paths: dict[str, str] = {}
        self.insights: dict = {"warnings": []}
        self._last_cancel_check = 0.0
        self._cancelled = False

    def progress(self, stage: str, message: str, done: int | None = None, total: int | None = None) -> None:
        with SessionLocal() as db:
            run = db.get(Run, self.run_id)
            run.progress = _clean({"stage": stage, "message": message, "done": done, "total": total})
            db.commit()

    def warn(self, message: str) -> None:
        if message not in self.insights["warnings"]:
            self.insights["warnings"].append(message)
            self.save_insights()

    def set_insight(self, key: str, value) -> None:
        self.insights[key] = value
        self.save_insights()

    def save_insights(self) -> None:
        with SessionLocal() as db:
            run = db.get(Run, self.run_id)
            run.insights = _clean(dict(self.insights))
            db.commit()

    def add_result(self, *, model_key: str, name: str, family: str, status: str, metrics: dict | None = None,
                   primary_score: float | None = None, cv_mean: float | None = None, cv_std: float | None = None,
                   train_seconds: float | None = None, params: dict | None = None, details: dict | None = None,
                   error: str | None = None, model_path: str | None = None) -> None:
        if model_path:
            self.model_paths[model_key] = model_path
        with SessionLocal() as db:
            db.add(ModelResult(
                run_id=self.run_id, model_key=model_key, name=name, family=family, status=status,
                metrics=_clean(metrics or {}), primary_score=_clean(primary_score), cv_mean=_clean(cv_mean),
                cv_std=_clean(cv_std), train_seconds=train_seconds, params=_clean(params or {}),
                details=_clean(details or {}), error=error,
            ))
            db.commit()

    def cancelled(self) -> bool:
        if self._cancelled:
            return True
        now = time.monotonic()
        if now - self._last_cancel_check < 1.5:
            return False
        self._last_cancel_check = now
        with SessionLocal() as db:
            self._cancelled = db.get(Run, self.run_id).status == "cancelling"
        return self._cancelled

    def time_limit(self) -> int:
        return int(self.config.get("time_limit_per_model", 300))
