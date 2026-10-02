from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

TaskName = Literal[
    "classification", "regression", "clustering", "forecasting",
    "text_classification", "image_classification", "object_detection",
]


class TaskSpec(BaseModel):
    task: TaskName
    target: str | None = None
    text_column: str | None = None
    date_column: str | None = None
    drop_columns: list[str] = Field(default_factory=list)
    metric: str
    horizon: int | None = None
    season_length: int | None = None
    n_clusters: int | None = None
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    source: Literal["llm", "rules", "user"] = "rules"
    confidence: float = 0.5


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    name: str = Field(default="", max_length=200)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str


class ProjectOut(BaseModel):
    id: str
    name: str
    requirement: str
    dataset_filename: str
    dataset_kind: str
    dataset_size_bytes: int
    status: str
    error: str | None
    profile: dict | None
    spec: dict | None
    spec_confirmed: bool
    created_at: datetime
    latest_run: dict | None = None


class SpecUpdate(BaseModel):
    task: TaskName
    target: str | None = None
    text_column: str | None = None
    date_column: str | None = None
    drop_columns: list[str] = Field(default_factory=list)
    metric: str
    horizon: int | None = Field(default=None, ge=1, le=10_000)
    season_length: int | None = Field(default=None, ge=2, le=10_000)
    n_clusters: int | None = Field(default=None, ge=2, le=100)


class RunCreate(BaseModel):
    backend: Literal["cpu", "colab"]
    model_keys: list[str] = Field(min_length=1, max_length=500)
    custom_models: list[str] = Field(default_factory=list, max_length=20)
    time_limit_per_model: int = Field(default=300, ge=10, le=7200)
    tune_top_k: int = Field(default=0, ge=0, le=5)
    epochs: int = Field(default=10, ge=1, le=300)  # colab deep-learning runs
    max_images: int | None = Field(default=None, ge=50)


class ModelResultOut(BaseModel):
    id: str
    model_key: str
    name: str
    family: str
    status: str
    metrics: dict[str, Any]
    primary_score: float | None
    cv_mean: float | None
    cv_std: float | None
    train_seconds: float | None
    params: dict[str, Any]
    details: dict[str, Any]
    error: str | None
    rank: int | None
    is_best: bool


class RunOut(BaseModel):
    id: str
    project_id: str
    backend: str
    status: str
    config: dict
    spec: dict
    progress: dict
    insights: dict
    summary: str | None
    error: str | None
    has_model: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    results: list[ModelResultOut]


class ColabResultIn(BaseModel):
    model_key: str = Field(max_length=300)
    name: str = Field(max_length=300)
    status: Literal["completed", "failed", "timeout", "running"] = "completed"
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    train_seconds: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=5000)


class ColabStatusIn(BaseModel):
    status: Literal["running", "completed", "failed"]
    message: str = Field(default="", max_length=2000)
    done: int | None = None
    total: int | None = None
