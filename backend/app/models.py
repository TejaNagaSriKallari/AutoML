import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    password_hash: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    projects: Mapped[list["Project"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    requirement: Mapped[str] = mapped_column(Text)
    dataset_filename: Mapped[str] = mapped_column(String(500))
    upload_path: Mapped[str] = mapped_column(String(1000))  # the original file (served to Colab)
    dataset_path: Mapped[str] = mapped_column(String(1000), default="")  # what training reads
    dataset_kind: Mapped[str] = mapped_column(String(40))  # table | image_folder | yolo_detection
    dataset_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    # analyzing -> ready | failed
    status: Mapped[str] = mapped_column(String(20), default="analyzing")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    spec: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    spec_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="projects")
    runs: Mapped[list["Run"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Run.created_at.desc()"
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    backend: Mapped[str] = mapped_column(String(20))  # cpu | colab
    # cpu:   queued -> running -> completed | failed | cancelled   (cancelling in between)
    # colab: awaiting_colab -> running -> completed | failed
    status: Mapped[str] = mapped_column(String(20), index=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)  # snapshot of the task spec at launch
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    insights: Mapped[dict] = mapped_column(JSON, default=dict)  # warnings, leakage, feature importance...
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    best_model_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    colab_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    colab_token_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="runs")
    results: Mapped[list["ModelResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="ModelResult.rank"
    )


class ModelResult(Base):
    __tablename__ = "model_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    model_key: Mapped[str] = mapped_column(String(300))
    name: Mapped[str] = mapped_column(String(300))
    family: Mapped[str] = mapped_column(String(60), default="")
    status: Mapped[str] = mapped_column(String(20))  # completed | failed | timeout | skipped
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    primary_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cv_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    cv_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    details: Mapped[dict] = mapped_column(JSON, default=dict)  # confusion matrix, etc.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[Run] = relationship(back_populates="results")
