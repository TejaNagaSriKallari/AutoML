import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # Core
    secret_key: str = "change-me-in-production"
    data_dir: Path = BACKEND_DIR / "data"
    database_url: str = ""  # empty = SQLite file inside data_dir; use postgresql+psycopg://... in production
    # Where pretrained weights (Hugging Face / timm / torch hub) are cached. They are large.
    model_cache_dir: Path | None = None
    frontend_origin: str = "http://localhost:3000"
    cookie_secure: bool = False
    access_token_days: int = 7

    # Public URL of this API, embedded in Colab notebooks so they can download the
    # dataset and stream results back. Colab cannot reach "localhost": in local dev,
    # expose the API with a tunnel (e.g. `cloudflared tunnel --url http://localhost:8000`)
    # and set this to the tunnel URL. When unreachable, notebooks fall back to manual mode.
    public_api_url: str = "http://localhost:8000"

    # Claude (requirement understanding + result explanations). Optional: without a key
    # the app falls back to rule-based task detection.
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5"

    # Limits for real users
    max_upload_mb: int = 2048
    max_active_runs_per_user: int = 2
    max_zip_members: int = 200_000
    max_extracted_gb: float = 10.0

    # Training defaults
    worker_processes: int = 0  # 0 = auto (half of CPU cores)
    default_time_limit_per_model: int = 300  # seconds
    max_tabular_rows: int = 200_000
    max_cpu_images: int = 8_000
    run_worker_in_api: bool = True  # dev convenience: run the job worker inside the API process
    colab_token_days: int = 7
    # Small free servers (e.g. Render free, 512 MB) install requirements-lite.txt without PyTorch:
    # text and image tasks then train on Colab only, and slow models start unselected.
    lite_mode: bool = False


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    if not s.database_url:
        s.database_url = f"sqlite:///{(s.data_dir / 'app.db').as_posix()}"
    # Hosted Postgres (Neon, Render, ...) hands out postgres:// or postgresql:// URLs;
    # SQLAlchemy needs the driver named to use psycopg 3.
    # Render sets RENDER_EXTERNAL_URL to the service's public URL, which Colab notebooks can reach.
    if "PUBLIC_API_URL" not in os.environ and os.environ.get("RENDER_EXTERNAL_URL"):
        s.public_api_url = os.environ["RENDER_EXTERNAL_URL"]
    for prefix in ("postgres://", "postgresql://"):
        if s.database_url.startswith(prefix):
            s.database_url = "postgresql+psycopg://" + s.database_url[len(prefix):]
    if s.model_cache_dir:
        s.model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(s.model_cache_dir / "hf"))
        os.environ.setdefault("TORCH_HOME", str(s.model_cache_dir / "torch"))
    return s
