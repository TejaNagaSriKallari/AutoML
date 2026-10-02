import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import init_db
from app.routers import auth, colab_api, projects, runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    stop = None
    if get_settings().run_worker_in_api:
        from app.worker import recover_stale, start_in_background

        recover_stale()
        stop = start_in_background()
    yield
    if stop:
        stop.set()


app = FastAPI(title="AutoML Studio API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Colab notebooks call /api/colab/* from another origin, but server-to-server via `requests`,
# so they do not need CORS; they authenticate with the run token header instead of cookies.

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(runs.router)
app.include_router(colab_api.router)


@app.get("/api/health")
def health():
    return {"ok": True}
