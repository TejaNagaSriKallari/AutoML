import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

_tmp = tempfile.mkdtemp(prefix="automl-test-", dir=os.environ.get("AUTOML_TEST_TMP"))
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = ""
os.environ["RUN_WORKER_IN_API"] = "false"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["WORKER_PROCESSES"] = "0"  # auto: sized from free memory
os.environ["PUBLIC_API_URL"] = "http://testserver"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def user(client):
    r = client.post("/api/auth/register", json={"email": "owner@example.com", "password": "correct horse", "name": "Owner"})
    assert r.status_code == 200, r.text
    return r.json()


def wait_ready(client, project_id):
    p = client.get(f"/api/projects/{project_id}").json()
    assert p["status"] == "ready", p
    return p


def execute_pending():
    """Runs the queued CPU job synchronously (what the worker process does)."""
    from app.training.runner import execute_run
    from app.worker import claim_next

    run_id = claim_next()
    assert run_id, "no queued run"
    execute_run(run_id)
    return run_id
