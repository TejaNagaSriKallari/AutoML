import io
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import execute_pending, wait_ready


def upload(client, name, requirement, filename, data: bytes):
    r = client.post("/api/projects", data={"name": name, "requirement": requirement},
                    files={"file": (filename, data)})
    assert r.status_code == 200, r.text
    return wait_ready(client, r.json()["id"])


def selected(client, project_id, backend="cpu", limit=None):
    c = client.get(f"/api/projects/{project_id}/candidates").json()
    keys = [m["key"] for m in c["backends"][backend] if m["selected"]]
    return keys[:limit] if limit else keys


def start_cpu(client, project_id, keys, **cfg):
    r = client.post(f"/api/projects/{project_id}/runs",
                    json={"backend": "cpu", "model_keys": keys, "time_limit_per_model": 120, **cfg})
    assert r.status_code == 200, r.text
    run_id = execute_pending()
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "completed", (run["error"], run["insights"].get("traceback"))
    return run


def csv_bytes(df):
    return df.to_csv(index=False).encode()


# --------------------------------------------------------------------------------------

def test_auth_required_and_isolation(client, user):
    from fastapi.testclient import TestClient

    from app.main import app

    df = pd.DataFrame({"a": range(30), "b": ["x", "y"] * 15})
    p = upload(client, "iso", "predict b from a", "d.csv", csv_bytes(df))
    other = TestClient(app)
    assert other.get("/api/projects").status_code == 401
    other.post("/api/auth/register", json={"email": "intruder@example.com", "password": "password123"})
    assert other.get(f"/api/projects/{p['id']}").status_code == 404
    assert client.post("/api/auth/login", json={"email": "owner@example.com", "password": "wrong-pass"}).status_code == 401


def test_tabular_classification_full_flow(client, user):
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    df = d.frame.copy()
    df["diagnosis"] = np.where(df.pop("target") == 1, "benign", "malignant")
    rng = np.random.default_rng(0)
    df["clinic"] = rng.choice(["north", "south", "east"], len(df))
    df["visit_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")
    df["patient_id"] = [f"P{i:05d}" for i in range(len(df))]
    df["followup_code"] = np.where(df["diagnosis"] == "benign", "B-ok", "M-referral")  # leaks the answer
    df.loc[rng.choice(len(df), 40, replace=False), "mean radius"] = np.nan

    p = upload(client, "Tumours", "I want to predict the diagnosis of each tumour", "tumours.csv", csv_bytes(df))
    spec = p["spec"]
    assert spec["task"] == "classification" and spec["target"] == "diagnosis"
    assert "patient_id" in spec["drop_columns"]
    kinds = {c["name"]: c["kind"] for c in p["profile"]["columns"]}
    assert kinds["visit_date"] == "datetime" and kinds["clinic"] == "categorical" and kinds["patient_id"] == "id"

    keys = ["classification/cpu/sk.LogisticRegression", "classification/cpu/sk.RandomForestClassifier",
            "classification/cpu/lightgbm", "classification/cpu/sk.DummyClassifier",
            "classification/cpu/sk.MultinomialNB", "classification/cpu/xgboost"]
    run = start_cpu(client, p["id"], keys, tune_top_k=1)
    done = [r for r in run["results"] if r["status"] == "completed"]
    assert len(done) == 7, [(r["name"], r["error"]) for r in run["results"]]  # 6 + 1 tuned
    assert any(r["model_key"].endswith("+tuned") for r in done)
    best = run["results"][0]
    assert best["rank"] == 1 and best["is_best"] and best["metrics"]["accuracy"] > 0.9
    assert {"accuracy", "f1_macro", "roc_auc"} <= set(best["metrics"])
    assert any(l["column"] == "followup_code" for l in run["insights"]["leakage"])
    assert run["insights"]["feature_importance"]
    assert run["summary"]

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/runs/{run['id']}/download").content))
    assert {"model.joblib", "automl_runtime.py", "predict_example.py", "metadata.json"} <= set(z.namelist())
    new = df.drop(columns=["diagnosis"]).head(20)
    r = client.post(f"/api/runs/{run['id']}/predict", files={"file": ("new.csv", csv_bytes(new))})
    assert r.status_code == 200, r.text
    preds = pd.read_csv(io.StringIO(r.text))["prediction"]
    assert set(preds) <= {"benign", "malignant"}


def test_all_classifiers_and_regressors_train(client, user):
    """Every auto-discovered model that is selected by default must train without crashing."""
    from sklearn.datasets import load_diabetes, load_wine

    wine = load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
    p = upload(client, "wine", "classify the cultivar", "wine.csv", csv_bytes(wine))
    run = start_cpu(client, p["id"], selected(client, p["id"]))
    bad = [(r["name"], r["status"], r["error"]) for r in run["results"] if r["status"] != "completed"]
    assert not bad, bad

    diab = load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"})
    p = upload(client, "diabetes", "predict disease progression", "diabetes.csv", csv_bytes(diab))
    assert p["spec"]["task"] == "regression"
    run = start_cpu(client, p["id"], selected(client, p["id"]))
    bad = [(r["name"], r["status"], r["error"]) for r in run["results"] if r["status"] != "completed"]
    assert not bad, bad
    assert run["results"][0]["metrics"]["r2"] > 0.35


def test_clustering(client, user):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=600, centers=4, n_features=5, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["region"] = np.random.default_rng(0).choice(["a", "b"], len(df))
    p = upload(client, "segments", "segment our customers into groups", "customers.csv", csv_bytes(df))
    assert p["spec"]["task"] == "clustering"
    run = start_cpu(client, p["id"], selected(client, p["id"]))
    best = run["results"][0]
    assert best["metrics"]["silhouette"] > 0.4
    out = pd.read_csv(io.StringIO(client.get(f"/api/runs/{run['id']}/download").text))
    assert "cluster" in out and len(out) == 600


def test_forecasting(client, user):
    dates = pd.date_range("2015-01-01", periods=96, freq="MS")
    t = np.arange(96)
    sales = 200 + 2 * t + 30 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(0).normal(0, 4, 96)
    df = pd.DataFrame({"month": dates.strftime("%Y-%m-%d"), "sales": sales.round(2)})
    p = upload(client, "sales", "forecast monthly sales for the next 6 months", "sales.csv", csv_bytes(df))
    spec = p["spec"]
    assert spec["task"] == "forecasting" and spec["horizon"] == 6 and spec["season_length"] == 12
    run = start_cpu(client, p["id"], selected(client, p["id"]))
    names = {r["model_key"].split("/")[-1]: r for r in run["results"]}
    assert names["holt_winters"]["status"] == "completed"
    best = run["results"][0]
    assert best["metrics"]["mape"] < 5, best
    fc = pd.read_csv(io.StringIO(client.get(f"/api/runs/{run['id']}/download").text))
    assert len(fc) == 6 and fc["date"].iloc[0].startswith("2023-01")


def test_text_classification(client, user):
    rng = np.random.default_rng(0)
    pos = ["great product, works perfectly", "absolutely love it", "excellent quality and fast delivery",
           "very happy with this purchase", "fantastic, would buy again", "superb value for money"]
    neg = ["terrible, broke after a day", "waste of money", "awful quality and slow delivery",
           "very disappointed with this", "do not buy, horrible", "the worst purchase ever"]
    rows = [(f"{rng.choice(pos)} {rng.choice(['!', '.', ' really', ' honestly'])}", "positive") for _ in range(120)]
    rows += [(f"{rng.choice(neg)} {rng.choice(['!', '.', ' really', ' honestly'])}", "negative") for _ in range(120)]
    df = pd.DataFrame(rows, columns=["review_text", "sentiment"]).sample(frac=1, random_state=0)
    p = upload(client, "reviews", "classify the sentiment of customer reviews", "reviews.csv", csv_bytes(df))
    assert p["spec"]["task"] == "text_classification" and p["spec"]["text_column"] == "review_text"
    keys = ["text_classification/cpu/enc:tfidf_word", "text_classification/cpu/enc:sentence-transformers/all-MiniLM-L6-v2",
            "text_classification/cpu/head:logreg", "text_classification/cpu/head:complement_nb",
            "text_classification/cpu/head:knn"]
    run = start_cpu(client, p["id"], keys)
    done = [r for r in run["results"] if r["status"] == "completed"]
    # tfidf x (logreg, nb) + MiniLM x (logreg, knn); knn needs dense, nb needs sparse
    assert len(done) == 4, [(r["name"], r["error"]) for r in run["results"]]
    assert run["results"][0]["metrics"]["accuracy"] > 0.9
    r = client.post(f"/api/runs/{run['id']}/predict",
                    files={"file": ("n.csv", csv_bytes(pd.DataFrame({"review_text": ["love it, excellent"]})))})
    assert r.status_code == 200, r.text
    assert pd.read_csv(io.StringIO(r.text))["prediction"][0] == "positive"


def _image_zip(n_per_class=24) -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for label, color in (("red_things", (200, 30, 30)), ("blue_things", (30, 30, 200))):
            for i in range(n_per_class):
                arr = np.clip(np.array(color) + rng.normal(0, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
                b = io.BytesIO()
                Image.fromarray(arr).save(b, format="PNG")
                z.writestr(f"my_dataset/{label}/img_{i}.png", b.getvalue())
        z.writestr("my_dataset/red_things/broken.png", b"not an image")
    return buf.getvalue()


def test_image_classification_cpu(client, user):
    p = upload(client, "colors", "tell red objects from blue objects", "images.zip", _image_zip())
    assert p["dataset_kind"] == "image_folder" and p["spec"]["task"] == "image_classification"
    keys = ["image_classification/cpu/backbone:mobilenetv3_large_100",
            "image_classification/cpu/head:logreg", "image_classification/cpu/head:knn"]
    run = start_cpu(client, p["id"], keys)
    done = [r for r in run["results"] if r["status"] == "completed"]
    assert len(done) == 2 and run["results"][0]["metrics"]["accuracy"] >= 0.9
    assert any("unreadable" in w for w in run["insights"]["warnings"])
    img = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), (210, 20, 20), np.uint8)).save(img, format="PNG")
    r = client.post(f"/api/runs/{run['id']}/predict", files={"file": ("x.png", img.getvalue())})
    assert r.status_code == 200, r.text
    assert pd.read_csv(io.StringIO(r.text))["prediction"][0] == "red_things"


def _compile_notebook(nb: dict):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(l for l in cell["source"] if not l.lstrip().startswith("!"))
            compile(src, "<cell>", "exec")


def test_colab_flow(client, user):
    p = upload(client, "colors-gpu", "classify the images", "images.zip", _image_zip(8))
    keys = selected(client, p["id"], "colab")
    assert "image_classification/colab/resnet50" in keys
    r = client.post(f"/api/projects/{p['id']}/runs", json={"backend": "colab", "model_keys": keys[:2],
                    "custom_models": ["resnet18.a1_in1k"], "epochs": 3})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "awaiting_colab"
    nb = client.get(f"/api/runs/{run['id']}/notebook").json()
    _compile_notebook(nb)
    cfg = "".join(nb["cells"][1]["source"])
    token = cfg.split("RUN_TOKEN = '")[1].split("'")[0]
    h = {"X-Run-Token": token}

    assert client.get(f"/api/colab/runs/{run['id']}/dataset", headers={"X-Run-Token": "nope"}).status_code == 401
    assert client.get(f"/api/colab/runs/{run['id']}/dataset", headers=h).content[:2] == b"PK"
    assert client.post(f"/api/colab/runs/{run['id']}/status", headers=h, json={"status": "running", "message": "go"}).status_code == 200
    for key, acc in ((keys[0], 0.91), (keys[1], 0.97), ("image_classification/colab/resnet18.a1_in1k", 0.88)):
        body = {"model_key": key, "name": key.split("/")[-1], "status": "completed",
                "metrics": {"accuracy": acc, "f1_macro": acc - 0.01}, "train_seconds": 60}
        assert client.post(f"/api/colab/runs/{run['id']}/results", headers=h, json=body).status_code == 200
    bad = {"model_key": "image_classification/colab/evil", "name": "x", "metrics": {}}
    assert client.post(f"/api/colab/runs/{run['id']}/results", headers=h, json=bad).status_code == 400
    client.post(f"/api/colab/runs/{run['id']}/status", headers=h, json={"status": "completed"})
    run = client.get(f"/api/runs/{run['id']}").json()
    assert run["status"] == "completed"
    assert run["results"][0]["model_key"] == keys[1] and run["results"][0]["is_best"]

    # manual mode: import results.json
    r = client.post(f"/api/projects/{p['id']}/runs", json={"backend": "colab", "model_keys": keys[:1]})
    run2 = r.json()
    results = {"run_id": run2["id"], "results": [{"model_key": keys[0], "name": "ResNet-50", "status": "completed",
                                                  "metrics": {"accuracy": 0.9}}]}
    r = client.post(f"/api/runs/{run2['id']}/import-results",
                    files={"file": ("results.json", json.dumps(results).encode())})
    assert r.status_code == 200 and r.json()["status"] == "completed"


def test_text_and_detection_notebooks_compile():
    from app import colab

    nb = colab.text_classification_notebook("r", "t", "http://x", "d.csv", ["text_classification/colab/roberta-base"],
                                            [], 2, 10, "text", "label")
    _compile_notebook(nb)
    nb = colab.object_detection_notebook("r", "t", "http://x", "d.zip", ["object_detection/colab/yolo11n.pt"], 5, 10)
    _compile_notebook(nb)


def test_invalid_selection_rejected(client, user):
    df = pd.DataFrame({"a": range(40), "b": ["x", "y"] * 20})
    p = upload(client, "sel", "predict b", "d.csv", csv_bytes(df))
    r = client.post(f"/api/projects/{p['id']}/runs", json={"backend": "cpu", "model_keys": ["regression/cpu/xgboost"]})
    assert r.status_code == 422
    r = client.post(f"/api/projects/{p['id']}/runs", json={"backend": "colab", "model_keys": ["classification/cpu/xgboost"]})
    assert r.status_code == 422
    r = client.put(f"/api/projects/{p['id']}/spec", json={"task": "regression", "target": "b", "metric": "r2"})
    assert r.status_code == 422
