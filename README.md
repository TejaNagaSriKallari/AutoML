# AutoML Studio

Users describe their problem in plain language and upload a dataset. The platform works out the ML task, trains **every suitable model**, and shows a leaderboard with the best model, a plain-English summary, downloads, and predictions on new data.

| Task | Data | Where it trains | Models |
|---|---|---|---|
| Classification | table | CPU | every compatible scikit-learn classifier (auto-discovered) + XGBoost, LightGBM, CatBoost |
| Regression | table | CPU | every compatible scikit-learn regressor (auto-discovered) + XGBoost, LightGBM, CatBoost |
| Clustering | table | CPU | KMeans family, DBSCAN/HDBSCAN/OPTICS, hierarchical, spectral, Gaussian mixtures (auto-k) |
| Forecasting | table with a date column | CPU | naive baselines, ETS/Holt-Winters, Theta, auto-ARIMA, 10 ML models on lag features |
| Text classification | table with a text column | CPU **(Option 1)** / Colab GPU **(Option 2)** | TF-IDF & sentence embeddings × classifiers / fine-tuned BERT, RoBERTa, DeBERTa… + any HF model |
| Image classification | ZIP of images | CPU **(Option 1)** / Colab GPU **(Option 2)** | frozen timm backbones × classifiers / fine-tuned ResNet, EfficientNet, ConvNeXt, ViT, Swin… + any timm model |
| Object detection | ZIP in YOLO format | Colab GPU | YOLOv8, YOLO11, RT-DETR |

## How it works

```
upload ─► profiling (column types, classes, frequency) ─► task spec (Claude, or rules) ─► user confirms
      ─► registry lists every model for the task, eligible or not (with the reason)
      ─► CPU: job queue → worker → process pool (time limit per model, crash isolation)
         Colab: generated notebook → downloads data with a run token → posts each result back
      ─► same split for all models, CV ranking, test metrics, leakage check, feature importance, summary
```

* **Option 1 (CPU, no GPU needed):** pretrained networks are used frozen to turn each image or text into a feature vector once. Many classifiers are then trained on those vectors.
* **Option 2 (Colab, free GPU):** the app generates a notebook containing the run ID, the selected models and a private run token. The user uploads it to Colab and clicks *Run all*. Results stream back live. If Colab can't reach the server, the notebook uses manual mode: upload the data in Colab, then import `results.json` back into the app. Both options use the **same dataset-discovery and split code** (`app/shared_data.py` is embedded into the notebook), so CPU and GPU results compare on the same test set.

## Run locally (Windows, this machine)

The virtualenv, model-weight cache and user data live on `D:\ML-RAG-runtime` because C: is nearly full.

```powershell
# 1. API + built-in training worker  (http://localhost:8000/docs)
cd backend
D:\ML-RAG-runtime\venv\Scripts\python -m uvicorn app.main:app --port 8000

# 2. Web app  (http://localhost:3000)
cd frontend
npm run dev
```

Optional: put `ANTHROPIC_API_KEY=...` in `backend/.env` to let Claude interpret requirements (otherwise rule-based detection is used).

### Colab live mode during development
Colab can't reach `localhost`. Expose the API through a tunnel and set `PUBLIC_API_URL`:

```powershell
cloudflared tunnel --url http://localhost:8000      # prints https://xxxx.trycloudflare.com
# backend/.env:  PUBLIC_API_URL=https://xxxx.trycloudflare.com   (restart the API)
```
Without a tunnel, notebooks automatically use manual mode.

### Fresh setup on another machine
```bash
cd backend
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                                   # set SECRET_KEY
cd ../frontend && npm install
```

## Production
```bash
SECRET_KEY=$(openssl rand -hex 32) docker compose up --build
docker compose up --scale worker=3     # more concurrent training runs
```
Postgres stores state, the API never trains, and each `worker` claims queued runs atomically. Behind HTTPS, set `COOKIE_SECURE=true`, `FRONTEND_ORIGIN`, and `PUBLIC_API_URL` to your public URLs.

## Tests
```powershell
cd backend
$env:AUTOML_TEST_TMP="D:\ML-RAG-runtime\cache\tests"
D:\ML-RAG-runtime\venv\Scripts\python -m pytest tests -q
```
End-to-end through the API: auth and isolation, every default tabular model on two datasets, leakage detection, tuning, clustering, forecasting, text (TF-IDF + MiniLM), images (MobileNetV3 features), the Colab token flow, manual import, and notebook syntax.

## Layout
```
backend/
  app/registry.py        every task + model (auto-discovery, eligibility rules)
  app/profiling.py       column types, image layouts, YOLO detection, frequency
  app/understanding.py   requirement -> TaskSpec (Claude structured output + rules + validation)
  app/training/          tabular, forecasting, text, image trainers; evaluation; tuning; process pool
  app/colab.py           notebook generators (image / text / detection)
  app/shared_data.py     discovery + split shared by CPU and Colab
  app/worker.py          DB-backed job queue worker with heartbeats
  automl_runtime.py      transformers inside saved models (shipped with every download)
frontend/src/app/        Next.js pages: dashboard, new project, project, run, model catalog
```

## Known limits
* Forecasting handles one series. Multiple rows per date are summed (per-store/per-product forecasting isn't supported yet).
* Colab-trained model weights stay in the user's Colab session (`/content/models`); only metrics come back.
* Free Colab sessions can disconnect. Finished models are kept; re-running the notebook continues.
* Uploads go through the API. For multi-GB datasets in production, switch to presigned S3 uploads.
