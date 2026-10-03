# AutoML Studio

**Describe your problem in plain English, upload a dataset, and get a leaderboard of trained models.**
AutoML Studio works out the machine-learning task, trains every model that fits it, ranks them on the same held-out test set, and explains the results. You can download the best model or use it to predict on new data.

### 🚀 [Try the live demo → automl-studio-web.onrender.com](https://automl-studio-web.onrender.com)

> Hosted on a free plan: if nobody has used it recently, the first load takes about a minute while the server wakes up.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

| | Link |
|---|---|
| Web app | https://automl-studio-web.onrender.com |
| API health check | https://automl-api-1ajs.onrender.com/api/health |
| Interactive API docs (Swagger) | https://automl-api-1ajs.onrender.com/docs |

---

## What it does

1. **Upload** a CSV, Excel, Parquet or JSON table, or a ZIP of images.
2. **Describe the goal**, e.g. *"predict which customers will churn"* or *"forecast sales for the next 30 days"*.
3. **Review the plan.** The app profiles every column and proposes the task, the column to predict, which columns to drop and the scoring metric, with a plain-language reason for each. You can change any of it.
4. **Train.** Every suitable model is trained on the same train/test split, with a time limit per model.
5. **Compare.** You get a leaderboard with cross-validation scores, test metrics, a confusion matrix, feature importance, data-leakage warnings and a written summary.
6. **Use the model.** Download it as a ZIP (model file plus a ready-to-run Python example), or upload new data and get a `predictions.csv` back.

## Supported tasks

| Task | Input | Where it trains | Models |
|---|---|---|---|
| Classification | table | Server CPU | every compatible scikit-learn classifier (auto-discovered) + XGBoost, LightGBM, CatBoost |
| Regression | table | Server CPU | every compatible scikit-learn regressor (auto-discovered) + XGBoost, LightGBM, CatBoost |
| Clustering | table | Server CPU | KMeans family, DBSCAN / HDBSCAN / OPTICS, hierarchical, spectral, Gaussian mixtures (automatic number of clusters) |
| Forecasting | table with a date column | Server CPU | naive baselines, ETS / Holt-Winters, Theta, auto-ARIMA, 10 ML models on lag features |
| Text classification | table with a text column | Google Colab GPU* | fine-tuned BERT, RoBERTa, DeBERTa… or any Hugging Face model |
| Image classification | ZIP of images in class folders | Google Colab GPU* | fine-tuned ResNet, EfficientNet, ConvNeXt, ViT, Swin… or any timm model |
| Object detection | ZIP in YOLO format | Google Colab GPU | YOLOv8, YOLO11, RT-DETR |

\* On a full server (not lite mode, see [Deployment](#deployment)) text and image tasks can also train on CPU, using frozen pretrained embeddings (TF-IDF, sentence-transformers, timm backbones) with classical classifiers on top.

### Training on Google Colab's free GPU
For deep-learning tasks the app generates a Jupyter notebook that already contains your run ID, the selected models and a private access token. Open it in [Google Colab](https://colab.research.google.com), choose a GPU runtime, and click **Run all**. The notebook downloads your dataset, trains, and sends each result back to the app live. If Colab can't reach the server, it switches to manual mode: you upload the data in Colab and import `results.json` back into the app.

CPU and Colab runs use the same data-loading and split code (`backend/app/shared_data.py` is embedded in the notebook), so their results can be compared directly.

## How it works

```
upload ─► profiling (column types, classes, date frequency, image/YOLO layout)
       ─► task detection (rules, or optionally an LLM) ─► user reviews and confirms
       ─► model registry: every model for the task, marked eligible or not (with the reason)
       ─► job queue ─► worker ─► process pool (time limit per model, crash isolation)
       ─► same split for every model ─► CV ranking, test metrics, leakage check,
          feature importance, plain-English summary ─► leaderboard, download, predict
```

### How the task is detected
Without any AI service, deterministic rules decide ([`backend/app/understanding.py`](backend/app/understanding.py)):

- **Column types** come from the values: numbers, a few repeating categories, dates, free text, ID-like columns (unique per row) and constant columns. ID-like and constant columns are dropped.
- **Target column:** a column named in your description, otherwise a common target name (`label`, `price`, `churn`, …), otherwise the last column, with a warning asking you to confirm.
- **Task:** forecasting if you mention forecasting and there's a date column; classification if the target has a few distinct values; text classification if there's a long text column; otherwise regression. Words like *segment* or *group* select clustering.
- **Metric:** F1 for imbalanced classes, accuracy for balanced classes, R² for regression, MAE for forecasts.

**Optional:** set `ANTHROPIC_API_KEY` and Claude reads your description plus the data profile to propose the setup, which helps with vague descriptions or unclear column names. Every proposal is still validated against the real columns, and the rules take over if the call fails.

### How the best model is chosen
The app doesn't guess. It trains every eligible model on the same split, ranks them by cross-validation score on the training data, and reports test-set metrics on data no model has seen. The top of the leaderboard is the best measured model.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS 4, Recharts |
| Backend API | FastAPI, SQLAlchemy 2, Pydantic 2, JWT sessions in httpOnly cookies |
| Machine learning | scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, Optuna (tuning) |
| Deep learning (optional) | PyTorch, timm, sentence-transformers |
| Job processing | database-backed job queue, worker with heartbeats, Pebble process pool |
| Database | PostgreSQL (production), SQLite (local development) |
| Hosting | Render (API + web), Neon (Postgres), Google Colab (GPU training) |

## Run it locally

**Requirements:** Python 3.12+, Node.js 20+.

```bash
git clone https://github.com/TejaNagaSriKallari/AutoML.git
cd AutoML
```

**1. Backend** (API + built-in training worker) at http://localhost:8000/docs
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# Full install (CPU deep learning included):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# …or the lighter install without PyTorch (text/image tasks then go to Colab):
#   pip install -r requirements-lite.txt   and set LITE_MODE=true in .env

cp .env.example .env          # then set SECRET_KEY to a long random string
python -m uvicorn app.main:app --port 8000
```

**2. Frontend** at http://localhost:3000
```bash
cd frontend
npm install
npm run dev
```

With no database configured, the backend uses a SQLite file inside `DATA_DIR`.

### Colab live mode during development
Colab can't reach `localhost`. Expose the API through a tunnel and set `PUBLIC_API_URL`:
```bash
cloudflared tunnel --url http://localhost:8000      # prints https://xxxx.trycloudflare.com
# backend/.env:  PUBLIC_API_URL=https://xxxx.trycloudflare.com   (then restart the API)
```
Without a tunnel, notebooks use manual mode automatically.

## Configuration

All backend settings are environment variables (or entries in `backend/.env`). See [`backend/.env.example`](backend/.env.example) for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | — | Signs login sessions. **Required in production.** |
| `DATABASE_URL` | SQLite in `DATA_DIR` | Postgres connection string (`postgresql://…` accepted) |
| `DATA_DIR` | `./data` | Uploaded datasets and trained models |
| `ANTHROPIC_API_KEY` | empty | Optional: LLM-based task detection and summaries |
| `LITE_MODE` | `false` | No PyTorch on the server: text/image tasks train on Colab only |
| `MAX_TABULAR_ROWS` | `200000` | Larger tables are randomly sampled down to this size |
| `MAX_UPLOAD_MB` | `2048` | Upload size limit |
| `MAX_ACTIVE_RUNS_PER_USER` | `2` | Concurrent training runs per user |
| `DEFAULT_TIME_LIMIT_PER_MODEL` | `300` | Seconds before a model is stopped |
| `WORKER_PROCESSES` | `0` (auto) | Parallel training processes |
| `RUN_WORKER_IN_API` | `true` | Run the job worker inside the API process |
| `PUBLIC_API_URL` | `http://localhost:8000` | API address written into Colab notebooks |
| `COOKIE_SECURE` | `false` | Set `true` behind HTTPS |

Frontend: `NEXT_PUBLIC_API_URL` (API address, or `/` to use the built-in proxy) and `BACKEND_URL` (where the proxy forwards `/api/*`).

## Deployment

### Free: Render + Neon (how the live demo runs)
[`render.yaml`](render.yaml) is a Render Blueprint that creates both services from this repo in one step. Step-by-step instructions are in **[DEPLOY.md](DEPLOY.md)**.

```
Browser ──► automl-studio-web (Next.js, Render)
               │  proxies /api/* (same site, so the login cookie works)
               ▼
            automl-api (FastAPI, Render, lite mode) ──► Neon Postgres
               │
               └──► generated notebooks ──► Google Colab GPU (text / image / detection)
```

Lite mode keeps the API within Render's free 512 MB: table tasks train on the server (up to 10,000 rows) and deep-learning tasks train on Colab.

### Full server: Docker Compose
```bash
SECRET_KEY=$(openssl rand -hex 32) docker compose up --build
docker compose up --scale worker=3     # more concurrent training runs
```
This runs Postgres, the API (which doesn't train), separate training workers that claim queued runs atomically, and the web app. Behind HTTPS, set `COOKIE_SECURE=true`, `FRONTEND_ORIGIN` and `PUBLIC_API_URL` to your public URLs.

## Tests

```bash
cd backend
python -m pytest tests -q
```
The end-to-end suite runs through the API. It covers authentication and per-user isolation, every default tabular model on two datasets, leakage detection, tuning, clustering, forecasting, text (TF-IDF + MiniLM), images (MobileNetV3 features), the Colab token flow, manual result import and notebook syntax.

## Project structure

```
backend/
  app/main.py            FastAPI app, CORS, routers
  app/registry.py        every task and model (auto-discovery, eligibility rules)
  app/profiling.py       column types, image layouts, YOLO detection, date frequency
  app/understanding.py   requirement → task spec (rules, optional LLM, validation)
  app/training/          tabular, forecasting, text and image trainers; evaluation; tuning; process pool
  app/colab.py           Colab notebook generators (image / text / detection)
  app/shared_data.py     dataset discovery and split shared by CPU and Colab
  app/worker.py          database-backed job queue worker with heartbeats
  automl_runtime.py      helpers bundled with every downloaded model
  requirements.txt       full install (with PyTorch)
  requirements-lite.txt  lite install for small servers
frontend/
  src/app/               pages: dashboard, new project, project, run, model catalog
  src/components/        leaderboard, charts, model picker, prediction panel, Colab panel
  next.config.ts         /api proxy for production
render.yaml              Render Blueprint (API + web)
docker-compose.yml       self-hosted production stack
DEPLOY.md                free deployment guide
```

## Known limits

- Forecasting handles one series. Multiple rows per date are summed; per-store or per-product forecasting isn't supported yet.
- Colab-trained model weights stay in your Colab session (`/content/models`). Only the metrics come back to the app.
- Free Colab sessions can disconnect. Finished models are kept, and re-running the notebook continues where it stopped.
- **Live demo (free hosting):** uploaded files and trained models are cleared whenever the server restarts. Your account, projects and results are kept. Download your model after training.
- Uploads go through the API. For multi-GB datasets in production, switch to direct-to-storage (presigned S3) uploads.

## Author

**Teja Naga Sri Kallari**, [@TejaNagaSriKallari](https://github.com/TejaNagaSriKallari)

Issues and suggestions are welcome: [open an issue](https://github.com/TejaNagaSriKallari/AutoML/issues).
