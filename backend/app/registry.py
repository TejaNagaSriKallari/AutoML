"""The model registry: every task the platform understands and every model that can solve it.

Tabular models are auto-discovered from scikit-learn (every compatible estimator), plus
XGBoost / LightGBM / CatBoost. Deep-learning catalogs list well-tested defaults, and users can
add any timm (images) or Hugging Face (text) model by name, so the catalog is open-ended.
"""
from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable

from app.config import get_settings

# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "classification": {
        "label": "Tabular classification",
        "modality": "tabular",
        "description": "Predict a category (yes/no, churn, fraud, species...) from table columns.",
        "metrics": ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "log_loss"],
        "backends": ["cpu"],
    },
    "regression": {
        "label": "Tabular regression",
        "modality": "tabular",
        "description": "Predict a number (price, demand, score...) from table columns.",
        "metrics": ["r2", "rmse", "mae", "mape"],
        "backends": ["cpu"],
    },
    "clustering": {
        "label": "Clustering / segmentation",
        "modality": "tabular",
        "description": "Group similar rows when there is no label column (customer segments...).",
        "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"],
        "backends": ["cpu"],
    },
    "forecasting": {
        "label": "Time-series forecasting",
        "modality": "time_series",
        "description": "Predict future values of a series ordered by date (sales next month...).",
        "metrics": ["mae", "rmse", "mape", "smape"],
        "backends": ["cpu"],
    },
    "text_classification": {
        "label": "Text classification",
        "modality": "text",
        "description": "Assign a category to text (sentiment, topic, spam, intent...).",
        "metrics": ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "backends": ["cpu", "colab"],
    },
    "image_classification": {
        "label": "Image classification",
        "modality": "image",
        "description": "Assign a category to each image (defect/ok, disease type, animal...).",
        "metrics": ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"],
        "backends": ["cpu", "colab"],
    },
    "object_detection": {
        "label": "Object detection",
        "modality": "image",
        "description": "Find and box objects inside images (YOLO-format labels).",
        "metrics": ["map50_95", "map50", "precision", "recall"],
        "backends": ["colab"],
    },
}

if get_settings().lite_mode:  # no PyTorch on the server: deep-learning tasks train on Colab
    for _task in ("text_classification", "image_classification"):
        TASKS[_task]["backends"] = ["colab"]

METRICS: dict[str, dict] = {
    "accuracy": {"label": "Accuracy", "higher_is_better": True, "help": "Share of predictions that are correct."},
    "balanced_accuracy": {"label": "Balanced accuracy", "higher_is_better": True, "help": "Average recall per class; robust to imbalance."},
    "f1_macro": {"label": "F1 (macro)", "higher_is_better": True, "help": "Balance of precision and recall, averaged over classes equally."},
    "precision_macro": {"label": "Precision (macro)", "higher_is_better": True, "help": "When the model predicts a class, how often it is right."},
    "recall_macro": {"label": "Recall (macro)", "higher_is_better": True, "help": "How many of each class the model finds."},
    "roc_auc": {"label": "ROC AUC", "higher_is_better": True, "help": "How well the model ranks positives above negatives (1.0 = perfect)."},
    "log_loss": {"label": "Log loss", "higher_is_better": False, "help": "Penalises confident wrong probabilities."},
    "r2": {"label": "R²", "higher_is_better": True, "help": "Share of the target's variance explained (1.0 = perfect, 0 = predicting the mean)."},
    "rmse": {"label": "RMSE", "higher_is_better": False, "help": "Typical error size, in the target's units; punishes big misses."},
    "mae": {"label": "MAE", "higher_is_better": False, "help": "Average absolute error, in the target's units."},
    "mape": {"label": "MAPE %", "higher_is_better": False, "help": "Average error as a percentage of the true value."},
    "smape": {"label": "sMAPE %", "higher_is_better": False, "help": "Symmetric percentage error; stable near zero."},
    "silhouette": {"label": "Silhouette", "higher_is_better": True, "help": "How well separated the clusters are (-1 to 1)."},
    "calinski_harabasz": {"label": "Calinski-Harabasz", "higher_is_better": True, "help": "Ratio of between- to within-cluster spread."},
    "davies_bouldin": {"label": "Davies-Bouldin", "higher_is_better": False, "help": "Average similarity between clusters (lower = more distinct)."},
    "map50_95": {"label": "mAP50-95", "higher_is_better": True, "help": "Detection quality averaged over box-overlap thresholds."},
    "map50": {"label": "mAP50", "higher_is_better": True, "help": "Detection quality at 50% box overlap."},
    "precision": {"label": "Precision", "higher_is_better": True, "help": "Share of predicted boxes that are correct."},
    "recall": {"label": "Recall", "higher_is_better": True, "help": "Share of real objects that were found."},
}


@dataclass
class ModelSpec:
    key: str
    name: str
    task: str
    backend: str  # cpu | colab
    family: str
    description: str = ""
    role: str = "model"  # model | backbone | head  (image/text CPU = backbone x head combinations)
    source: str = "curated"
    default: bool = True
    speed: str = "medium"  # fast | medium | slow
    max_rows: int | None = None
    conditions: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    scaler: str = "standard"
    input_kind: tuple[str, ...] = ("dense",)  # for heads: which feature types they accept
    scale_target: bool = False  # regression: standardise y (kernel/SVM/GP/MLP need it)
    build: Callable[[], Any] | None = field(default=None, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("build", None)
        d["conditions"] = list(self.conditions)
        d["input_kind"] = list(self.input_kind)
        return d


# --------------------------------------------------------------------------------------
# Tabular: auto-discovered scikit-learn estimators + gradient boosting libraries
# --------------------------------------------------------------------------------------

_SK_EXCLUDE = {
    # meta-estimators that need a base estimator, multi-output wrappers, or are unsuitable
    "ClassifierChain", "MultiOutputClassifier", "OneVsOneClassifier", "OneVsRestClassifier",
    "OutputCodeClassifier", "StackingClassifier", "VotingClassifier", "SelfTrainingClassifier",
    "FixedThresholdClassifier", "TunedThresholdClassifierCV", "RadiusNeighborsClassifier", "CategoricalNB",
    "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",  # deprecated in scikit-learn 1.8
    "CCA", "PLSCanonical", "IsotonicRegression", "MultiOutputRegressor", "MultiTaskElasticNet",
    "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV", "RegressorChain", "StackingRegressor",
    "VotingRegressor", "TransformedTargetRegressor", "RadiusNeighborsRegressor",
    "FeatureAgglomeration", "SpectralBiclustering", "SpectralCoclustering",
}
_SK_MAX_ROWS = {
    "GaussianProcessClassifier": 2_000, "GaussianProcessRegressor": 2_000,
    "SVC": 20_000, "NuSVC": 20_000, "SVR": 20_000, "NuSVR": 20_000,
    "KernelRidge": 8_000, "LabelPropagation": 5_000, "LabelSpreading": 5_000,
    "TheilSenRegressor": 5_000, "QuantileRegressor": 5_000,
    "GradientBoostingClassifier": 150_000, "GradientBoostingRegressor": 150_000,
    "KNeighborsClassifier": 100_000, "KNeighborsRegressor": 100_000,
    # clustering: larger datasets are clustered on a sample instead of excluded
    "AffinityPropagation": 3_000, "MeanShift": 5_000, "SpectralClustering": 5_000,
    "AgglomerativeClustering": 10_000, "OPTICS": 10_000, "HDBSCAN": 50_000,
}
_SK_SLOW = set(_SK_MAX_ROWS) | {"MLPClassifier", "MLPRegressor", "LogisticRegressionCV", "AdaBoostClassifier",
                                "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor", "CalibratedClassifierCV"}
_SK_FAST = {"LogisticRegression", "RidgeClassifier", "RidgeClassifierCV", "GaussianNB", "BernoulliNB",
            "MultinomialNB", "ComplementNB", "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis",
            "NearestCentroid", "DecisionTreeClassifier", "ExtraTreeClassifier", "DummyClassifier", "Perceptron",
            "SGDClassifier", "PassiveAggressiveClassifier", "LinearRegression", "Ridge", "RidgeCV", "Lasso",
            "ElasticNet", "BayesianRidge", "DecisionTreeRegressor", "ExtraTreeRegressor", "DummyRegressor",
            "SGDRegressor", "Lars", "LassoLars", "LassoLarsIC", "OrthogonalMatchingPursuit", "KMeans",
            "MiniBatchKMeans", "Birch", "BisectingKMeans"}
_SK_OVERRIDES: dict[str, dict] = {
    "LogisticRegression": {"max_iter": 2000},
    "LogisticRegressionCV": {"max_iter": 2000, "cv": 3},
    "LinearSVC": {"max_iter": 5000},
    "LinearSVR": {"max_iter": 5000},
    "MLPClassifier": {"max_iter": 400, "early_stopping": True},
    "MLPRegressor": {"max_iter": 400, "early_stopping": True},
    "HuberRegressor": {"max_iter": 1000},
    "QuantileRegressor": {"solver": "highs", "alpha": 0.0},
    "KMeans": {"n_init": 10},
    "BisectingKMeans": {"n_init": 3},
    "HDBSCAN": {"min_cluster_size": 10},
    "QuadraticDiscriminantAnalysis": {"reg_param": 0.1},  # unregularised QDA fails on collinear features
}
_SK_SCALER = {"MultinomialNB": "minmax", "ComplementNB": "minmax"}
_SK_SCALE_TARGET = {"SVR", "NuSVR", "LinearSVR", "KernelRidge", "GaussianProcessRegressor", "MLPRegressor"}
_SK_CONDITIONS = {"GammaRegressor": ("positive_target",), "PoissonRegressor": ("nonnegative_target",)}


def _family(name: str) -> str:
    rules = [
        ("Dummy", "baseline"), ("HistGradientBoosting", "gradient_boosting"), ("GradientBoosting", "gradient_boosting"),
        ("AdaBoost", "boosting"), ("Forest", "ensemble"), ("ExtraTrees", "ensemble"), ("Bagging", "ensemble"),
        ("Tree", "tree"), ("SVC", "svm"), ("SVR", "svm"), ("SVM", "svm"), ("Neighbors", "neighbors"),
        ("Centroid", "neighbors"), ("NB", "naive_bayes"), ("DiscriminantAnalysis", "discriminant"),
        ("GaussianProcess", "gaussian_process"), ("MLP", "neural_network"), ("Label", "graph"),
        ("KernelRidge", "kernel"), ("PLS", "decomposition"), ("Calibrated", "calibrated"),
        ("Mixture", "probabilistic"), ("KMeans", "centroid"), ("Birch", "hierarchical"),
        ("Agglomerative", "hierarchical"), ("DBSCAN", "density"), ("OPTICS", "density"), ("MeanShift", "density"),
        ("Spectral", "graph"), ("Affinity", "graph"),
    ]
    for needle, fam in rules:
        if needle in name:
            return fam
    return "linear"


def _humanize(name: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return s.replace("C V", "CV").replace("S V C", "SVC").replace("S V R", "SVR").replace("N B", "NB")


def _instantiate(cls, name: str):
    params = inspect.signature(cls.__init__).parameters
    kwargs = dict(_SK_OVERRIDES.get(name, {}))
    if "random_state" in params:
        kwargs.setdefault("random_state", 42)
    if "n_jobs" in params:
        kwargs.setdefault("n_jobs", 1)
    return cls(**kwargs)


def _sklearn_specs() -> list[ModelSpec]:
    from sklearn.utils import all_estimators

    specs = []
    for type_filter, task in (("classifier", "classification"), ("regressor", "regression"), ("cluster", "clustering")):
        for name, cls in all_estimators(type_filter=type_filter):
            if name in _SK_EXCLUDE:
                continue
            try:
                _instantiate(cls, name)
            except TypeError:  # requires constructor arguments -> meta-estimator
                continue
            doc = (inspect.getdoc(cls) or "").split("\n")[0]
            specs.append(ModelSpec(
                key=f"{task}/cpu/sk.{name}",
                name=_humanize(name),
                task=task,
                backend="cpu",
                family=_family(name),
                description=doc,
                source="auto-discovered: scikit-learn",
                speed="fast" if name in _SK_FAST else "slow" if name in _SK_SLOW else "medium",
                max_rows=_SK_MAX_ROWS.get(name),
                conditions=_SK_CONDITIONS.get(name, ()),
                scaler=_SK_SCALER.get(name, "standard"),
                scale_target=name in _SK_SCALE_TARGET,
                params={"class": f"{cls.__module__}.{name}", **_SK_OVERRIDES.get(name, {})},
                build=(lambda c=cls, n=name: _instantiate(c, n)),
            ))
    # Gaussian mixtures live in sklearn.mixture, not sklearn.cluster
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

    for cls in (GaussianMixture, BayesianGaussianMixture):
        specs.append(ModelSpec(
            key=f"clustering/cpu/sk.{cls.__name__}", name=_humanize(cls.__name__), task="clustering",
            backend="cpu", family="probabilistic", description=(inspect.getdoc(cls) or "").split("\n")[0],
            source="auto-discovered: scikit-learn", speed="medium", params={"class": f"sklearn.mixture.{cls.__name__}"},
            build=(lambda c=cls: c(random_state=42)),
        ))
    return specs


def _optional(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _boosting_specs() -> list[ModelSpec]:
    specs = []
    if _optional("xgboost"):
        import xgboost as xgb

        common = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", n_jobs=2, random_state=42)
        specs += [
            ModelSpec("classification/cpu/xgboost", "XGBoost", "classification", "cpu", "gradient_boosting",
                      "Regularised gradient-boosted trees; a top performer on tabular data.", source="xgboost",
                      params=common, build=lambda: xgb.XGBClassifier(**common, eval_metric="logloss")),
            ModelSpec("regression/cpu/xgboost", "XGBoost", "regression", "cpu", "gradient_boosting",
                      "Regularised gradient-boosted trees; a top performer on tabular data.", source="xgboost",
                      params=common, build=lambda: xgb.XGBRegressor(**common)),
        ]
    if _optional("lightgbm"):
        import lightgbm as lgb

        common = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)
        specs += [
            ModelSpec("classification/cpu/lightgbm", "LightGBM", "classification", "cpu", "gradient_boosting",
                      "Fast histogram gradient boosting from Microsoft.", source="lightgbm", speed="fast",
                      params=common, build=lambda: lgb.LGBMClassifier(**common)),
            ModelSpec("regression/cpu/lightgbm", "LightGBM", "regression", "cpu", "gradient_boosting",
                      "Fast histogram gradient boosting from Microsoft.", source="lightgbm", speed="fast",
                      params=common, build=lambda: lgb.LGBMRegressor(**common)),
        ]
    if _optional("catboost"):
        import catboost as cb

        common = dict(iterations=500, learning_rate=0.08, depth=6, verbose=0, thread_count=2, random_seed=42,
                      allow_writing_files=False)
        specs += [
            ModelSpec("classification/cpu/catboost", "CatBoost", "classification", "cpu", "gradient_boosting",
                      "Gradient boosting with ordered boosting; strong defaults.", source="catboost",
                      params=common, build=lambda: cb.CatBoostClassifier(**common)),
            ModelSpec("regression/cpu/catboost", "CatBoost", "regression", "cpu", "gradient_boosting",
                      "Gradient boosting with ordered boosting; strong defaults.", source="catboost",
                      params=common, build=lambda: cb.CatBoostRegressor(**common)),
        ]
    return specs


# --------------------------------------------------------------------------------------
# Time series forecasting
# --------------------------------------------------------------------------------------

def _forecast_specs() -> list[ModelSpec]:
    t, b = "forecasting", "cpu"
    rows = [
        ("naive", "Naive (last value)", "baseline", "Repeats the last observed value.", "fast", ()),
        ("seasonal_naive", "Seasonal naive", "baseline", "Repeats the value from one season ago.", "fast", ("seasonal",)),
        ("drift", "Drift", "baseline", "Extends the line between first and last observation.", "fast", ()),
        ("moving_average", "Moving average", "baseline", "Average of the most recent window.", "fast", ()),
        ("ses", "Simple exponential smoothing", "statistical", "Weighted average favouring recent values.", "fast", ()),
        ("holt", "Holt (trend)", "statistical", "Exponential smoothing with a trend.", "fast", ()),
        ("holt_winters", "Holt-Winters (trend + season)", "statistical", "Exponential smoothing with trend and seasonality.", "fast", ("seasonal",)),
        ("theta", "Theta", "statistical", "Theta method; strong in forecasting competitions.", "fast", ()),
        ("arima", "ARIMA (auto order)", "statistical", "Autoregressive integrated moving average; order picked by AIC.", "medium", ("min_points_30",)),
        ("lag.linear", "Linear regression on lags", "ml_lag", "Linear model on past values + calendar features.", "fast", ()),
        ("lag.ridge", "Ridge on lags", "ml_lag", "Regularised linear model on lag features.", "fast", ()),
        ("lag.random_forest", "Random forest on lags", "ml_lag", "Tree ensemble on lag features.", "medium", ()),
        ("lag.extra_trees", "Extra trees on lags", "ml_lag", "Randomised tree ensemble on lag features.", "medium", ()),
        ("lag.hist_gb", "Hist gradient boosting on lags", "ml_lag", "Gradient boosting on lag features.", "medium", ()),
        ("lag.lightgbm", "LightGBM on lags", "ml_lag", "LightGBM on lag + calendar features.", "medium", ()),
        ("lag.xgboost", "XGBoost on lags", "ml_lag", "XGBoost on lag + calendar features.", "medium", ()),
        ("lag.knn", "KNN on lags", "ml_lag", "Finds similar past windows.", "fast", ()),
        ("lag.svr", "SVR on lags", "ml_lag", "Support vector regression on lag features.", "medium", ()),
    ]
    return [ModelSpec(f"{t}/{b}/{k}", n, t, b, fam, d, speed=s, conditions=c) for k, n, fam, d, s, c in rows]


# --------------------------------------------------------------------------------------
# Text classification
# --------------------------------------------------------------------------------------

TEXT_ENCODERS = {
    "tfidf_word": ("TF-IDF words (1-2 grams)", "sparse", True, "Classic bag-of-words; fast, strong baseline."),
    "tfidf_char": ("TF-IDF characters (2-5 grams)", "sparse", True, "Robust to typos and morphology."),
    "sentence-transformers/all-MiniLM-L6-v2": ("MiniLM-L6 sentence embeddings", "dense", True, "Small pretrained transformer encoder (English)."),
    "BAAI/bge-small-en-v1.5": ("BGE-small embeddings", "dense", True, "High-quality small English embedding model."),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("Multilingual MiniLM embeddings", "dense", False, "50+ languages."),
    "sentence-transformers/all-mpnet-base-v2": ("MPNet-base embeddings", "dense", False, "Larger, more accurate, slower on CPU."),
}
HEADS = {
    # key: (name, family, accepts, default_for_text, default_for_image, description)
    "logreg": ("Logistic Regression", "linear", ("sparse", "dense"), True, True, "Linear probabilistic classifier."),
    "linear_svc": ("Linear SVM", "svm", ("sparse", "dense"), True, True, "Maximum-margin linear classifier."),
    "svc_rbf": ("RBF SVM", "svm", ("dense",), False, True, "Non-linear kernel SVM."),
    "ridge": ("Ridge Classifier", "linear", ("sparse", "dense"), True, True, "Least-squares linear classifier."),
    "sgd": ("SGD Classifier", "linear", ("sparse", "dense"), True, False, "Linear model trained by stochastic gradient descent."),
    "complement_nb": ("Complement Naive Bayes", "naive_bayes", ("sparse",), True, False, "Naive Bayes tuned for imbalanced text."),
    "multinomial_nb": ("Multinomial Naive Bayes", "naive_bayes", ("sparse",), True, False, "Classic text Naive Bayes."),
    "knn": ("k-Nearest Neighbours", "neighbors", ("dense",), True, True, "Votes among the most similar examples."),
    "random_forest": ("Random Forest", "ensemble", ("sparse", "dense"), False, True, "Ensemble of decision trees."),
    "mlp": ("Neural network (MLP)", "neural_network", ("dense",), True, True, "Small fully connected network on the features."),
    "lightgbm": ("LightGBM", "gradient_boosting", ("dense",), False, False, "Gradient boosting on the features."),
}


def build_head(key: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.naive_bayes import ComplementNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC

    if key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, random_state=42, verbose=-1)
    return {
        "logreg": lambda: LogisticRegression(max_iter=3000, C=1.0),
        "linear_svc": lambda: LinearSVC(max_iter=5000, C=0.5),
        "svc_rbf": lambda: SVC(C=5.0, gamma="scale", probability=False, random_state=42),
        "ridge": lambda: RidgeClassifier(alpha=1.0),
        "sgd": lambda: SGDClassifier(loss="modified_huber", random_state=42),
        "complement_nb": lambda: ComplementNB(),
        "multinomial_nb": lambda: MultinomialNB(alpha=0.1),
        "knn": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=42),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True, random_state=42),
    }[key]()


def _text_specs() -> list[ModelSpec]:
    t = "text_classification"
    specs = []
    for key, (name, kind, default, desc) in TEXT_ENCODERS.items():
        specs.append(ModelSpec(f"{t}/cpu/enc:{key}", name, t, "cpu", "tfidf" if kind == "sparse" else "transformer_embedding",
                               desc, role="backbone", default=default, speed="fast" if kind == "sparse" else "medium",
                               input_kind=(kind,), params={"encoder": key}))
    for key, (name, fam, accepts, d_text, _, desc) in HEADS.items():
        specs.append(ModelSpec(f"{t}/cpu/head:{key}", name, t, "cpu", fam, desc, role="head", default=d_text,
                               input_kind=accepts))
    colab = [
        ("distilbert-base-uncased", "DistilBERT", True, "Fast distilled BERT."),
        ("bert-base-uncased", "BERT-base", True, "The original BERT encoder."),
        ("roberta-base", "RoBERTa-base", True, "Robustly optimised BERT; strong default."),
        ("microsoft/deberta-v3-small", "DeBERTa-v3-small", True, "Disentangled attention; excellent accuracy per size."),
        ("google/electra-small-discriminator", "ELECTRA-small", False, "Very efficient small encoder."),
        ("albert-base-v2", "ALBERT-base", False, "Parameter-shared BERT; small memory."),
        ("xlm-roberta-base", "XLM-RoBERTa-base", False, "Multilingual (100 languages)."),
    ]
    for hf, name, default, desc in colab:
        specs.append(ModelSpec(f"{t}/colab/{hf}", name, t, "colab", "transformer_finetune", desc + " Fine-tuned end to end on GPU.",
                               default=default, speed="slow", params={"hf_model": hf}))
    return specs


# --------------------------------------------------------------------------------------
# Image classification + object detection
# --------------------------------------------------------------------------------------

IMAGE_BACKBONES = [
    # timm name, display name, default on CPU, description
    ("mobilenetv3_large_100", "MobileNetV3-Large", True, "Very fast mobile CNN."),
    ("efficientnet_b0", "EfficientNet-B0", True, "Efficient CNN; great accuracy per FLOP."),
    ("resnet50", "ResNet-50", True, "Classic residual CNN; strong general features."),
    ("convnext_tiny", "ConvNeXt-Tiny", True, "Modern CNN with transformer-era design."),
    ("vit_small_patch16_224", "ViT-Small/16", True, "Vision Transformer."),
    ("regnety_016", "RegNetY-1.6GF", False, "Efficient design-space CNN."),
    ("densenet121", "DenseNet-121", False, "Densely connected CNN."),
    ("swin_tiny_patch4_window7_224", "Swin-Tiny", False, "Shifted-window vision transformer."),
    ("vit_base_patch32_clip_224.openai", "CLIP ViT-B/32", False, "CLIP image encoder; very general features."),
    ("efficientnet_b3", "EfficientNet-B3", False, "Larger EfficientNet; better accuracy, slower."),
]
COLAB_IMAGE_DEFAULTS = {"resnet50", "efficientnet_b0", "convnext_tiny", "vit_small_patch16_224",
                        "mobilenetv3_large_100", "swin_tiny_patch4_window7_224"}
DETECTORS = [
    ("yolov8n.pt", "YOLOv8-nano", True, "Smallest, fastest YOLOv8."),
    ("yolov8s.pt", "YOLOv8-small", True, "Balanced YOLOv8."),
    ("yolo11n.pt", "YOLO11-nano", True, "Latest-generation nano YOLO."),
    ("yolo11s.pt", "YOLO11-small", True, "Latest-generation small YOLO."),
    ("yolo11m.pt", "YOLO11-medium", False, "More accurate, needs more GPU time."),
    ("rtdetr-l.pt", "RT-DETR-L", False, "Real-time detection transformer."),
]


def _image_specs() -> list[ModelSpec]:
    t = "image_classification"
    specs = []
    for name, display, default, desc in IMAGE_BACKBONES:
        specs.append(ModelSpec(f"{t}/cpu/backbone:{name}", display, t, "cpu", "pretrained_cnn" if "vit" not in name and "swin" not in name else "pretrained_transformer",
                               desc + " Used frozen as a feature extractor.", role="backbone", default=default,
                               speed="medium", params={"timm_model": name}))
        specs.append(ModelSpec(f"{t}/colab/{name}", display, t, "colab", "finetune",
                               desc + " Fine-tuned end to end on GPU.", default=name in COLAB_IMAGE_DEFAULTS,
                               speed="slow", params={"timm_model": name}))
    for key, (name, fam, accepts, _, d_img, desc) in HEADS.items():
        if "dense" in accepts:
            specs.append(ModelSpec(f"{t}/cpu/head:{key}", name, t, "cpu", fam, desc, role="head", default=d_img))
    for weights, display, default, desc in DETECTORS:
        specs.append(ModelSpec(f"object_detection/colab/{weights}", display, "object_detection", "colab", "detector",
                               desc, default=default, speed="slow", params={"weights": weights}))
    return specs


# --------------------------------------------------------------------------------------
# Lookup + eligibility
# --------------------------------------------------------------------------------------

@lru_cache
def all_models() -> dict[str, ModelSpec]:
    specs = _sklearn_specs() + _boosting_specs() + _forecast_specs() + _text_specs() + _image_specs()
    return {s.key: s for s in specs}


def get_model(key: str) -> ModelSpec:
    try:
        return all_models()[key]
    except KeyError:
        raise KeyError(f"Unknown model '{key}'") from None


def catalog() -> dict:
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    for s in all_models().values():
        by_task[s.task].append(s.public())
    return {"tasks": TASKS, "metrics": METRICS, "models": by_task,
            "custom_models": {"image_classification": "Any timm model name (timm.list_models(pretrained=True))",
                              "text_classification": "Any Hugging Face model id for sequence classification"}}


def candidates(spec: dict, profile: dict) -> list[dict]:
    """Every registered model for the task, each marked eligible or not (with the reason)."""
    task = spec["task"]
    n_rows = _n_rows(spec, profile)
    target_stats = _target_stats(spec, profile)
    lite = get_settings().lite_mode
    out = []
    for s in all_models().values():
        if s.task != task:
            continue
        eligible, reason, note = True, None, None
        if s.max_rows and n_rows and n_rows > s.max_rows:
            if task == "clustering":
                note = f"Fitted on a {s.max_rows:,}-row sample (dataset has {n_rows:,} rows)."
            else:
                eligible, reason = False, f"Too slow for {n_rows:,} rows (limit {s.max_rows:,})."
        for cond in s.conditions:
            if cond == "positive_target" and target_stats and target_stats.get("min", 1) is not None and target_stats.get("min", 1) <= 0:
                eligible, reason = False, "Needs a strictly positive target."
            if cond == "nonnegative_target" and target_stats and target_stats.get("min", 0) is not None and target_stats.get("min", 0) < 0:
                eligible, reason = False, "Needs a non-negative target."
            if cond == "seasonal" and not spec.get("season_length"):
                eligible, reason = False, "No seasonality detected/declared."
            if cond == "min_points_30" and n_rows and n_rows < 30:
                eligible, reason = False, "Needs at least 30 time points."
        d = s.public()
        default = s.default and not (lite and s.backend == "cpu" and s.speed == "slow")
        d.update(eligible=eligible, reason=reason, note=note, selected=eligible and default)
        out.append(d)
    return out


def _n_rows(spec: dict, profile: dict) -> int | None:
    if "n_rows" in profile:
        return int(profile["n_rows"])
    if "n_images" in profile:
        return int(profile["n_images"])
    return None


def _target_stats(spec: dict, profile: dict) -> dict | None:
    for c in profile.get("columns", []):
        if c["name"] == spec.get("target"):
            return c.get("stats")
    return None
