"""Turns the user's free-text requirement + the dataset profile into a validated TaskSpec.

Claude proposes the spec when an API key is configured; deterministic rules are the fallback
and also repair any field Claude gets wrong (e.g. a column name that does not exist).
"""
import json
import logging
import re

import anthropic

from app.config import get_settings
from app.registry import METRICS, TASKS
from app.schemas import TaskSpec

log = logging.getLogger(__name__)

_FORECAST = re.compile(r"\b(forecast\w*|predict(ing)? (the )?future|next \d* ?(hours?|days?|weeks?|months?|quarters?|years?)|time[- ]series|demand planning|projection)\b", re.I)
_CLUSTER = re.compile(r"\b(cluster\w*|segment\w*|group (similar|customers|users)|unsupervised|personas?|find groups)\b", re.I)
_TEXT = re.compile(r"\b(text|review|tweet|message|e-?mail|comment|sentiment|document|description|spam|intent|ticket|headline|nlp)\w*\b", re.I)
_HORIZON = re.compile(r"next (\d+) ?(hours?|days?|weeks?|months?|quarters?|years?)", re.I)
_COMMON_TARGETS = ["target", "label", "labels", "class", "y", "outcome", "result", "churn", "price", "sales",
                   "survived", "default", "fraud", "is_fraud", "diagnosis", "species", "quality", "rating", "category"]


def infer_spec(requirement: str, dataset_kind: str, profile: dict) -> TaskSpec:
    rules = rule_based_spec(requirement, dataset_kind, profile)
    settings = get_settings()
    if not settings.anthropic_api_key or dataset_kind != "table":
        return rules
    try:
        proposed = _ask_claude(requirement, profile, rules)
    except Exception as e:  # never block a project on the LLM; rules are a complete answer
        log.warning("Claude spec inference failed, using rules: %s", e)
        rules.warnings.append("AI requirement analysis was unavailable; task was detected with rules.")
        return rules
    return validate_spec(proposed, dataset_kind, profile, fallback=rules)


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------

def _usable(col: dict) -> bool:
    return col["kind"] not in {"id", "constant", "empty"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _mentioned_columns(requirement: str, cols: list[dict]) -> list[str]:
    req = f" {_norm(requirement)} "
    hits = []
    for c in cols:
        n = _norm(c["name"])
        if n and f" {n} " in req:
            hits.append((req.index(f" {n} "), -len(n), c["name"]))
    return [name for _, _, name in sorted(hits)]


def _is_classification_target(col: dict, n_rows: int) -> bool:
    if col["kind"] == "categorical":
        return True
    if col["kind"] == "numeric":
        nu = col["n_unique"]
        return col.get("integer_valued", False) and (nu <= 2 or (nu <= 20 and nu / max(n_rows, 1) < 0.05))
    return False


def _imbalanced(col: dict | None) -> bool:
    tops = (col or {}).get("top_values") or []
    if len(tops) < 2:
        return False
    counts = [t["count"] for t in tops]
    return min(counts) / max(counts) < 0.3


def rule_based_spec(requirement: str, dataset_kind: str, profile: dict) -> TaskSpec:
    if dataset_kind == "image_folder":
        counts = list(profile.get("classes", {}).values())
        imbalanced = bool(counts) and min(counts) / max(counts) < 0.3
        return TaskSpec(task="image_classification", target="label", metric="f1_macro" if imbalanced else "accuracy",
                        rationale=f"The upload is a folder of images in {profile.get('n_classes')} labelled classes.",
                        warnings=list(profile.get("notes", [])), source="rules", confidence=0.95)
    if dataset_kind == "yolo_detection":
        return TaskSpec(task="object_detection", metric="map50_95",
                        rationale="The upload contains images with YOLO bounding-box label files.",
                        warnings=list(profile.get("notes", [])), source="rules", confidence=0.95)

    cols = profile["columns"]
    by_name = {c["name"]: c for c in cols}
    n_rows = profile["n_rows"]
    usable = [c for c in cols if _usable(c)]
    mentioned = [m for m in _mentioned_columns(requirement, cols) if _usable(by_name[m])]
    date_cols = [c["name"] for c in cols if c["kind"] == "datetime"]
    text_cols = sorted((c for c in cols if c["kind"] == "text"), key=lambda c: -c.get("avg_length", 0))
    drop = [c["name"] for c in cols if not _usable(c)]
    warnings = []

    if _CLUSTER.search(requirement) and not any(w in requirement.lower() for w in ("predict", "classify")):
        return TaskSpec(task="clustering", drop_columns=drop, metric="silhouette",
                        rationale="The requirement asks to find groups/segments and does not name a column to predict.",
                        source="rules", confidence=0.7)

    target = next((m for m in mentioned if by_name[m]["kind"] in {"numeric", "categorical"}), None)
    if not target:
        lower = {c["name"].lower(): c["name"] for c in usable}
        target = next((lower[t] for t in _COMMON_TARGETS if t in lower), None)
    if not target:
        candidates_ = [c["name"] for c in usable if c["kind"] in {"numeric", "categorical"}]
        target = candidates_[-1] if candidates_ else None
        if target:
            warnings.append(f"No target column was named; assumed the last column '{target}'. Please confirm.")
    if not target:
        return TaskSpec(task="clustering", drop_columns=drop, metric="silhouette",
                        rationale="No column looks like something to predict, so the data will be grouped instead.",
                        warnings=["Could not find a target column."], source="rules", confidence=0.3)

    tcol = by_name[target]
    if _FORECAST.search(requirement) and date_cols and tcol["kind"] == "numeric":
        date_col = date_cols[0]
        m = _HORIZON.search(requirement)
        horizon = int(m.group(1)) if m else None
        dinfo = by_name[date_col]
        n_points = dinfo.get("n_unique_dates") or n_rows
        return TaskSpec(task="forecasting", target=target, date_column=date_col, metric="mae",
                        horizon=horizon or max(1, min(30, n_points // 10)), season_length=dinfo.get("season_length"),
                        drop_columns=drop, warnings=warnings, source="rules", confidence=0.75,
                        rationale=f"The requirement mentions forecasting and '{date_col}' gives the time order of '{target}'.")

    if _is_classification_target(tcol, n_rows):
        metric = "f1_macro" if _imbalanced(tcol) else "accuracy"
        other_features = [c for c in usable if c["name"] != target and c["kind"] != "text"]
        if text_cols and (_TEXT.search(requirement) or len(other_features) <= 1):
            text_col = text_cols[0]["name"]
            return TaskSpec(task="text_classification", target=target, text_column=text_col, metric=metric,
                            drop_columns=drop, warnings=warnings, source="rules", confidence=0.75,
                            rationale=f"'{target}' is a category and '{text_col}' holds free text to learn from.")
        return TaskSpec(task="classification", target=target, metric=metric, drop_columns=drop, warnings=warnings,
                        source="rules", confidence=0.7,
                        rationale=f"'{target}' has {tcol['n_unique']} distinct values, so this is classification.")
    return TaskSpec(task="regression", target=target, metric="r2", drop_columns=drop, warnings=warnings,
                    source="rules", confidence=0.7,
                    rationale=f"'{target}' is a continuous number, so this is regression.")


# --------------------------------------------------------------------------------------
# Validation (applies to Claude's proposal and to user edits)
# --------------------------------------------------------------------------------------

def validate_spec(spec: TaskSpec, dataset_kind: str, profile: dict, fallback: TaskSpec | None = None) -> TaskSpec:
    """Raises ValueError for user edits that cannot work; repairs LLM proposals from `fallback`."""
    def fail(msg: str):
        if fallback is None:
            raise ValueError(msg)
        log.info("Repairing spec: %s", msg)
        return fallback

    allowed_metrics = TASKS[spec.task]["metrics"]
    if dataset_kind == "image_folder" and spec.task != "image_classification":
        return fail("An image folder dataset can only be used for image classification.")
    if dataset_kind == "yolo_detection" and spec.task != "object_detection":
        return fail("A YOLO dataset can only be used for object detection.")
    if dataset_kind == "table" and TASKS[spec.task]["modality"] == "image":
        return fail("Image tasks need a ZIP of images, not a table.")
    if spec.metric not in allowed_metrics:
        spec.metric = allowed_metrics[0] if spec.task != "classification" else "accuracy"

    if dataset_kind == "table":
        names = {c["name"]: c for c in profile["columns"]}
        if spec.task != "clustering":
            if spec.target not in names:
                return fail(f"Target column '{spec.target}' does not exist.")
        if spec.task == "classification" and names[spec.target]["n_unique"] < 2:
            return fail("The target must have at least two different values.")
        if spec.task == "regression" and names[spec.target]["kind"] != "numeric":
            return fail(f"Regression needs a numeric target; '{spec.target}' is {names[spec.target]['kind']}.")
        if spec.task == "text_classification" and spec.text_column not in names:
            return fail("Choose the column that contains the text.")
        if spec.task == "forecasting":
            if spec.date_column not in names:
                return fail("Forecasting needs a date/time column.")
            if names[spec.target]["kind"] != "numeric":
                return fail("Forecasting needs a numeric target.")
            if not spec.season_length:
                spec.season_length = names[spec.date_column].get("season_length")
            spec.horizon = spec.horizon or 12
        spec.drop_columns = [c for c in spec.drop_columns if c in names and c not in {spec.target, spec.text_column, spec.date_column}]
    return spec


# --------------------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------------------

_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "enum": ["classification", "regression", "clustering", "forecasting", "text_classification"]},
        "target": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "text_column": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "date_column": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "drop_columns": {"type": "array", "items": {"type": "string"}},
        "metric": {"type": "string", "enum": sorted(METRICS)},
        "horizon": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "n_clusters": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "rationale": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["task", "target", "text_column", "date_column", "drop_columns", "metric", "horizon",
                 "n_clusters", "rationale", "warnings", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = """You are the planning step of an AutoML platform. A user described their goal in plain language and uploaded a table. Decide how the platform should frame the machine-learning problem.

Choose exactly one task:
- classification: predict a category column from the other columns.
- regression: predict a continuous numeric column.
- clustering: no column to predict; the user wants groups or segments.
- forecasting: predict future values of a numeric column ordered by a date column.
- text_classification: predict a category mainly from a free-text column.

Use only column names that appear in the profile, spelled exactly. Put identifiers, leakage-prone columns (values only known after the outcome, or near-copies of the target) and useless columns in drop_columns, and explain each in warnings. Choose the metric that matches what the user cares about; for imbalanced classes prefer f1_macro or balanced_accuracy over accuracy. The rationale is shown to a non-expert user: two sentences, plain language. Confidence is 0-1: how sure you are this framing matches the user's intent."""


def _compact_profile(profile: dict) -> dict:
    cols = []
    for c in profile["columns"][:200]:
        cols.append({k: c[k] for k in ("name", "kind", "n_unique", "missing_pct", "samples") if k in c}
                    | ({"top_values": c["top_values"][:5]} if c.get("top_values") else {})
                    | ({"freq": c.get("freq")} if c["kind"] == "datetime" else {}))
    return {"n_rows": profile["n_rows"], "n_cols": profile["n_cols"], "columns": cols}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _create(**kwargs):
    """Messages call with server-side refusal fallbacks enabled."""
    return _client().beta.messages.create(
        model=get_settings().claude_model,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **kwargs,
    )


def _ask_claude(requirement: str, profile: dict, rules: TaskSpec) -> TaskSpec:
    prompt = (
        f"<requirement>\n{requirement}\n</requirement>\n\n"
        f"<dataset_profile>\n{json.dumps(_compact_profile(profile), default=str)}\n</dataset_profile>\n\n"
        f"A rule-based detector suggested: {rules.model_dump_json(include={'task', 'target', 'text_column', 'date_column', 'metric'})}. "
        "Use it only as a hint."
    )
    response = _create(
        max_tokens=16000,
        system=_SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SPEC_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    return TaskSpec(**data, source="llm")


def explain_results(task: str, metric: str, leaderboard: list[dict], insights: dict) -> str:
    """Plain-English summary of a finished run. Falls back to a template without Claude."""
    template = _template_summary(task, metric, leaderboard)
    if not get_settings().anthropic_api_key or not leaderboard:
        return template
    try:
        response = _create(
            max_tokens=4000,
            output_config={"effort": "low"},
            system="You explain machine-learning results to people who are not data scientists. Be concrete and honest, "
                   "under 170 words, plain prose (no headings, no tables). Mention the winner, how big its lead is, "
                   "what the metric means in practice, and any warning that should change what the user does next.",
            messages=[{"role": "user", "content": json.dumps({
                "task": task, "primary_metric": metric, "metric_help": METRICS.get(metric, {}).get("help"),
                "top_models": leaderboard[:8], "insights": insights,
            }, default=str)}],
        )
        if response.stop_reason == "refusal":
            return template
        return next((b.text for b in response.content if b.type == "text"), template).strip()
    except Exception as e:
        log.warning("Claude summary failed: %s", e)
        return template


def _template_summary(task: str, metric: str, leaderboard: list[dict]) -> str:
    if not leaderboard:
        return "No model finished successfully. Check the errors in the table below."
    best = leaderboard[0]
    label = METRICS.get(metric, {}).get("label", metric)
    s = f"{best['name']} performed best with {label} = {best['score']:.4f} on held-out data"
    if len(leaderboard) > 1:
        s += f", ahead of {leaderboard[1]['name']} ({leaderboard[1]['score']:.4f})"
    return s + f". {len(leaderboard)} models were compared on the same train/test split."
