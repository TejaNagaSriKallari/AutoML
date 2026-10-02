import type { MetricInfo } from "./types";

export const TASK_LABELS: Record<string, string> = {
  classification: "Tabular classification",
  regression: "Tabular regression",
  clustering: "Clustering",
  forecasting: "Time-series forecasting",
  text_classification: "Text classification",
  image_classification: "Image classification",
  object_detection: "Object detection",
};

export const METRICS: Record<string, MetricInfo> = {
  accuracy: { label: "Accuracy", higher_is_better: true, help: "Share of predictions that are correct." },
  balanced_accuracy: { label: "Balanced acc.", higher_is_better: true, help: "Average recall per class; robust to imbalance." },
  f1_macro: { label: "F1 (macro)", higher_is_better: true, help: "Precision/recall balance averaged over classes." },
  precision_macro: { label: "Precision", higher_is_better: true, help: "When the model predicts a class, how often it is right." },
  recall_macro: { label: "Recall", higher_is_better: true, help: "How many of each class the model finds." },
  roc_auc: { label: "ROC AUC", higher_is_better: true, help: "Ranking quality (1.0 = perfect)." },
  log_loss: { label: "Log loss", higher_is_better: false, help: "Penalises confident wrong probabilities." },
  r2: { label: "R²", higher_is_better: true, help: "Variance explained (1.0 = perfect, 0 = predicting the mean)." },
  rmse: { label: "RMSE", higher_is_better: false, help: "Typical error size in target units." },
  mae: { label: "MAE", higher_is_better: false, help: "Average absolute error in target units." },
  mape: { label: "MAPE %", higher_is_better: false, help: "Average percentage error." },
  smape: { label: "sMAPE %", higher_is_better: false, help: "Symmetric percentage error." },
  silhouette: { label: "Silhouette", higher_is_better: true, help: "Cluster separation (-1 to 1)." },
  calinski_harabasz: { label: "Calinski-H.", higher_is_better: true, help: "Between/within cluster spread." },
  davies_bouldin: { label: "Davies-B.", higher_is_better: false, help: "Cluster similarity (lower = more distinct)." },
  map50_95: { label: "mAP50-95", higher_is_better: true, help: "Detection quality over IoU thresholds." },
  map50: { label: "mAP50", higher_is_better: true, help: "Detection quality at 50% overlap." },
  precision: { label: "Precision", higher_is_better: true, help: "Correct share of predicted boxes." },
  recall: { label: "Recall", higher_is_better: true, help: "Share of objects found." },
  n_clusters: { label: "Clusters", higher_is_better: true, help: "Number of clusters found." },
  noise_pct: { label: "Noise %", higher_is_better: false, help: "Rows not assigned to any cluster." },
};

export const TASK_METRICS: Record<string, string[]> = {
  classification: ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc", "log_loss"],
  regression: ["r2", "rmse", "mae", "mape"],
  clustering: ["silhouette", "calinski_harabasz", "davies_bouldin"],
  forecasting: ["mae", "rmse", "mape", "smape"],
  text_classification: ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"],
  image_classification: ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"],
  object_detection: ["map50_95", "map50", "precision", "recall"],
};

export function metricLabel(m: string) {
  return METRICS[m]?.label ?? m;
}

export function fmtMetric(metric: string, v: number | null | undefined) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "precision", "recall"].includes(metric))
    return `${(v * 100).toFixed(1)}%`;
  if (metric === "n_clusters") return String(v);
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10) return v.toFixed(2);
  return v.toFixed(4);
}

export function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtSeconds(s: number | null | undefined) {
  if (s === null || s === undefined) return "–";
  if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function timeAgo(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
