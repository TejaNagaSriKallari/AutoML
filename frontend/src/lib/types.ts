export type User = { id: string; email: string; name: string };

export type Column = {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "datetime" | "text" | "id" | "constant" | "empty";
  missing_pct: number;
  n_unique: number;
  unique_ratio: number;
  samples: unknown[];
  stats?: { min: number | null; max: number | null; mean: number | null; std: number | null };
  top_values?: { value: unknown; count: number }[];
  avg_length?: number;
  freq?: string | null;
  season_length?: number | null;
};

export type Profile = {
  // table
  n_rows?: number;
  n_cols?: number;
  n_duplicate_rows?: number | null;
  columns?: Column[];
  // images
  n_images?: number;
  n_classes?: number;
  classes?: Record<string, number> | string[];
  layout?: string;
  splits?: Record<string, number | string>;
  median_size?: [number, number] | null;
  notes?: string[];
};

export type TaskName =
  | "classification"
  | "regression"
  | "clustering"
  | "forecasting"
  | "text_classification"
  | "image_classification"
  | "object_detection";

export type Spec = {
  task: TaskName;
  target: string | null;
  text_column: string | null;
  date_column: string | null;
  drop_columns: string[];
  metric: string;
  horizon: number | null;
  season_length: number | null;
  n_clusters: number | null;
  rationale: string;
  warnings: string[];
  source: "llm" | "rules" | "user";
  confidence: number;
};

export type Project = {
  id: string;
  name: string;
  requirement: string;
  dataset_filename: string;
  dataset_kind: "table" | "image_folder" | "yolo_detection" | "unknown";
  dataset_size_bytes: number;
  status: "analyzing" | "ready" | "failed";
  error: string | null;
  profile: Profile | null;
  spec: Spec | null;
  spec_confirmed: boolean;
  created_at: string;
  latest_run: { id: string; status: string; backend: string; created_at: string } | null;
};

export type Candidate = {
  key: string;
  name: string;
  task: TaskName;
  backend: "cpu" | "colab";
  family: string;
  description: string;
  role: "model" | "backbone" | "head";
  source: string;
  default: boolean;
  speed: "fast" | "medium" | "slow";
  max_rows: number | null;
  params: Record<string, unknown>;
  input_kind: string[];
  eligible: boolean;
  reason: string | null;
  note: string | null;
  selected: boolean;
};

export type TaskInfo = { label: string; modality: string; description: string; metrics: string[]; backends: string[] };

export type Candidates = {
  task: TaskName;
  task_info: TaskInfo;
  backends: { cpu: Candidate[]; colab: Candidate[] };
};

export type ModelResult = {
  id: string;
  model_key: string;
  name: string;
  family: string;
  status: "completed" | "failed" | "timeout" | "skipped" | "running";
  metrics: Record<string, number>;
  primary_score: number | null;
  cv_mean: number | null;
  cv_std: number | null;
  train_seconds: number | null;
  params: Record<string, unknown>;
  details: {
    confusion_matrix?: number[][];
    pred_vs_actual?: [number, number][];
    test_pred?: number[];
    future_pred?: number[];
    cluster_sizes?: Record<string, number>;
    chosen_k?: number | null;
    [k: string]: unknown;
  };
  error: string | null;
  rank: number | null;
  is_best: boolean;
};

export type Insights = {
  warnings?: string[];
  leakage?: { column: string; score: number; metric: string }[];
  feature_importance?: { feature: string; importance: number; std: number }[];
  data?: Record<string, unknown> & { classes?: string[] };
  series?: { history_dates: string[]; history: number[]; test_dates: string[]; future_dates: string[] };
  traceback?: string;
};

export type Run = {
  id: string;
  project_id: string;
  backend: "cpu" | "colab";
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | "awaiting_colab";
  config: {
    model_keys: string[];
    custom_models?: string[];
    time_limit_per_model?: number;
    tune_top_k?: number;
    epochs?: number;
  };
  spec: Spec;
  progress: { stage?: string; message?: string; done?: number | null; total?: number | null };
  insights: Insights;
  summary: string | null;
  error: string | null;
  has_model: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  results: ModelResult[];
};

export type MetricInfo = { label: string; higher_is_better: boolean; help: string };

export type Catalog = {
  tasks: Record<TaskName, TaskInfo>;
  metrics: Record<string, MetricInfo>;
  models: Record<TaskName, Candidate[]>;
  custom_models: Record<string, string>;
};
