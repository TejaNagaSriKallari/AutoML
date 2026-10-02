"use client";

import { Download, Sparkles, Trophy } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ClusterSizes, ConfusionMatrix, FeatureImportanceChart, ForecastChart, PredVsActual, ScoreChart } from "@/components/Charts";
import { ColabPanel } from "@/components/ColabPanel";
import { Leaderboard } from "@/components/Leaderboard";
import { PredictPanel } from "@/components/PredictPanel";
import { Shell } from "@/components/Shell";
import { Alert, Badge, Button, Card, CardTitle, ProgressBar, Spinner, StatusBadge } from "@/components/ui";
import { api, download, post } from "@/lib/api";
import { fmtMetric, metricLabel, METRICS, TASK_LABELS, timeAgo } from "@/lib/format";
import type { Run } from "@/lib/types";

const ACTIVE = ["queued", "running", "cancelling", "awaiting_colab"];

export default function RunPage() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setRun(await api<Run>(`/api/runs/${id}`));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  const active = run ? ACTIVE.includes(run.status) : false;
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, run?.backend === "colab" ? 4000 : 2000);
    return () => clearInterval(t);
  }, [active, load, run?.backend]);

  if (error) return <Shell><Alert tone="bad">{error}</Alert></Shell>;
  if (!run) return <Shell><Spinner label="Loading…" /></Shell>;

  const metric = run.spec.metric;
  const best = run.results.find((r) => r.rank === 1);
  const completed = run.results.filter((r) => r.status === "completed");
  const p = run.progress;
  const pct = p.total ? Math.round(((p.done ?? 0) / p.total) * 100) : null;
  const classes = (run.insights.data?.classes as string[] | undefined) ?? (best?.details.classes as string[] | undefined) ?? [];
  const hib = METRICS[metric]?.higher_is_better ?? true;
  const runnerUp = run.results.find((r) => r.rank === 2);

  async function cancel() {
    setBusy(true);
    try {
      setRun(await post<Run>(`/api/runs/${run!.id}/cancel`));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell>
      <div className="space-y-6">
        <div className="flex flex-wrap items-start gap-3">
          <div className="flex-1">
            <Link href={`/projects/${run.project_id}`} className="text-sm text-muted hover:text-text">← Project</Link>
            <h1 className="mt-1 flex flex-wrap items-center gap-2 text-2xl font-semibold">
              {TASK_LABELS[run.spec.task]}
              <StatusBadge status={run.status} />
              <Badge>{run.backend === "cpu" ? "CPU" : "Colab GPU"}</Badge>
            </h1>
            <p className="mt-1 text-sm text-muted">
              Ranked by {metricLabel(metric)} ({hib ? "higher" : "lower"} is better)
              {run.spec.target && run.spec.task !== "image_classification" && <> · target <code>{run.spec.target}</code></>}
              {" "}· started {timeAgo(run.created_at)}
            </p>
          </div>
          {active && run.status !== "cancelling" && (
            <Button variant="danger" size="sm" onClick={cancel} loading={busy}>Cancel</Button>
          )}
        </div>

        {run.backend === "colab" && run.status !== "completed" && <ColabPanel run={run} onChange={setRun} />}

        {active && run.backend === "cpu" && (
          <Card>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span>{p.message ?? "Working…"}</span>
              {p.total ? <span className="tabular text-muted">{p.done ?? 0}/{p.total}</span> : null}
            </div>
            <ProgressBar value={pct} />
            <p className="mt-2 text-xs text-muted">Results appear below as each model finishes. You can leave this page.</p>
          </Card>
        )}
        {run.backend === "colab" && run.status === "running" && p.message && (
          <Card><Spinner label={`Colab: ${p.message}`} /></Card>
        )}

        {run.status === "failed" && run.error && <Alert tone="bad">{run.error}</Alert>}

        {best && run.status === "completed" && (
          <Card className="border-good/40">
            <div className="flex flex-wrap items-start gap-6">
              <div>
                <p className="flex items-center gap-1.5 text-sm text-muted"><Trophy className="h-4 w-4 text-good" /> Best model</p>
                <p className="mt-1 text-xl font-semibold">{best.name}</p>
                <p className="tabular mt-2 text-4xl font-semibold">{fmtMetric(metric, best.primary_score)}</p>
                <p className="text-sm text-muted">
                  {metricLabel(metric)} on held-out test data
                  {runnerUp?.primary_score != null && best.primary_score != null && (
                    <> · next best {fmtMetric(metric, runnerUp.primary_score)}</>
                  )}
                </p>
              </div>
              {run.summary && (
                <div className="min-w-64 flex-1 rounded-lg bg-surface-2 p-4 text-sm leading-relaxed">
                  <p className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted"><Sparkles className="h-3.5 w-3.5" /> Summary</p>
                  {run.summary}
                </div>
              )}
            </div>
            {run.has_model && (
              <div className="mt-5 flex flex-wrap items-start gap-4 border-t border-border pt-4">
                <Button onClick={() => download(`/api/runs/${run.id}/download`, "model.zip")}>
                  <Download className="h-4 w-4" />
                  {run.spec.task === "forecasting" ? "Download forecast (CSV)" : run.spec.task === "clustering" ? "Download rows with clusters (CSV)" : "Download best model"}
                </Button>
                {["classification", "regression", "text_classification", "image_classification"].includes(run.spec.task) && <PredictPanel run={run} />}
              </div>
            )}
          </Card>
        )}

        {(run.insights.warnings?.length ?? 0) > 0 && (
          <div className="space-y-2">
            {run.insights.warnings!.map((w) => <Alert key={w}>{w}</Alert>)}
          </div>
        )}

        {run.results.length > 0 && (
          <Card>
            <CardTitle aside={<span className="text-sm text-muted">{completed.length} of {run.results.length} finished successfully</span>}>
              Leaderboard
            </CardTitle>
            <ScoreChart results={run.results} metric={metric} />
            <div className="mt-4">
              <Leaderboard results={run.results} task={run.spec.task} metric={metric} />
            </div>
            <p className="mt-3 text-xs text-muted">
              All models use the same train/test split.
              {run.results.some((r) => r.cv_mean !== null) && " Ranking uses cross-validation on the training part (more stable); the other columns are measured on the untouched test part."}
              {" "}Click a row for details.
            </p>
          </Card>
        )}

        {run.spec.task === "forecasting" && best && run.insights.series && (
          <Card>
            <CardTitle>Forecast of the best model</CardTitle>
            <ForecastChart series={run.insights.series} best={best} />
          </Card>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          {run.insights.feature_importance && (
            <Card>
              <CardTitle>What drives the best model&apos;s predictions</CardTitle>
              <FeatureImportanceChart items={run.insights.feature_importance} />
              <p className="mt-2 text-xs text-muted">How much the test score drops when a column is shuffled (permutation importance).</p>
            </Card>
          )}
          {best?.details.confusion_matrix && (
            <Card>
              <CardTitle>Where the best model gets confused</CardTitle>
              <ConfusionMatrix matrix={best.details.confusion_matrix} classes={classes} />
            </Card>
          )}
          {best?.details.pred_vs_actual && (
            <Card>
              <CardTitle>Predicted vs actual (test set)</CardTitle>
              <PredVsActual points={best.details.pred_vs_actual} />
            </Card>
          )}
          {best?.details.cluster_sizes && (
            <Card>
              <CardTitle>Cluster sizes: {best.name}</CardTitle>
              <ClusterSizes sizes={best.details.cluster_sizes} />
            </Card>
          )}
        </div>
      </div>
    </Shell>
  );
}
