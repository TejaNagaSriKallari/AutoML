"use client";

import { Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { put } from "@/lib/api";
import { metricLabel, METRICS, TASK_LABELS, TASK_METRICS } from "@/lib/format";
import type { Project, Spec, TaskName } from "@/lib/types";
import { Alert, Badge, Button, Card, CardTitle, Field, inputClass } from "./ui";

const TABLE_TASKS: TaskName[] = ["classification", "regression", "clustering", "forecasting", "text_classification"];

export function SpecEditor({ project, onSaved }: { project: Project; onSaved: (p: Project) => void }) {
  const original = project.spec!;
  const [spec, setSpec] = useState<Spec>(original);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const columns = useMemo(() => project.profile?.columns ?? [], [project.profile]);
  const isTable = project.dataset_kind === "table";
  const tasks: TaskName[] = isTable
    ? TABLE_TASKS
    : project.dataset_kind === "image_folder"
      ? ["image_classification"]
      : ["object_detection"];
  const set = <K extends keyof Spec>(k: K, v: Spec[K]) => setSpec((s) => ({ ...s, [k]: v }));
  const usable = columns.filter((c) => !["id", "constant", "empty"].includes(c.kind));
  const needsTarget = ["classification", "regression", "forecasting", "text_classification"].includes(spec.task);

  function changeTask(task: TaskName) {
    const metrics = TASK_METRICS[task];
    setSpec((s) => ({ ...s, task, metric: metrics.includes(s.metric) ? s.metric : metrics[0] }));
  }

  async function save() {
    setBusy(true);
    setError(null);
    try {
      const body = {
        task: spec.task,
        target: needsTarget ? spec.target : null,
        text_column: spec.task === "text_classification" ? spec.text_column : null,
        date_column: spec.task === "forecasting" ? spec.date_column : null,
        drop_columns: spec.drop_columns,
        metric: spec.metric,
        horizon: spec.task === "forecasting" ? spec.horizon : null,
        season_length: spec.task === "forecasting" ? spec.season_length : null,
        n_clusters: spec.task === "clustering" ? spec.n_clusters : null,
      };
      onSaved(await put<Project>(`/api/projects/${project.id}/spec`, body));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardTitle
        aside={
          <div className="flex items-center gap-2">
            <Badge tone={original.source === "llm" ? "accent" : "neutral"}>
              {original.source === "llm" && <Sparkles className="h-3 w-3" />}
              {original.source === "llm" ? "AI analysis" : original.source === "user" ? "set by you" : "rule-based detection"}
            </Badge>
            {project.spec_confirmed && <Badge tone="good">confirmed</Badge>}
          </div>
        }
      >
        What we understood
      </CardTitle>
      <p className="mb-4 text-sm">
        <span className="font-medium">{TASK_LABELS[original.task]}</span>
        {original.target && original.task !== "image_classification" && (
          <> predicting <code className="rounded bg-surface-2 px-1">{original.target}</code></>
        )}
        {original.rationale && <span className="text-muted"> · {original.rationale}</span>}
      </p>
      {original.warnings.length > 0 && (
        <div className="mb-4 space-y-2">
          {original.warnings.map((w) => (
            <Alert key={w}>{w}</Alert>
          ))}
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Task">
          <select className={inputClass} value={spec.task} onChange={(e) => changeTask(e.target.value as TaskName)} disabled={!isTable}>
            {tasks.map((t) => (
              <option key={t} value={t}>{TASK_LABELS[t]}</option>
            ))}
          </select>
        </Field>
        {isTable && needsTarget && (
          <Field label={spec.task === "forecasting" ? "Value to forecast" : "Column to predict (target)"}>
            <select className={inputClass} value={spec.target ?? ""} onChange={(e) => set("target", e.target.value)}>
              <option value="" disabled>Choose…</option>
              {usable.filter((c) => c.kind !== "text" && c.kind !== "datetime").map((c) => (
                <option key={c.name} value={c.name}>{c.name} ({c.kind}, {c.n_unique} values)</option>
              ))}
            </select>
          </Field>
        )}
        {spec.task === "text_classification" && (
          <Field label="Text column">
            <select className={inputClass} value={spec.text_column ?? ""} onChange={(e) => set("text_column", e.target.value)}>
              <option value="" disabled>Choose…</option>
              {usable.filter((c) => c.kind === "text" || c.kind === "categorical").map((c) => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </Field>
        )}
        {spec.task === "forecasting" && (
          <>
            <Field label="Date column">
              <select className={inputClass} value={spec.date_column ?? ""} onChange={(e) => set("date_column", e.target.value)}>
                <option value="" disabled>Choose…</option>
                {columns.filter((c) => c.kind === "datetime").map((c) => (
                  <option key={c.name} value={c.name}>{c.name}{c.freq ? ` (${c.freq})` : ""}</option>
                ))}
              </select>
            </Field>
            <Field label="Forecast horizon (steps)">
              <input className={inputClass} type="number" min={1} value={spec.horizon ?? ""} onChange={(e) => set("horizon", e.target.value ? Number(e.target.value) : null)} />
            </Field>
            <Field label="Season length" hint="e.g. 12 for monthly data with yearly pattern. Blank = auto.">
              <input className={inputClass} type="number" min={2} value={spec.season_length ?? ""} onChange={(e) => set("season_length", e.target.value ? Number(e.target.value) : null)} />
            </Field>
          </>
        )}
        {spec.task === "clustering" && (
          <Field label="Number of clusters" hint="Leave blank to let each model find the best number.">
            <input className={inputClass} type="number" min={2} max={100} value={spec.n_clusters ?? ""} onChange={(e) => set("n_clusters", e.target.value ? Number(e.target.value) : null)} />
          </Field>
        )}
        <Field label="Metric used to rank models" hint={METRICS[spec.metric]?.help}>
          <select className={inputClass} value={spec.metric} onChange={(e) => set("metric", e.target.value)}>
            {TASK_METRICS[spec.task].map((m) => (
              <option key={m} value={m}>{metricLabel(m)}</option>
            ))}
          </select>
        </Field>
      </div>

      {isTable && spec.task !== "text_classification" && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium">
            Columns excluded from training ({spec.drop_columns.length})
          </summary>
          <p className="mt-2 text-xs text-muted">IDs and constant columns are excluded automatically. Also exclude anything only known after the outcome.</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {columns.filter((c) => c.name !== spec.target && c.name !== spec.date_column).map((c) => {
              const auto = ["id", "constant", "empty"].includes(c.kind);
              const on = auto || spec.drop_columns.includes(c.name);
              return (
                <label key={c.name} className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs ${on ? "border-bad/40 bg-bad-soft" : "border-border"}`}>
                  <input
                    type="checkbox"
                    checked={on}
                    disabled={auto}
                    onChange={(e) =>
                      set("drop_columns", e.target.checked ? [...spec.drop_columns, c.name] : spec.drop_columns.filter((d) => d !== c.name))
                    }
                  />
                  {c.name}
                  {auto && <span className="text-muted">({c.kind})</span>}
                </label>
              );
            })}
          </div>
        </details>
      )}

      {error && <div className="mt-4"><Alert tone="bad">{error}</Alert></div>}
      <div className="mt-5 flex justify-end">
        <Button onClick={save} loading={busy}>
          {project.spec_confirmed ? "Save changes" : "Looks right — choose models"}
        </Button>
      </div>
    </Card>
  );
}
