"use client";

import { useEffect, useMemo, useState } from "react";
import { Shell } from "@/components/Shell";
import { Alert, Badge, Card, inputClass, Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import { metricLabel, TASK_LABELS } from "@/lib/format";
import type { Catalog, TaskName } from "@/lib/types";

export default function ModelsPage() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState<TaskName>("classification");
  const [q, setQ] = useState("");

  useEffect(() => {
    api<Catalog>("/api/catalog").then(setCat).catch((e) => setError(e.message));
  }, []);

  const models = useMemo(() => {
    const list = cat?.models[task] ?? [];
    const s = q.toLowerCase();
    return list.filter((m) => !s || `${m.name} ${m.family} ${m.description}`.toLowerCase().includes(s));
  }, [cat, task, q]);

  const total = cat ? Object.values(cat.models).reduce((n, l) => n + l.length, 0) : 0;

  return (
    <Shell>
      <h1 className="text-2xl font-semibold">Model catalog</h1>
      <p className="mt-1 mb-6 text-muted">
        {total} built-in models across {cat ? Object.keys(cat.tasks).length : "…"} tasks. Tabular models are discovered automatically
        from scikit-learn, plus XGBoost, LightGBM and CatBoost. For images and text you can add any timm or Hugging Face model by name.
      </p>
      {error && <Alert tone="bad">{error}</Alert>}
      {!cat && !error && <Spinner label="Loading catalog…" />}
      {cat && (
        <>
          <div className="mb-4 flex flex-wrap gap-2">
            {(Object.keys(cat.tasks) as TaskName[]).map((t) => (
              <button
                key={t}
                onClick={() => setTask(t)}
                className={`rounded-full border px-3 py-1 text-sm ${t === task ? "border-accent bg-accent-soft text-accent" : "border-border text-muted hover:text-text"}`}
              >
                {TASK_LABELS[t]} <span className="text-xs">({cat.models[t].length})</span>
              </button>
            ))}
          </div>
          <Card className="mb-4">
            <p className="text-sm">{cat.tasks[task].description}</p>
            <p className="mt-2 text-xs text-muted">
              Metrics: {cat.tasks[task].metrics.map(metricLabel).join(", ")} · Runs on:{" "}
              {cat.tasks[task].backends.map((b) => (b === "cpu" ? "CPU" : "Colab GPU")).join(" and ")}
            </p>
          </Card>
          <input className={`${inputClass} mb-4 max-w-sm`} placeholder="Search models…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {models.map((m) => (
              <div key={m.key} className="rounded-lg border border-border bg-surface p-3">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="font-medium">{m.name}</span>
                  <Badge tone={m.backend === "cpu" ? "neutral" : "accent"}>{m.backend === "cpu" ? "CPU" : "GPU"}</Badge>
                  {m.role !== "model" && <Badge>{m.role}</Badge>}
                </div>
                <p className="mt-1 line-clamp-2 text-xs text-muted">{m.description}</p>
                <p className="mt-1 text-xs text-muted">{m.family.replaceAll("_", " ")} · {m.source}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </Shell>
  );
}
