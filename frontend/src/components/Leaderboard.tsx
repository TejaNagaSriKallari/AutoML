"use client";

import { ChevronDown, ChevronRight, Trophy } from "lucide-react";
import { Fragment, useState } from "react";
import { fmtMetric, fmtSeconds, METRICS, metricLabel, TASK_METRICS } from "@/lib/format";
import type { ModelResult, TaskName } from "@/lib/types";
import { StatusBadge } from "./ui";

export function Leaderboard({ results, task, metric }: { results: ModelResult[]; task: TaskName; metric: string }) {
  const [open, setOpen] = useState<string | null>(null);
  const shown = new Set(results.flatMap((r) => Object.keys(r.metrics)));
  const cols = [metric, ...TASK_METRICS[task].filter((m) => m !== metric && shown.has(m))];
  if (task === "clustering") cols.push(...["n_clusters", "noise_pct"].filter((m) => shown.has(m)));
  const hasCv = results.some((r) => r.cv_mean !== null);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-muted">
          <tr className="border-b border-border">
            <th className="w-8 py-2 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Model</th>
            {hasCv && (
              <th className="py-2 pr-3 text-right font-medium" title="Mean ± std across cross-validation folds of the training data (used for ranking)">
                CV {metricLabel(metric)}
              </th>
            )}
            {cols.map((m) => (
              <th key={m} className="py-2 pr-3 text-right font-medium" title={`Test set. ${METRICS[m]?.help ?? ""}`}>
                {m === metric ? <span className="text-text">{metricLabel(m)}</span> : metricLabel(m)}
              </th>
            ))}
            <th className="py-2 pr-3 text-right font-medium">Time</th>
            <th className="py-2 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const isOpen = open === r.id;
            return (
              <Fragment key={r.id}>
                <tr
                  onClick={() => setOpen(isOpen ? null : r.id)}
                  className={`cursor-pointer border-b border-border/60 hover:bg-surface-2 ${r.rank === 1 ? "bg-good-soft/50" : ""}`}
                >
                  <td className="tabular py-2 text-muted">
                    {r.rank === 1 ? <Trophy className="h-4 w-4 text-good" aria-label="best" /> : r.rank ?? "–"}
                  </td>
                  <td className="py-2 pr-3">
                    <span className="flex items-center gap-1 font-medium">
                      {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5 text-muted" />}
                      {r.name}
                    </span>
                    <span className="ml-5 text-xs text-muted">{r.family.replaceAll("_", " ")}</span>
                  </td>
                  {hasCv && (
                    <td className="tabular py-2 pr-3 text-right">
                      {r.cv_mean !== null ? fmtMetric(metric, r.cv_mean) : "–"}
                      {r.cv_std !== null && <span className="text-xs text-muted"> ±{fmtMetric(metric, r.cv_std)}</span>}
                    </td>
                  )}
                  {cols.map((m) => (
                    <td key={m} className={`tabular py-2 pr-3 text-right ${m === metric ? "font-semibold" : "text-muted"}`}>
                      {fmtMetric(m, r.metrics[m])}
                    </td>
                  ))}
                  <td className="tabular py-2 pr-3 text-right text-muted">{fmtSeconds(r.train_seconds)}</td>
                  <td className="py-2"><StatusBadge status={r.status} /></td>
                </tr>
                {isOpen && (
                  <tr className="border-b border-border/60 bg-surface-2/50">
                    <td />
                    <td colSpan={cols.length + 4} className="py-3 pr-3 text-xs">
                      {r.error && <pre className="mb-2 whitespace-pre-wrap rounded bg-bad-soft p-2 text-bad">{r.error}</pre>}
                      {Object.keys(r.params).length > 0 && (
                        <div className="mb-1">
                          <span className="font-medium">Parameters: </span>
                          <code className="break-all text-muted">{JSON.stringify(r.params)}</code>
                        </div>
                      )}
                      <div className="text-muted">key: <code>{r.model_key}</code></div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
