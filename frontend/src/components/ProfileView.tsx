"use client";

import type { Project } from "@/lib/types";
import { Badge, Card, CardTitle } from "./ui";

const KIND_TONE: Record<string, "neutral" | "accent" | "good" | "warn" | "bad"> = {
  numeric: "accent",
  categorical: "good",
  datetime: "warn",
  text: "neutral",
  id: "bad",
  constant: "bad",
  empty: "bad",
};

export function ProfileView({ project }: { project: Project }) {
  const p = project.profile;
  if (!p) return null;

  if (project.dataset_kind === "image_folder" || project.dataset_kind === "yolo_detection") {
    const classes = Array.isArray(p.classes)
      ? p.classes.map((c) => [c, null] as [string, number | null])
      : Object.entries(p.classes ?? {});
    const max = Math.max(1, ...classes.map(([, n]) => n ?? 0));
    return (
      <Card>
        <CardTitle>Dataset</CardTitle>
        <div className="mb-4 flex flex-wrap gap-4 text-sm">
          <span><b>{p.n_images?.toLocaleString()}</b> images</span>
          <span><b>{p.n_classes}</b> classes</span>
          {p.layout && <span className="text-muted">layout: {p.layout.replace("_", " ")}</span>}
          {p.median_size && <span className="text-muted">median size {p.median_size[0]}×{p.median_size[1]}</span>}
        </div>
        {p.notes?.map((n) => <p key={n} className="mb-2 text-sm text-warn">{n}</p>)}
        <div className="max-h-72 space-y-1.5 overflow-y-auto pr-2">
          {classes.map(([name, n]) => (
            <div key={name} className="flex items-center gap-3 text-sm">
              <span className="w-40 truncate" title={name}>{name}</span>
              {n !== null && (
                <>
                  <div className="h-2 flex-1 rounded-full bg-surface-2">
                    <div className="h-2 rounded-full bg-accent" style={{ width: `${(n / max) * 100}%` }} />
                  </div>
                  <span className="tabular w-12 text-right text-muted">{n}</span>
                </>
              )}
            </div>
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <CardTitle>
        Dataset · {p.n_rows?.toLocaleString()} rows × {p.n_cols} columns
      </CardTitle>
      {!!p.n_duplicate_rows && <p className="mb-3 text-sm text-warn">{p.n_duplicate_rows} duplicate rows.</p>}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-muted">
            <tr className="border-b border-border">
              <th className="py-2 pr-3 font-medium">Column</th>
              <th className="py-2 pr-3 font-medium">Type</th>
              <th className="py-2 pr-3 text-right font-medium">Missing</th>
              <th className="py-2 pr-3 text-right font-medium">Distinct</th>
              <th className="py-2 font-medium">Examples / range</th>
            </tr>
          </thead>
          <tbody>
            {p.columns?.map((c) => (
              <tr key={c.name} className="border-b border-border/60 last:border-0">
                <td className="py-2 pr-3 font-medium">{c.name}</td>
                <td className="py-2 pr-3"><Badge tone={KIND_TONE[c.kind]}>{c.kind}</Badge></td>
                <td className={`tabular py-2 pr-3 text-right ${c.missing_pct > 20 ? "text-warn" : ""}`}>{c.missing_pct}%</td>
                <td className="tabular py-2 pr-3 text-right">{c.n_unique.toLocaleString()}</td>
                <td className="max-w-md truncate py-2 text-muted">
                  {c.stats && c.stats.min !== null
                    ? `${c.stats.min} … ${c.stats.max} (mean ${Number(c.stats.mean).toFixed(2)})`
                    : c.samples.slice(0, 3).map(String).join(", ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
