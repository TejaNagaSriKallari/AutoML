"use client";

import { FolderOpen, Plus } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Shell } from "@/components/Shell";
import { Alert, Badge, Spinner, StatusBadge } from "@/components/ui";
import { api } from "@/lib/api";
import { fmtBytes, TASK_LABELS, timeAgo } from "@/lib/format";
import type { Project } from "@/lib/types";

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Project[]>("/api/projects").then(setProjects).catch((e) => setError(e.message));
  }, []);

  return (
    <Shell>
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Projects</h1>
        <Link href="/projects/new" className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-white">
          <Plus className="h-4 w-4" /> New project
        </Link>
      </div>
      {error && <Alert tone="bad">{error}</Alert>}
      {!projects && !error && <Spinner label="Loading projects…" />}
      {projects?.length === 0 && (
        <div className="rounded-xl border border-dashed border-border p-12 text-center">
          <FolderOpen className="mx-auto mb-3 h-8 w-8 text-muted" />
          <p className="font-medium">No projects yet</p>
          <p className="mt-1 text-sm text-muted">Describe a problem and upload a dataset to compare models.</p>
          <Link href="/projects/new" className="mt-4 inline-flex h-10 items-center rounded-lg bg-accent px-4 text-sm font-medium text-white">
            Create your first project
          </Link>
        </div>
      )}
      <div className="grid gap-3">
        {projects?.map((p) => (
          <Link key={p.id} href={`/projects/${p.id}`} className="rounded-xl border border-border bg-surface p-4 hover:border-accent/50">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{p.name}</span>
              {p.spec && <Badge tone="accent">{TASK_LABELS[p.spec.task]}</Badge>}
              <StatusBadge status={p.latest_run?.status ?? p.status} />
              <span className="ml-auto text-xs text-muted">{timeAgo(p.created_at)}</span>
            </div>
            <p className="mt-1 line-clamp-1 text-sm text-muted">{p.requirement}</p>
            <p className="mt-1 text-xs text-muted">
              {p.dataset_filename} · {fmtBytes(p.dataset_size_bytes)}
            </p>
          </Link>
        ))}
      </div>
    </Shell>
  );
}
