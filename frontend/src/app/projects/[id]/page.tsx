"use client";

import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ModelPicker } from "@/components/ModelPicker";
import { ProfileView } from "@/components/ProfileView";
import { Shell } from "@/components/Shell";
import { SpecEditor } from "@/components/SpecEditor";
import { Alert, Button, Card, CardTitle, Spinner, StatusBadge } from "@/components/ui";
import { api, post } from "@/lib/api";
import { fmtBytes, timeAgo } from "@/lib/format";
import type { Project, Run } from "@/lib/types";

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const load = useCallback(async () => {
    try {
      const p = await api<Project>(`/api/projects/${id}`);
      setProject(p);
      if (p.status === "ready") setRuns(await api<Run[]>(`/api/projects/${id}/runs`));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (project?.status !== "analyzing") return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [project?.status, load]);

  async function remove() {
    if (!confirm("Delete this project, its dataset and all training results?")) return;
    try {
      await api(`/api/projects/${id}`, { method: "DELETE" });
      router.replace("/dashboard");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function reanalyze() {
    setProject(await post<Project>(`/api/projects/${id}/reanalyze`));
  }

  return (
    <Shell>
      {error && <Alert tone="bad">{error}</Alert>}
      {!project && !error && <Spinner label="Loading…" />}
      {project && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-start gap-3">
            <div className="min-w-0 flex-1">
              <Link href="/dashboard" className="text-sm text-muted hover:text-text">← Projects</Link>
              <h1 className="mt-1 text-2xl font-semibold">{project.name}</h1>
              <p className="mt-1 max-w-3xl text-muted">{project.requirement}</p>
              <p className="mt-1 text-xs text-muted">
                {project.dataset_filename} · {fmtBytes(project.dataset_size_bytes)} · created {timeAgo(project.created_at)}
              </p>
            </div>
            <Button variant="danger" size="sm" onClick={remove}>
              <Trash2 className="h-4 w-4" /> Delete
            </Button>
          </div>

          {project.status === "analyzing" && (
            <Card>
              <Spinner label="Analysing your data and requirement: detecting column types, the task, the target and the metric…" />
            </Card>
          )}
          {project.status === "failed" && (
            <Card className="space-y-3">
              <Alert tone="bad">{project.error}</Alert>
              <Button variant="secondary" onClick={reanalyze}>Try again</Button>
            </Card>
          )}

          {project.status === "ready" && project.spec && (
            <>
              <SpecEditor project={project} onSaved={setProject} />
              {project.spec_confirmed ? (
                <ModelPicker project={project} />
              ) : (
                <p className="text-sm text-muted">Confirm the setup above to see every model that can solve it.</p>
              )}
              {runs.length > 0 && (
                <Card>
                  <CardTitle>Training runs</CardTitle>
                  <div className="divide-y divide-border">
                    {runs.map((r) => {
                      const best = r.results.find((x) => x.rank === 1);
                      return (
                        <Link key={r.id} href={`/runs/${r.id}`} className="flex flex-wrap items-center gap-3 py-3 text-sm hover:text-accent">
                          <StatusBadge status={r.status} />
                          <span>{r.backend === "cpu" ? "CPU" : "Colab GPU"}</span>
                          <span className="text-muted">{r.results.length} models</span>
                          {best && <span>best: <b>{best.name}</b></span>}
                          <span className="ml-auto text-xs text-muted">{timeAgo(r.created_at)}</span>
                        </Link>
                      );
                    })}
                  </div>
                </Card>
              )}
              <details>
                <summary className="cursor-pointer text-sm font-medium">Dataset profile</summary>
                <div className="mt-3">
                  <ProfileView project={project} />
                </div>
              </details>
            </>
          )}
        </div>
      )}
    </Shell>
  );
}
