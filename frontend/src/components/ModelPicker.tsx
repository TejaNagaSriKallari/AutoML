"use client";

import { Cpu, Zap } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { api, post } from "@/lib/api";
import type { Candidate, Candidates, Project, Run } from "@/lib/types";
import { Alert, Badge, Button, Card, CardTitle, Field, inputClass, Spinner } from "./ui";

type Backend = "cpu" | "colab";

const ROLE_TITLES: Record<string, string> = {
  backbone: "Feature extractors (pretrained, frozen)",
  head: "Classifiers trained on those features",
};

export function ModelPicker({ project }: { project: Project }) {
  const [cands, setCands] = useState<Candidates | null>(null);
  const [backend, setBackend] = useState<Backend>("cpu");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [custom, setCustom] = useState("");
  const [timeLimit, setTimeLimit] = useState(300);
  const [tuneTopK, setTuneTopK] = useState(0);
  const [epochs, setEpochs] = useState(10);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const router = useRouter();

  useEffect(() => {
    api<Candidates>(`/api/projects/${project.id}/candidates`)
      .then((c) => {
        setCands(c);
        const b: Backend = c.task_info.backends.includes("cpu") ? "cpu" : "colab";
        setBackend(b);
        setSelected(new Set(c.backends[b].filter((m) => m.selected).map((m) => m.key)));
        setEpochs(c.task === "text_classification" ? 3 : c.task === "object_detection" ? 50 : 10);
      })
      .catch((e) => setError(e.message));
  }, [project.id, project.spec]);

  const list = useMemo(() => cands?.backends[backend] ?? [], [cands, backend]);
  const groups = useMemo(() => {
    const g = new Map<string, Candidate[]>();
    for (const m of list) {
      const k = m.role !== "model" ? m.role : m.family;
      g.set(k, [...(g.get(k) ?? []), m]);
    }
    return [...g.entries()].sort(([a], [b]) => (a === "backbone" ? -1 : b === "backbone" ? 1 : a.localeCompare(b)));
  }, [list]);

  if (error && !cands) return <Alert tone="bad">{error}</Alert>;
  if (!cands) return <Spinner label="Finding suitable models…" />;

  const task = cands.task;
  const combo = list.some((m) => m.role !== "model");
  const customAllowed = task === "image_classification" || task === "text_classification";
  const customList = custom.split(/[\s,]+/).filter(Boolean);
  const nSel = (role?: string) => list.filter((m) => selected.has(m.key) && (!role || m.role === role)).length;
  const total = combo
    ? (nSel("backbone") + (backend === "cpu" ? customList.length : 0)) * nSel("head")
    : nSel() + (backend === "colab" ? customList.length : 0);

  function switchBackend(b: Backend) {
    setBackend(b);
    setSelected(new Set(cands!.backends[b].filter((m) => m.selected).map((m) => m.key)));
    setTimeLimit(b === "colab" ? 1800 : 300);
  }

  function toggle(key: string, on: boolean) {
    setSelected((s) => {
      const n = new Set(s);
      if (on) n.add(key);
      else n.delete(key);
      return n;
    });
  }

  function setGroup(items: Candidate[], on: boolean) {
    setSelected((s) => {
      const n = new Set(s);
      items.filter((m) => m.eligible).forEach((m) => (on ? n.add(m.key) : n.delete(m.key)));
      return n;
    });
  }

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const run = await post<Run>(`/api/projects/${project.id}/runs`, {
        backend,
        model_keys: [...selected],
        custom_models: customAllowed ? customList : [],
        time_limit_per_model: timeLimit,
        tune_top_k: tuneTopK,
        epochs,
      });
      router.push(`/runs/${run.id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  const eligibleCount = list.filter((m) => m.eligible).length;
  return (
    <Card>
      <CardTitle aside={<span className="text-sm text-muted">{eligibleCount} suitable of {list.length} known models</span>}>
        Models to train and compare
      </CardTitle>

      <div className="mb-5 grid gap-3 sm:grid-cols-2">
        {(["cpu", "colab"] as Backend[]).map((b) => {
          const available = cands.task_info.backends.includes(b);
          const Icon = b === "cpu" ? Cpu : Zap;
          return (
            <button
              key={b}
              disabled={!available}
              onClick={() => switchBackend(b)}
              className={`rounded-xl border p-4 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                backend === b ? "border-accent bg-accent-soft" : "border-border hover:border-accent/40"
              }`}
            >
              <div className="flex items-center gap-2 font-medium">
                <Icon className="h-4 w-4" />
                {b === "cpu" ? "Train on our servers (CPU)" : "Train on your free Colab GPU"}
              </div>
              <p className="mt-1 text-sm text-muted">
                {!available
                  ? b === "cpu"
                    ? "Not available on this server. Use the free Colab GPU."
                    : "Not needed: this task trains fast on CPU."
                  : b === "cpu"
                    ? combo
                      ? "Pretrained networks extract features once; many classifiers are compared on them. No setup."
                      : "Every model runs in parallel with a time limit. No setup."
                    : "Full fine-tuning on a GPU. We generate a notebook; results stream back here."}
              </p>
            </button>
          );
        })}
      </div>

      <div className="space-y-4">
        {groups.map(([group, items]) => {
          const on = items.filter((m) => selected.has(m.key)).length;
          return (
            <div key={group}>
              <div className="mb-2 flex items-center gap-2">
                <h3 className="text-sm font-semibold capitalize">{ROLE_TITLES[group] ?? group.replaceAll("_", " ")}</h3>
                <span className="text-xs text-muted">{on}/{items.length}</span>
                <button className="ml-auto text-xs text-accent" onClick={() => setGroup(items, true)}>all</button>
                <button className="text-xs text-muted hover:text-text" onClick={() => setGroup(items, false)}>none</button>
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {items.map((m) => (
                  <label
                    key={m.key}
                    title={m.reason ?? m.description}
                    className={`flex cursor-pointer items-start gap-2 rounded-lg border p-2.5 text-sm ${
                      !m.eligible ? "cursor-not-allowed opacity-50" : selected.has(m.key) ? "border-accent/60 bg-accent-soft/50" : "border-border"
                    }`}
                  >
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      disabled={!m.eligible}
                      checked={selected.has(m.key)}
                      onChange={(e) => toggle(m.key, e.target.checked)}
                    />
                    <span className="min-w-0">
                      <span className="flex flex-wrap items-center gap-1.5 font-medium">
                        {m.name}
                        {m.speed === "slow" && <Badge tone="warn">slow</Badge>}
                      </span>
                      <span className="line-clamp-2 text-xs text-muted">{m.reason ?? m.note ?? m.description}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {customAllowed && (
        <div className="mt-5">
          <Field
            label={task === "image_classification" ? "Add any timm model" : "Add any Hugging Face model"}
            hint={
              task === "image_classification"
                ? "Comma-separated timm names, e.g. resnet18.a1_in1k, efficientnet_b2. Any of the 1,000+ pretrained timm models works."
                : backend === "cpu"
                  ? "Comma-separated sentence-transformers ids, e.g. intfloat/e5-small-v2."
                  : "Comma-separated model ids, e.g. answerdotai/ModernBERT-base."
            }
          >
            <input className={inputClass} value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="optional" />
          </Field>
        </div>
      )}

      <div className="mt-5 grid gap-4 sm:grid-cols-3">
        <Field label="Time limit per model" hint={backend === "cpu" ? "Models over the limit are stopped and marked." : "Per model inside Colab."}>
          <select className={inputClass} value={timeLimit} onChange={(e) => setTimeLimit(Number(e.target.value))}>
            {(backend === "cpu" ? [60, 120, 300, 600, 1200, 1800] : [600, 1200, 1800, 3600, 7200]).map((s) => (
              <option key={s} value={s}>{s < 3600 ? `${s / 60} min` : `${s / 3600} h`}</option>
            ))}
          </select>
        </Field>
        {backend === "cpu" && (task === "classification" || task === "regression") && (
          <Field label="Tune the best models" hint="Hyperparameter search (Optuna) on the top models.">
            <select className={inputClass} value={tuneTopK} onChange={(e) => setTuneTopK(Number(e.target.value))}>
              <option value={0}>No tuning</option>
              {[1, 2, 3, 5].map((k) => <option key={k} value={k}>Top {k}</option>)}
            </select>
          </Field>
        )}
        {backend === "colab" && (
          <Field label="Max epochs" hint="Early stopping ends sooner when validation stops improving.">
            <input className={inputClass} type="number" min={1} max={300} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </Field>
        )}
      </div>

      {error && <div className="mt-4"><Alert tone="bad">{error}</Alert></div>}
      <div className="mt-5 flex flex-wrap items-center justify-end gap-3">
        <span className="text-sm text-muted">{total} model{total === 1 ? "" : "s"} will be trained</span>
        <Button onClick={start} loading={busy} disabled={total === 0}>
          {backend === "cpu" ? "Start training" : "Create Colab notebook"}
        </Button>
      </div>
    </Card>
  );
}
