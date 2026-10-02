"use client";

import { FileUp, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState, type DragEvent, type FormEvent } from "react";
import { Shell } from "@/components/Shell";
import { Alert, Button, Card, Field, inputClass, ProgressBar } from "@/components/ui";
import { uploadWithProgress } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import type { Project } from "@/lib/types";

const EXAMPLES = [
  "Predict which customers will churn next month from their usage and billing history.",
  "Forecast weekly sales for the next 8 weeks.",
  "Classify product photos into defective and OK.",
  "Detect the sentiment (positive/negative/neutral) of support tickets.",
  "Group our customers into segments with similar behaviour.",
];

export default function NewProject() {
  const [name, setName] = useState("");
  const [requirement, setRequirement] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return setError("Choose a dataset file.");
    setError(null);
    const form = new FormData();
    form.append("name", name || file.name.replace(/\.[^.]+$/, ""));
    form.append("requirement", requirement);
    form.append("file", file);
    setProgress(0);
    try {
      const p = await uploadWithProgress<Project>("/api/projects", form, setProgress);
      router.push(`/projects/${p.id}`);
    } catch (err) {
      setError((err as Error).message);
      setProgress(null);
    }
  }

  return (
    <Shell>
      <h1 className="mb-6 text-2xl font-semibold">New project</h1>
      <form onSubmit={submit} className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <Card className="space-y-5">
          <Field label="Project name">
            <input className={inputClass} value={name} onChange={(e) => setName(e.target.value)} placeholder="Customer churn" maxLength={200} />
          </Field>
          <Field label="What do you want to achieve?" hint="Plain language is fine. Mention the column you want to predict if you know it.">
            <textarea
              className={`${inputClass} min-h-32`}
              required
              minLength={5}
              maxLength={5000}
              value={requirement}
              onChange={(e) => setRequirement(e.target.value)}
              placeholder="e.g. Predict whether a loan applicant will default, using the 'default' column."
            />
          </Field>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button type="button" key={ex} onClick={() => setRequirement(ex)} className="rounded-full border border-border px-3 py-1 text-xs text-muted hover:text-text">
                {ex.length > 48 ? ex.slice(0, 46) + "…" : ex}
              </button>
            ))}
          </div>
          <div>
            <span className="mb-1 block text-sm font-medium">Dataset</span>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
                dragging ? "border-accent bg-accent-soft" : "border-border hover:border-accent/50"
              }`}
            >
              <FileUp className="mb-2 h-6 w-6 text-muted" />
              {file ? (
                <div className="flex items-center gap-2 text-sm">
                  <span className="font-medium">{file.name}</span>
                  <span className="text-muted">{fmtBytes(file.size)}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                    }}
                    className="rounded p-0.5 text-muted hover:text-text"
                    aria-label="Remove file"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <>
                  <p className="text-sm font-medium">Drop a file here or click to choose</p>
                  <p className="mt-1 text-xs text-muted">CSV, Excel, Parquet, JSON, or ZIP (images / YOLO dataset)</p>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                hidden
                accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json,.jsonl,.zip"
                onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
              />
            </div>
          </div>
          {progress !== null && (
            <div className="space-y-1">
              <ProgressBar value={progress} />
              <p className="text-xs text-muted">{progress < 100 ? `Uploading… ${progress}%` : "Upload finished, analysing…"}</p>
            </div>
          )}
          {error && <Alert tone="bad">{error}</Alert>}
          <Button type="submit" loading={progress !== null} disabled={!file || requirement.length < 5}>
            Upload and analyse
          </Button>
        </Card>
        <Card className="h-fit space-y-3 text-sm">
          <h2 className="font-semibold">Supported data</h2>
          <div>
            <p className="font-medium">Tables</p>
            <p className="text-muted">One row per example. Used for classification, regression, clustering, forecasting (with a date column) and text classification (with a text column).</p>
          </div>
          <div>
            <p className="font-medium">Image classification (ZIP)</p>
            <pre className="mt-1 rounded-lg bg-surface-2 p-2 text-xs">{`dataset.zip
├── cats/  img1.jpg …
└── dogs/  img7.jpg …`}</pre>
            <p className="mt-1 text-muted">Also: train/val/test folders, or images + labels.csv.</p>
          </div>
          <div>
            <p className="font-medium">Object detection (ZIP)</p>
            <p className="text-muted">YOLO format: data.yaml + images/ + labels/.</p>
          </div>
        </Card>
      </form>
    </Shell>
  );
}
