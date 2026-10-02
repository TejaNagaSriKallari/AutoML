"use client";

import { Download, ExternalLink, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { api, download, post } from "@/lib/api";
import type { Run } from "@/lib/types";
import { Alert, Button, Card, CardTitle } from "./ui";

export function ColabPanel({ run, onChange }: { run: Run; onChange: (r: Run) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function act(name: string, fn: () => Promise<void>) {
    setBusy(name);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const importResults = (file: File) =>
    act("import", async () => {
      const form = new FormData();
      form.append("file", file);
      onChange(await api<Run>(`/api/runs/${run.id}/import-results`, { method: "POST", body: form }));
    });

  return (
    <Card>
      <CardTitle>Train on your free Colab GPU</CardTitle>
      <ol className="mb-5 space-y-3 text-sm">
        <li>
          <b>1. Download the notebook.</b> All the training code is already written inside it, along with your models and a
          private token. You don&apos;t write any code.
          <div className="mt-2 flex flex-wrap gap-2">
            <Button size="sm" loading={busy === "nb"} onClick={() => act("nb", () => download(`/api/runs/${run.id}/notebook`, "automl_colab.ipynb"))}>
              <Download className="h-4 w-4" /> Download notebook
            </Button>
          </div>
        </li>
        <li>
          <b>2. Upload it to Colab.</b> Open Colab, choose the <i>Upload</i> tab in the &quot;Open notebook&quot; window (or{" "}
          <i>File → Upload notebook</i>), and pick the file you just downloaded. Opening Colab alone shows an empty notebook.
          <div className="mt-2">
            <a href="https://colab.research.google.com/" target="_blank" rel="noreferrer"
               className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border px-3 text-sm hover:bg-surface-2">
              Open Colab to upload the notebook <ExternalLink className="h-3.5 w-3.5" />
            </a>
          </div>
        </li>
        <li>
          <b>3. Turn on the GPU and run:</b> <i>Runtime → Change runtime type → T4 GPU</i>, then <i>Runtime → Run all</i>.
        </li>
        <li>
          <b>4. Watch results arrive here.</b> If Colab can&apos;t reach this app&apos;s server, the notebook switches to manual mode: it asks you to
          upload the dataset there, and at the end downloads <code>results.json</code>. Import that file here:
          <div className="mt-2 flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" loading={busy === "import"} onClick={() => fileRef.current?.click()}>
              <Upload className="h-4 w-4" /> Import results.json
            </Button>
            {run.results.length > 0 && run.status !== "completed" && (
              <Button size="sm" variant="secondary" loading={busy === "finish"}
                      onClick={() => act("finish", async () => onChange(await post<Run>(`/api/runs/${run.id}/finish`)))}>
                Mark as finished
              </Button>
            )}
            <input ref={fileRef} type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importResults(e.target.files[0])} />
          </div>
        </li>
      </ol>
      <p className="text-xs text-muted">
        Downloading the notebook again creates a new token, and older copies stop reporting. Free Colab sessions can disconnect after a few
        hours; models already finished are kept, so you can run the notebook again.
      </p>
      {error && <div className="mt-3"><Alert tone="bad">{error}</Alert></div>}
    </Card>
  );
}
