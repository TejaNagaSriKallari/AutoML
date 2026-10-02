"use client";

import { Upload } from "lucide-react";
import { useRef, useState } from "react";
import { download } from "@/lib/api";
import type { Run } from "@/lib/types";
import { Alert, Button } from "./ui";

export function PredictPanel({ run }: { run: Run }) {
  const ref = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isImage = run.spec.task === "image_classification";

  async function go(file: File) {
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      await download(`/api/runs/${run.id}/predict`, "predictions.csv", { method: "POST", body: form });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (ref.current) ref.current.value = "";
    }
  }

  return (
    <div>
      <Button variant="secondary" loading={busy} onClick={() => ref.current?.click()}>
        <Upload className="h-4 w-4" /> Predict on new {isImage ? "images" : "data"}
      </Button>
      <p className="mt-1 text-xs text-muted">
        {isImage ? "An image or a ZIP of images." : "A file with the same columns as the training data; the target column is not needed."}
      </p>
      <input
        ref={ref}
        type="file"
        hidden
        accept={isImage ? ".zip,.jpg,.jpeg,.png,.bmp,.webp" : ".csv,.tsv,.xlsx,.xls,.parquet,.json"}
        onChange={(e) => e.target.files?.[0] && go(e.target.files[0])}
      />
      {error && <div className="mt-2"><Alert tone="bad">{error}</Alert></div>}
    </div>
  );
}
