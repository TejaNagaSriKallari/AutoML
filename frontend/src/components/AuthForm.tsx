"use client";

import { BrainCircuit } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, type FormEvent } from "react";
import { post } from "@/lib/api";
import { useAuth } from "./AuthProvider";
import { Alert, Button, Field, inputClass } from "./ui";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { refresh } = useAuth();
  const router = useRouter();
  const next = useSearchParams().get("next") || "/dashboard";

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await post(`/api/auth/${mode}`, mode === "login" ? { email, password } : { email, password, name });
      await refresh();
      router.replace(next.startsWith("/") ? next : "/dashboard");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-xl border border-border bg-surface p-6">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <BrainCircuit className="h-5 w-5 text-accent" /> AutoML Studio
        </Link>
        <h1 className="text-xl font-semibold">{mode === "login" ? "Log in" : "Create your account"}</h1>
        {mode === "register" && (
          <Field label="Name">
            <input className={inputClass} value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
          </Field>
        )}
        <Field label="Email">
          <input className={inputClass} type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" />
        </Field>
        <Field label="Password" hint={mode === "register" ? "At least 8 characters." : undefined}>
          <input
            className={inputClass}
            type="password"
            required
            minLength={mode === "register" ? 8 : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
          />
        </Field>
        {error && <Alert tone="bad">{error}</Alert>}
        <Button type="submit" loading={busy} className="w-full">
          {mode === "login" ? "Log in" : "Sign up"}
        </Button>
        <p className="text-center text-sm text-muted">
          {mode === "login" ? (
            <>No account? <Link className="text-accent" href="/register">Sign up</Link></>
          ) : (
            <>Already registered? <Link className="text-accent" href="/login">Log in</Link></>
          )}
        </p>
      </form>
    </div>
  );
}
