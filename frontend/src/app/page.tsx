"use client";

import { BrainCircuit, Cpu, FileUp, Trophy } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useAuth } from "@/components/AuthProvider";

export default function Landing() {
  const { user } = useAuth();
  const router = useRouter();
  useEffect(() => {
    if (user) router.replace("/dashboard");
  }, [user, router]);

  const steps = [
    { icon: FileUp, title: "Describe & upload", text: "Write what you want to predict and upload a table, texts, or a ZIP of images." },
    { icon: BrainCircuit, title: "We frame the problem", text: "The task, target and metric are detected from your words and your data. You confirm." },
    { icon: Cpu, title: "Every suitable model trains", text: "On our CPUs, or fine-tuned on your free Colab GPU, all on the same test split." },
    { icon: Trophy, title: "Compare & use the best", text: "Leaderboard, plain-English summary, download the winner or predict on new data." },
  ];
  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-10">
      <header className="flex items-center gap-2 font-semibold">
        <BrainCircuit className="h-5 w-5 text-accent" /> AutoML Studio
        <div className="ml-auto flex gap-2 text-sm">
          <Link href="/login" className="rounded-lg px-3 py-1.5 hover:bg-surface-2">Log in</Link>
          <Link href="/register" className="rounded-lg bg-accent px-3 py-1.5 font-medium text-white">Sign up</Link>
        </div>
      </header>
      <section className="py-20">
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Tell us the problem. We train every model that fits, and show you which one wins.
        </h1>
        <p className="mt-5 max-w-2xl text-lg text-muted">
          Classification, regression, clustering, forecasting, text and image classification, object detection.
          No code, and no GPU needed.
        </p>
        <Link href="/register" className="mt-8 inline-flex h-11 items-center rounded-lg bg-accent px-5 font-medium text-white">
          Start a project
        </Link>
      </section>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map((s) => (
          <div key={s.title} className="rounded-xl border border-border bg-surface p-5">
            <s.icon className="mb-3 h-5 w-5 text-accent" />
            <h3 className="font-medium">{s.title}</h3>
            <p className="mt-1 text-sm text-muted">{s.text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
