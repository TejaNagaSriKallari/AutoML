"use client";

import { BrainCircuit, LogOut } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import clsx from "clsx";
import { useAuth } from "./AuthProvider";
import { Spinner } from "./ui";

export function Shell({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuth();
  const pathname = usePathname();
  if (loading || !user)
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner label="Loading…" />
      </div>
    );
  const nav = [
    { href: "/dashboard", label: "Projects" },
    { href: "/models", label: "Model catalog" },
  ];
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-border bg-surface/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4">
          <Link href="/dashboard" className="flex items-center gap-2 font-semibold">
            <BrainCircuit className="h-5 w-5 text-accent" />
            AutoML Studio
          </Link>
          <nav className="flex gap-1">
            {nav.map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className={clsx(
                  "rounded-md px-3 py-1.5 text-sm",
                  pathname.startsWith(n.href) ? "bg-surface-2 font-medium" : "text-muted hover:text-text",
                )}
              >
                {n.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="hidden text-muted sm:inline">{user.email}</span>
            <button onClick={logout} className="rounded-md p-1.5 text-muted hover:bg-surface-2 hover:text-text" title="Log out">
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
    </div>
  );
}
