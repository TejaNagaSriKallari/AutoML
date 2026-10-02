import path from "node:path";
import type { NextConfig } from "next";

// In production the browser only talks to this site: /api/* is proxied to the backend, so the
// session cookie stays first-party (browsers block cookies set by a backend on another domain).
// Set BACKEND_URL at build time and NEXT_PUBLIC_API_URL=/ to enable it. Local dev leaves both unset.
const backendUrl = process.env.BACKEND_URL?.replace(/\/$/, "");

const nextConfig: NextConfig = {
  outputFileTracingRoot: path.join(__dirname),
  experimental: {
    // Uploads and predictions can take a while; the default proxy timeout is 30s.
    proxyTimeout: 10 * 60 * 1000,
  },
  async rewrites() {
    return backendUrl ? [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }] : [];
  },
};

export default nextConfig;
