"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtMetric, metricLabel } from "@/lib/format";
import type { Insights, ModelResult } from "@/lib/types";

const AXIS = { stroke: "var(--muted)", fontSize: 12, tickLine: false, axisLine: { stroke: "var(--chart-grid)" } };
const GRID = <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="0" vertical={false} />;
const TOOLTIP = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontSize: 12,
    color: "var(--text)",
  },
  labelStyle: { color: "var(--text)", fontWeight: 600 },
  itemStyle: { color: "var(--text)" },
  cursor: { fill: "var(--surface-2)" },
};

/** Horizontal bars of the primary metric, best first. One series: no legend, the title names it. */
export function ScoreChart({ results, metric }: { results: ModelResult[]; metric: string }) {
  const data = results
    .filter((r) => r.status === "completed" && r.primary_score !== null)
    .slice(0, 25)
    .map((r) => ({ name: r.name, score: r.cv_mean ?? r.primary_score, test: r.primary_score, best: r.rank === 1 }));
  if (data.length < 2) return null;
  const height = Math.max(160, data.length * 30 + 40);
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }} barCategoryGap={4}>
          <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
          <XAxis type="number" {...AXIS} tickFormatter={(v) => fmtMetric(metric, v)} domain={["auto", "auto"]} />
          <YAxis type="category" dataKey="name" width={200} {...AXIS} interval={0} />
          <Tooltip
            {...TOOLTIP}
            formatter={(v, k) => [fmtMetric(metric, Number(v)), k === "score" ? "Ranking score" : String(k)]}
          />
          <Bar dataKey="score" maxBarSize={24} radius={[0, 4, 4, 0]} isAnimationActive={false}
               label={{ position: "right", fill: "var(--muted)", fontSize: 11, formatter: (v: unknown) => fmtMetric(metric, Number(v)) }}>
            {data.map((d) => (
              <Cell key={d.name} fill="var(--chart-1)" fillOpacity={d.best ? 1 : 0.45} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function FeatureImportanceChart({ items }: { items: NonNullable<Insights["feature_importance"]> }) {
  const data = items.filter((d) => d.importance > 0).slice(0, 15);
  if (!data.length) return <p className="text-sm text-muted">No single feature changes the score much on its own.</p>;
  return (
    <div style={{ height: data.length * 28 + 40 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
          <XAxis type="number" {...AXIS} tickFormatter={(v) => v.toFixed(2)} />
          <YAxis type="category" dataKey="feature" width={180} {...AXIS} interval={0} />
          <Tooltip {...TOOLTIP} formatter={(v) => [Number(v).toFixed(4), "Score drop when shuffled"]} />
          <Bar dataKey="importance" fill="var(--chart-1)" maxBarSize={20} radius={[0, 4, 4, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConfusionMatrix({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const max = Math.max(1, ...matrix.flat());
  const labels = classes.length === matrix.length ? classes : matrix.map((_, i) => String(i));
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1 text-left font-normal text-muted">actual ↓ / predicted →</th>
            {labels.map((l) => (
              <th key={l} className="max-w-20 truncate p-1 font-medium" title={l}>{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="max-w-32 truncate p-1 text-left font-medium" title={labels[i]}>{labels[i]}</th>
              {row.map((v, j) => {
                const strength = v / max;
                return (
                  <td
                    key={j}
                    title={`actual ${labels[i]}, predicted ${labels[j]}: ${v}`}
                    className="tabular h-9 min-w-10 border-2 border-surface p-1 text-center"
                    style={{
                      background: `color-mix(in oklab, var(--chart-1) ${Math.round(8 + strength * 80)}%, var(--surface))`,
                      color: strength > 0.5 ? "white" : "var(--text)",
                      outline: i === j ? "1px solid var(--chart-grid)" : undefined,
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PredVsActual({ points }: { points: [number, number][] }) {
  const data = points.map(([actual, predicted]) => ({ actual, predicted }));
  const lo = Math.min(...points.flat());
  const hi = Math.max(...points.flat());
  return (
    <div className="h-72">
      <ResponsiveContainer>
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 16 }}>
          {GRID}
          <XAxis type="number" dataKey="actual" name="Actual" domain={[lo, hi]} {...AXIS} tickFormatter={(v) => fmtMetric("rmse", v)}
                 label={{ value: "Actual", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 12 }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[lo, hi]} {...AXIS} tickFormatter={(v) => fmtMetric("rmse", v)} />
          <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="var(--muted)" strokeWidth={1} />
          <Tooltip {...TOOLTIP} formatter={(v) => fmtMetric("rmse", Number(v))} />
          <Scatter data={data} fill="var(--chart-1)" fillOpacity={0.55} stroke="var(--surface)" strokeWidth={1} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ForecastChart({ series, best }: { series: NonNullable<Insights["series"]>; best: ModelResult }) {
  const testPred = (best.details.test_pred ?? []) as number[];
  const futurePred = (best.details.future_pred ?? []) as number[];
  const nHist = series.history.length;
  const testStart = nHist - testPred.length;
  const rows: { date: string; actual?: number; forecast?: number }[] = series.history_dates.map((d, i) => ({
    date: d.slice(0, 10),
    actual: series.history[i],
    forecast: i >= testStart ? testPred[i - testStart] : undefined,
  }));
  // join the forecast line to the last observed point so the future continues the curve
  if (rows.length) rows[rows.length - 1].forecast ??= rows[rows.length - 1].actual;
  series.future_dates.forEach((d, i) => rows.push({ date: String(d).slice(0, 10), forecast: futurePred[i] }));
  const fmt = (v: number) => fmtMetric("rmse", v);
  return (
    <div className="h-80">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ left: 8, right: 16, top: 8, bottom: 4 }}>
          {GRID}
          <XAxis dataKey="date" {...AXIS} minTickGap={40} />
          <YAxis {...AXIS} tickFormatter={fmt} width={64} />
          <Tooltip {...TOOLTIP} cursor={{ stroke: "var(--muted)", strokeWidth: 1 }} formatter={(v, k) => [fmt(Number(v)), String(k)]} />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--text)" }} />
          {testPred.length > 0 && <ReferenceLine x={rows[testStart]?.date} stroke="var(--chart-grid)" label={{ value: "test period", fill: "var(--muted)", fontSize: 11, position: "insideTopLeft" }} />}
          <ReferenceLine x={rows[nHist - 1]?.date} stroke="var(--muted)" label={{ value: "future", fill: "var(--muted)", fontSize: 11, position: "insideTopRight" }} />
          <Line name="Actual" dataKey="actual" stroke="var(--chart-1)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
          <Line name={`Forecast (${best.name})`} dataKey="forecast" stroke="var(--chart-2)" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ClusterSizes({ sizes }: { sizes: Record<string, number> }) {
  const data = Object.entries(sizes).map(([k, v]) => ({ cluster: k === "-1" ? "noise" : `#${k}`, rows: v }));
  return (
    <div className="h-56">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: 8, right: 8, top: 16, bottom: 4 }}>
          {GRID}
          <XAxis dataKey="cluster" {...AXIS} />
          <YAxis {...AXIS} />
          <Tooltip {...TOOLTIP} />
          <Bar dataKey="rows" fill="var(--chart-1)" maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}
               label={{ position: "top", fill: "var(--muted)", fontSize: 11 }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export { metricLabel };
