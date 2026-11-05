import { MetricCard } from "@/components/metric-card";
import { PageHeader } from "@/components/page-header";
import { getOverview, getServices } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function MonitoringPage() {
  const [overview, services] = await Promise.all([getOverview(), getServices()]);
  const grafana = services.find((service) => service.name === "Grafana");
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="Observability" title="Monitoring" description="Tóm tắt tín hiệu drift và trạng thái phục vụ; Grafana giữ dashboard chuyên sâu." />
      <section className="mt-7 grid gap-4 sm:grid-cols-3">
        <MetricCard label="Drift state" value={overview.drift?.status ?? "NO_DATA"} detail="Prometheus reid_embedding_drift_detected" />
        <MetricCard label="Ready services" value={`${overview.services?.ready ?? 0}/${overview.services?.checked ?? 0}`} detail="MLflow, Prometheus, Qdrant, Triton, API" />
        <MetricCard label="Collection points" value={String(overview.collection?.points_count ?? 0)} detail="Qdrant active reference collection" />
      </section>
      <article className="panel mt-6 rounded-2xl p-6">
        <p className="eyebrow">Full dashboard</p><h2 className="mt-2 text-xl font-bold">Grafana model & system telemetry</h2>
        <p className="mt-2 text-sm leading-6 text-slate-400">Xem latency p50/p95/p99, error rate, GPU/CPU/RAM, embedding norm và similarity distribution.</p>
        {grafana && <a href={grafana.url} target="_blank" rel="noreferrer" className="mt-5 inline-flex rounded-lg border border-teal-300/35 bg-teal-300/10 px-4 py-2.5 text-sm font-bold text-teal-200 hover:bg-teal-300/15">Mở Grafana ↗</a>}
      </article>
    </div>
  );
}
