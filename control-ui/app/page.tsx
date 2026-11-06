import Link from "next/link";
import { MetricCard } from "@/components/metric-card";
import { PageHeader } from "@/components/page-header";
import { StatusPill } from "@/components/status-pill";
import { Unavailable } from "@/components/unavailable";
import { getOverview } from "@/lib/api";

export const dynamic = "force-dynamic";

function text(value: unknown, fallback = "—") {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

export default async function OverviewPage() {
  const data = await getOverview();
  const ready = data.services?.ready ?? 0;
  const checked = data.services?.checked ?? 0;
  const modelRelease = text(data.model?.release_id, "No active release");
  const points = text(data.collection?.points_count, "0");
  const cycle = data.latest_cycle;
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="System overview" title="MLOps Control Center" description="Theo dõi model, drift, hạ tầng và vòng đời v1 → v2 từ một giao diện duy nhất." />
      {data.unavailable && <Unavailable message={data.error ?? "Hãy khởi động control-api."} />}
      <section className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Service health" value={`${ready}/${checked}`} detail="Các dependency đang phản hồi" accent={ready > 0 && ready === checked} />
        <MetricCard label="Serving release" value={modelRelease} detail="Model release từ inference API" />
        <MetricCard label="Qdrant vectors" value={points} detail="Reference embeddings đang phục vụ" />
        <MetricCard label="Drift signal" value={data.drift?.status ?? "NO_DATA"} detail="Tín hiệu drift gần nhất từ Prometheus" accent={data.drift?.status === "STABLE"} />
      </section>
      <section className="mt-6 grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
        <article className="panel rounded-2xl p-6">
          <div className="flex items-center justify-between gap-4">
            <div><p className="eyebrow">Latest lifecycle</p><h2 className="mt-2 text-xl font-bold">Cycle gần nhất</h2></div>
            {cycle && <StatusPill status={cycle.status} />}
          </div>
          {cycle ? (
            <div className="mt-6">
              <p className="font-mono text-xs text-slate-500">{cycle.id}</p>
              <div className="mt-4 grid gap-4 sm:grid-cols-3">
                <div><p className="text-xs text-slate-500">Dataset</p><p className="mt-1 font-bold">{cycle.dataset}</p></div>
                <div><p className="text-xs text-slate-500">Source</p><p className="mt-1 truncate font-bold">{cycle.source_release}</p></div>
                <div><p className="text-xs text-slate-500">Target</p><p className="mt-1 truncate font-bold">{cycle.target_release}</p></div>
              </div>
              <Link href={`/pipeline?cycle=${cycle.id}`} className="mt-6 inline-flex rounded-lg bg-teal-300 px-4 py-2.5 text-sm font-black text-slate-950 transition hover:bg-teal-200">Mở pipeline →</Link>
            </div>
          ) : <p className="mt-8 text-sm text-slate-500">Chưa có cycle. Mở Pipeline để bắt đầu vòng đời đầu tiên.</p>}
        </article>
        <article className="panel rounded-2xl p-6">
          <p className="eyebrow">Quick assessment</p><h2 className="mt-2 text-xl font-bold">Release posture</h2>
          <div className="mt-6 space-y-4 text-sm">
            <div className="flex justify-between border-b border-slate-800 pb-3"><span className="text-slate-500">Embedding dimension</span><b>{text(data.model?.embedding_dim)}</b></div>
            <div className="flex justify-between border-b border-slate-800 pb-3"><span className="text-slate-500">Collection status</span><b>{text(data.collection?.status)}</b></div>
            <div className="flex justify-between"><span className="text-slate-500">Drift detected</span><b>{data.drift?.detected == null ? "Unknown" : data.drift.detected ? "Yes" : "No"}</b></div>
          </div>
        </article>
      </section>
    </div>
  );
}
