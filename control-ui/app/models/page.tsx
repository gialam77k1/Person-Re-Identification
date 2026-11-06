import { PageHeader } from "@/components/page-header";
import { StatusPill } from "@/components/status-pill";
import { getCycles, getOverview } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ModelsPage() {
  const [overview, cycles] = await Promise.all([getOverview(), getCycles()]);
  const model = overview.model ?? {};
  const releases = Array.from(new Set([model.release_id, ...cycles.map((cycle) => cycle.target_release)].filter(Boolean)));
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="Model registry" title="Models & releases" description="Truy vết model đang phục vụ và các candidate được tạo từ lifecycle." />
      <section className="mt-7 grid gap-5">
        {releases.length ? releases.map((release, index) => (
          <article key={String(release)} className="panel grid gap-5 rounded-2xl p-6 md:grid-cols-[1fr_auto] md:items-center">
            <div><p className="eyebrow">{index === 0 ? "Serving model" : "Lifecycle candidate"}</p><h2 className="mt-2 text-xl font-bold">{String(release)}</h2><p className="mt-2 text-sm text-slate-500">Embedding {String(model.embedding_dim ?? 512)}D · Git {String(model.git_commit ?? "pending")}</p></div>
            <StatusPill status={index === 0 ? "READY" : "PENDING"} />
          </article>
        )) : <p className="mt-8 text-slate-500">Chưa đọc được model release từ Control API.</p>}
      </section>
    </div>
  );
}
