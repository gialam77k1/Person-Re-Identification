export default function OverviewPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <p className="eyebrow">System overview</p>
      <div className="mt-3 flex flex-col justify-between gap-4 border-b border-slate-800 pb-7 sm:flex-row sm:items-end">
        <div>
          <h1 className="text-3xl font-black tracking-tight sm:text-4xl">MLOps Control Center</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
            Một nơi duy nhất để quan sát model, drift và toàn bộ vòng đời từ v1 đến v2.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-bold text-teal-200">
          <span className="h-2 w-2 animate-pulse rounded-full bg-teal-400" /> LIVE CONTROL PLANE
        </div>
      </div>
      <section className="panel mt-8 rounded-2xl p-7">
        <p className="eyebrow">Initializing</p>
        <h2 className="mt-2 text-xl font-bold">Control API connection</h2>
        <p className="mt-2 text-sm text-slate-400">Dashboard modules will appear after the API client is connected.</p>
      </section>
    </div>
  );
}
