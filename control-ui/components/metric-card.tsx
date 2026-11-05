export function MetricCard({ label, value, detail, accent = false }: { label: string; value: string | number; detail: string; accent?: boolean }) {
  return (
    <article className={`panel rounded-2xl p-5 ${accent ? "border-teal-300/35" : ""}`}>
      <p className="text-[11px] font-black uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className={`mt-3 break-words text-3xl font-black ${accent ? "text-teal-300" : "text-white"}`}>{value}</p>
      <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
    </article>
  );
}
