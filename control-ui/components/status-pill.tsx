export function StatusPill({ status }: { status: string }) {
  const tone =
    status === "PASS" || status === "READY" || status === "STABLE"
      ? "border-teal-300/30 bg-teal-300/10 text-teal-200"
      : status === "FAIL" || status === "UNAVAILABLE" || status === "DRIFT"
        ? "border-rose-400/30 bg-rose-400/10 text-rose-200"
        : status.includes("WAITING") || status === "LINK"
          ? "border-amber-300/30 bg-amber-300/10 text-amber-200"
          : "border-sky-300/30 bg-sky-300/10 text-sky-200";
  return <span className={`rounded-full border px-2.5 py-1 text-[10px] font-black tracking-wider ${tone}`}>{status}</span>;
}
