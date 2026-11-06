"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { StatusPill } from "@/components/status-pill";
import type { Cycle, CycleEvent, Stage } from "@/lib/types";

const API = (process.env.NEXT_PUBLIC_CONTROL_API_URL ?? "http://localhost:8090").replace(/\/$/, "");
const labels: Record<string, string> = {
  validate_data: "Validate dataset",
  extract_embeddings: "Extract current embeddings",
  detect_drift: "Detect drift",
  evaluate_v1: "Evaluate model v1",
  retraining_approval: "Retraining approval",
  train_v2: "Train model v2",
  evaluate_v2: "Evaluate model v2",
  validate_release: "Validation gate",
  export_onnx: "Export ONNX & verify parity",
  register_candidate: "Register MLflow candidate",
  deploy_staging: "Deploy staging",
  staging_test: "Test staging",
  production_approval: "Production approval",
};

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail ?? `Request failed with ${response.status}`);
  return body as T;
}

function StageRow({ stage }: { stage: Stage }) {
  const symbol = stage.status === "PASS" ? "✓" : stage.status === "FAIL" ? "×" : stage.status === "RUNNING" ? "●" : stage.status === "WAITING_APPROVAL" ? "!" : "○";
  const color = stage.status === "PASS" ? "text-teal-300" : stage.status === "FAIL" ? "text-rose-300" : stage.status === "RUNNING" ? "text-sky-300" : stage.status === "WAITING_APPROVAL" ? "text-amber-300" : "text-slate-600";
  return (
    <li className="relative grid grid-cols-[2rem_1fr_auto] gap-3 pb-6 last:pb-0">
      <div className={`relative z-10 grid h-8 w-8 place-items-center rounded-full border border-current bg-[#101b28] font-black ${color}`}>{symbol}</div>
      <div className="pt-1"><p className="text-sm font-bold">{labels[stage.name] ?? stage.name}</p>{stage.detail && <p className="mt-1 text-xs text-slate-500">{stage.detail}</p>}{stage.status === "RUNNING" && <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-sky-400 transition-all" style={{ width: `${Math.max(4, stage.progress * 100)}%` }} /></div>}</div>
      <div className="pt-1"><StatusPill status={stage.status} /></div>
      {stage.position < 12 && <span className="absolute left-[15px] top-8 h-[calc(100%-1.4rem)] w-px bg-slate-800" />}
    </li>
  );
}

export function PipelineControl() {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const [activeId, setActiveId] = useState("");
  const [cycle, setCycle] = useState<Cycle | null>(null);
  const [events, setEvents] = useState<CycleEvent[]>([]);
  const [dataset, setDataset] = useState("MSMT17");
  const [actor, setActor] = useState("gialam77k1");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refreshList = useCallback(async () => {
    const response = await jsonRequest<{ cycles: Cycle[] }>("/control/cycles");
    setCycles(response.cycles);
    if (!activeId && response.cycles[0]) setActiveId(response.cycles[0].id);
  }, [activeId]);

  const refreshCycle = useCallback(async () => {
    if (!activeId) return;
    const [nextCycle, eventResponse] = await Promise.all([
      jsonRequest<Cycle>(`/control/cycles/${activeId}`),
      jsonRequest<{ events: CycleEvent[] }>(`/control/cycles/${activeId}/events`),
    ]);
    setCycle(nextCycle);
    setEvents(eventResponse.events);
  }, [activeId]);

  useEffect(() => {
    const initial = window.setTimeout(
      () => refreshList().catch((reason) => setError(String(reason))),
      0,
    );
    return () => window.clearTimeout(initial);
  }, [refreshList]);
  useEffect(() => {
    const initial = window.setTimeout(
      () => refreshCycle().catch((reason) => setError(String(reason))),
      0,
    );
    const timer = window.setInterval(() => refreshCycle().catch(() => undefined), 3000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refreshCycle]);

  const metrics = useMemo(() => events.filter((event) => event.kind === "stage.metrics").at(-1)?.payload.metrics as Record<string, unknown> | undefined, [events]);
  const stage = (name: string) => cycle?.stages.find((item) => item.name === name);
  const canApprove = cycle?.status === "WAITING_APPROVAL" && stage("retraining_approval")?.status === "WAITING_APPROVAL";
  const canPromote = cycle?.status === "WAITING_APPROVAL" && stage("staging_test")?.status === "PASS";

  async function perform(callback: () => Promise<void>) {
    setBusy(true); setError("");
    try { await callback(); await refreshList(); await refreshCycle(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }

  async function start() {
    await perform(async () => {
      const idempotency = `${dataset.toLowerCase()}-${new Date().toISOString().slice(0, 10)}-v2`;
      const response = await jsonRequest<{ cycle: Cycle }>("/control/cycles", { method: "POST", body: JSON.stringify({ dataset, actor, idempotency_key: idempotency }) });
      setActiveId(response.cycle.id); setCycle(response.cycle);
    });
  }

  async function action(name: "approve-retraining" | "promote" | "rollback", prompt: string) {
    if (!cycle || !window.confirm(prompt)) return;
    await perform(async () => {
      const updated = await jsonRequest<Cycle>(`/control/cycles/${cycle.id}/${name}`, { method: "POST", body: JSON.stringify({ actor }) });
      setCycle(updated);
    });
  }

  return (
    <div>
      <section className="panel mt-7 rounded-2xl p-5">
        <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto] md:items-end">
          <label className="text-xs font-bold text-slate-400">Dataset<select value={dataset} onChange={(event) => setDataset(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-3 text-sm text-white"><option>MSMT17</option></select></label>
          <label className="text-xs font-bold text-slate-400">Operator<input value={actor} onChange={(event) => setActor(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-3 text-sm text-white" /></label>
          <button onClick={start} disabled={busy || !actor.trim()} className="rounded-lg bg-teal-300 px-5 py-3 text-sm font-black text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-40">{busy ? "PROCESSING…" : "START MLOPS LIFECYCLE"}</button>
        </div>
        {error && <p role="alert" className="mt-4 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">{error}</p>}
      </section>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_0.65fr]">
        <section className="panel rounded-2xl p-6">
          <div className="flex flex-col justify-between gap-4 border-b border-slate-800 pb-5 sm:flex-row sm:items-center">
            <div><p className="eyebrow">Lifecycle timeline</p><h2 className="mt-2 text-xl font-bold">Model v1 → model v2</h2></div>
            <select value={activeId} onChange={(event) => setActiveId(event.target.value)} className="max-w-xs rounded-lg border border-slate-700 bg-[#09111b] px-3 py-2 text-xs text-slate-300"><option value="">Select cycle</option>{cycles.map((item) => <option key={item.id} value={item.id}>{item.dataset} · {item.status} · {item.id.slice(0, 8)}</option>)}</select>
          </div>
          {cycle ? <ol className="mt-6">{cycle.stages.map((item) => <StageRow key={item.name} stage={item} />)}</ol> : <p className="mt-8 text-sm text-slate-500">Bấm Start để tạo cycle đầu tiên.</p>}
        </section>

        <aside className="space-y-6">
          <section className="panel rounded-2xl p-5">
            <div className="flex items-center justify-between"><p className="eyebrow">Cycle control</p>{cycle && <StatusPill status={cycle.status} />}</div>
            {cycle && <p className="mt-3 break-all font-mono text-[11px] text-slate-500">{cycle.id}</p>}
            <div className="mt-5 grid gap-2">
              <button disabled={!canApprove || busy} onClick={() => action("approve-retraining", "Phê duyệt retraining model v2?")} className="rounded-lg border border-amber-300/30 bg-amber-300/10 px-4 py-3 text-sm font-bold text-amber-100 disabled:cursor-not-allowed disabled:opacity-30">Approve Retraining</button>
              <button disabled={!canPromote || busy} onClick={() => action("promote", "Promote model candidate lên production?")} className="rounded-lg border border-teal-300/30 bg-teal-300/10 px-4 py-3 text-sm font-bold text-teal-100 disabled:cursor-not-allowed disabled:opacity-30">Promote to Production</button>
              <button disabled={!cycle || busy} onClick={() => action("rollback", "Rollback production về model v1?")} className="rounded-lg border border-rose-300/30 bg-rose-300/10 px-4 py-3 text-sm font-bold text-rose-100 disabled:cursor-not-allowed disabled:opacity-30">Rollback to v1</button>
            </div>
          </section>
          <section className="panel rounded-2xl p-5"><p className="eyebrow">Live metrics</p>{metrics ? <dl className="mt-4 grid grid-cols-2 gap-3">{Object.entries(metrics).map(([key, value]) => <div key={key} className="rounded-lg bg-[#09111b] p-3"><dt className="text-[10px] uppercase text-slate-500">{key}</dt><dd className="mt-1 font-mono text-lg font-bold text-sky-200">{String(value)}</dd></div>)}</dl> : <p className="mt-4 text-sm text-slate-500">Metrics sẽ xuất hiện khi runner gửi callback.</p>}</section>
          <section className="panel rounded-2xl p-5"><p className="eyebrow">Live events</p><div className="mt-4 max-h-80 space-y-3 overflow-auto">{events.slice().reverse().map((event) => <div key={event.id} className="border-l border-slate-700 pl-3"><p className="text-xs font-semibold text-slate-300">{event.message}</p><p className="mt-1 font-mono text-[10px] text-slate-600">{new Date(event.created_at).toLocaleString()}</p></div>)}</div></section>
        </aside>
      </div>
    </div>
  );
}
