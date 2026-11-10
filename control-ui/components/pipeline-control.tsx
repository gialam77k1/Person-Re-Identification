"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { StatusPill } from "@/components/status-pill";
import type { Cycle, CycleEvent, Stage } from "@/lib/types";

const API = (process.env.NEXT_PUBLIC_CONTROL_API_URL ?? "http://localhost:8090").replace(/\/$/, "");
const labels: Record<string, string> = {
  validate_data: "Validate dataset",
  extract_embeddings: "Extract current embeddings",
  detect_drift: "Detect drift",
  evaluate_v1: "Evaluate source champion",
  retraining_approval: "Retraining approval",
  train_v2: "Train next candidate",
  evaluate_v2: "Evaluate next candidate",
  validate_release: "Validation gate",
  export_onnx: "Export ONNX & verify parity",
  publish_artifacts: "Publish artifacts to DVC / MinIO",
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

function releaseVersion(release?: string) {
  const version = release?.match(/:v(\d+)/)?.[1];
  return version ? `v${version}` : release ?? "model";
}

function StageRow({ stage, cycle }: { stage: Stage; cycle: Cycle }) {
  const symbol = stage.status === "PASS" ? "✓" : stage.status === "FAIL" ? "×" : stage.status === "RUNNING" ? "●" : stage.status === "WAITING_APPROVAL" ? "!" : "○";
  const color = stage.status === "PASS" ? "text-teal-300" : stage.status === "FAIL" ? "text-rose-300" : stage.status === "RUNNING" ? "text-sky-300" : stage.status === "WAITING_APPROVAL" ? "text-amber-300" : "text-slate-600";
  return (
    <li className="relative grid grid-cols-[2rem_1fr_auto] gap-3 pb-6 last:pb-0">
      <div className={`relative z-10 grid h-8 w-8 place-items-center rounded-full border border-current bg-[#101b28] font-black ${color}`}>{symbol}</div>
      <div className="pt-1"><p className="text-sm font-bold">{stage.name === "evaluate_v1" ? `Evaluate ${releaseVersion(cycle.source_release)}` : stage.name === "train_v2" ? `Train ${releaseVersion(cycle.target_release)}` : stage.name === "evaluate_v2" ? `Evaluate ${releaseVersion(cycle.target_release)}` : labels[stage.name] ?? stage.name}</p>{stage.detail && <p className="mt-1 text-xs text-slate-500">{stage.detail}</p>}{stage.status === "RUNNING" && <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-sky-400 transition-all" style={{ width: `${Math.max(4, stage.progress * 100)}%` }} /></div>}</div>
      <div className="pt-1"><StatusPill status={stage.status} /></div>
      {stage.name !== "production_approval" && <span className="absolute left-[15px] top-8 h-[calc(100%-1.4rem)] w-px bg-slate-800" />}
    </li>
  );
}

function metricText(value: unknown, digits = 4) {
  return typeof value === "number" ? value.toFixed(digits) : String(value ?? "—");
}

export function PipelineControl() {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const [activeId, setActiveId] = useState("");
  const [cycle, setCycle] = useState<Cycle | null>(null);
  const [events, setEvents] = useState<CycleEvent[]>([]);
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
  const trainingMetrics = useMemo(() => events.filter((event) => event.kind === "stage.metrics" && event.payload.stage === "train_v2").at(-1)?.payload.metrics as Record<string, unknown> | undefined, [events]);
  const driftEvidence = useMemo(() => events.filter((event) => event.kind === "stage.metrics" && event.payload.stage === "detect_drift").at(-1)?.payload.metrics as Record<string, unknown> | undefined, [events]);
  const reviewEvidence = useMemo(() => events.filter((event) => event.kind === "stage.metrics" && event.payload.stage === "evaluate_v1").at(-1)?.payload.metrics as Record<string, unknown> | undefined, [events]);
  const approvedDecision = useMemo(() => events.filter((event) => event.kind === "retraining.decision_created").at(-1)?.payload, [events]);
  const driftChecks = driftEvidence?.checks as Record<string, { value: number; maximum: number; passed: boolean }> | undefined;
  const stage = (name: string) => cycle?.stages.find((item) => item.name === name);
  const canApprove = cycle?.status === "WAITING_APPROVAL" && stage("retraining_approval")?.status === "WAITING_APPROVAL";
  const canPromote = cycle?.status === "WAITING_APPROVAL" && stage("register_candidate")?.status === "PASS";

  async function perform(callback: () => Promise<void>) {
    setBusy(true); setError("");
    try { await callback(); await refreshList(); await refreshCycle(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
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
      {error && <p role="alert" className="mt-7 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">{error}</p>}
      <div className="mt-7 grid gap-6 xl:grid-cols-[1fr_0.65fr]">
        <section className="panel rounded-2xl p-6">
          <div className="flex flex-col justify-between gap-4 border-b border-slate-800 pb-5 sm:flex-row sm:items-center">
            <div><p className="eyebrow">Lifecycle timeline</p><h2 className="mt-2 text-xl font-bold">{cycle ? `${releaseVersion(cycle.source_release)} → ${releaseVersion(cycle.target_release)}` : "Champion → next version"}</h2></div>
            <select value={activeId} onChange={(event) => setActiveId(event.target.value)} className="max-w-xs rounded-lg border border-slate-700 bg-[#09111b] px-3 py-2 text-xs text-slate-300"><option value="">Select cycle</option>{cycles.map((item) => <option key={item.id} value={item.id}>{item.dataset} · {item.status} · {item.id.slice(0, 8)}</option>)}</select>
          </div>
          {cycle ? <ol className="mt-6">{cycle.stages.map((item) => <StageRow key={item.name} stage={item} cycle={cycle} />)}</ol> : <p className="mt-8 text-sm text-slate-500">Import batch hợp lệ trong Data Inbox để tạo cycle đầu tiên.</p>}
        </section>

        <aside className="space-y-6">
          <section className="panel rounded-2xl p-5">
            <div className="flex items-center justify-between"><p className="eyebrow">Cycle control</p>{cycle && <StatusPill status={cycle.status} />}</div>
            {cycle && <p className="mt-3 break-all font-mono text-[11px] text-slate-500">{cycle.id}</p>}
            <label className="mt-4 block text-xs font-bold text-slate-400">Operator<input value={actor} onChange={(event) => setActor(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-2 text-sm text-white" /></label>
            <div className="mt-5 grid gap-2">
              <button disabled={!canApprove || busy} onClick={() => action("approve-retraining", `Phê duyệt retraining ${cycle?.target_release ?? "model kế tiếp"} từ evidence ${String(reviewEvidence?.artifact_name ?? "chưa xác định")}?`)} className="rounded-lg border border-amber-300/30 bg-amber-300/10 px-4 py-3 text-sm font-bold text-amber-100 disabled:cursor-not-allowed disabled:opacity-30">Approve Retraining</button>
              <button disabled={!canPromote || busy} onClick={() => action("promote", "Promote model candidate lên production?")} className="rounded-lg border border-teal-300/30 bg-teal-300/10 px-4 py-3 text-sm font-bold text-teal-100 disabled:cursor-not-allowed disabled:opacity-30">Promote to Production</button>
              <button disabled={!cycle || busy} onClick={() => action("rollback", `Rollback production về ${cycle?.source_release ?? "model trước"}?`)} className="rounded-lg border border-rose-300/30 bg-rose-300/10 px-4 py-3 text-sm font-bold text-rose-100 disabled:cursor-not-allowed disabled:opacity-30">Rollback previous</button>
            </div>
          </section>
          <section className="panel overflow-hidden rounded-2xl p-5">
            <div className="flex items-center justify-between gap-3">
              <div><p className="eyebrow">{releaseVersion(cycle?.target_release)} training</p><p className="mt-2 text-sm text-slate-400">Self-hosted GPU runner</p></div>
              {stage("train_v2") && <StatusPill status={stage("train_v2")!.status} />}
            </div>
            {trainingMetrics ? <>
              <div className="mt-5 flex items-end justify-between"><p className="text-3xl font-black text-white">Epoch {metricText(trainingMetrics.epoch, 0)}<span className="text-base font-medium text-slate-500"> / {metricText(trainingMetrics.epochs, 0)}</span></p><p className="font-mono text-xs text-slate-500">{metricText(trainingMetrics.elapsed_seconds, 1)}s</p></div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-sky-400 to-teal-300 transition-all" style={{ width: `${Math.min(100, Math.max(2, Number(trainingMetrics.epoch ?? 0) / Math.max(1, Number(trainingMetrics.epochs ?? 30)) * 100))}%` }} /></div>
              <dl className="mt-5 grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Train loss</dt><dd className="mt-1 font-mono text-lg font-bold text-sky-200">{metricText(trainingMetrics.train_loss)}</dd></div>
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">mAP</dt><dd className="mt-1 font-mono text-lg font-bold text-teal-200">{metricText(trainingMetrics.mAP)}</dd></div>
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Backbone LR</dt><dd className="mt-1 font-mono text-sm text-sky-200">{metricText(trainingMetrics.backbone_lr, 7)}</dd></div>
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Head LR</dt><dd className="mt-1 font-mono text-sm text-sky-200">{metricText(trainingMetrics.head_lr, 7)}</dd></div>
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">GPU utilization</dt><dd className="mt-1 font-mono text-sm text-amber-200">{metricText(trainingMetrics.gpu_utilization_percent, 1)}%</dd></div>
                <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">GPU memory</dt><dd className="mt-1 font-mono text-sm text-amber-200">{metricText(trainingMetrics.gpu_memory_allocated_mib, 1)} MiB</dd></div>
              </dl>
              <div className="mt-4 rounded-lg border border-slate-800 bg-black/30 p-3"><p className="text-[10px] font-bold uppercase tracking-widest text-slate-600">Latest runner log</p><p className="mt-2 break-words font-mono text-xs leading-5 text-slate-300">{metricText(trainingMetrics.latest_log)}</p></div>
              {Boolean(trainingMetrics.mlflow_run_url) && <a href={String(trainingMetrics.mlflow_run_url)} target="_blank" rel="noreferrer" className="mt-4 inline-flex text-xs font-bold text-sky-300 hover:text-sky-200">Open live MLflow run ↗</a>}
            </> : <p className="mt-4 text-sm text-slate-500">Telemetry sẽ xuất hiện tự động khi runner bắt đầu epoch đầu tiên.</p>}
          </section>
          <section className="panel rounded-2xl p-5">
            <div className="flex items-center justify-between"><p className="eyebrow">Approval evidence</p>{Boolean(reviewEvidence?.status) && <StatusPill status={String(reviewEvidence?.status)} />}</div>
            {driftEvidence ? <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Release</dt><dd className="mt-1 break-all font-mono text-sky-200">{String(driftEvidence.release_id)}</dd></div>
              <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Drift</dt><dd className="mt-1 font-mono text-rose-200">{String(driftEvidence.drift_detected)}</dd></div>
              <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Current samples</dt><dd className="mt-1 font-mono">{String(driftEvidence.current_samples)}</dd></div>
              <div className="rounded-lg bg-[#09111b] p-3"><dt className="text-slate-500">Reference samples</dt><dd className="mt-1 font-mono">{String(driftEvidence.reference_samples)}</dd></div>
            </dl> : <p className="mt-4 text-sm text-slate-500">Evidence xuất hiện sau bước detect drift.</p>}
            {driftChecks && <div className="mt-3 space-y-2">{Object.entries(driftChecks).map(([name, check]) => <div key={name} className="flex justify-between gap-3 text-[11px]"><span className="text-slate-400">{name}</span><span className={check.passed ? "font-mono text-teal-300" : "font-mono text-rose-300"}>{check.value.toFixed(4)} / {check.maximum}</span></div>)}</div>}
            {approvedDecision && <div className="mt-4 rounded-lg border border-teal-300/20 bg-teal-300/5 p-3 text-xs"><p className="font-bold text-teal-200">{String(approvedDecision.status)}</p><p className="mt-1 text-slate-400">Approved by {String(approvedDecision.approved_by)} · {String(approvedDecision.approved_at)}</p><p className="mt-2 break-all font-mono text-[10px] text-slate-600">SHA256 {String(approvedDecision.decision_sha256)}</p></div>}
          </section>
          <section className="panel rounded-2xl p-5"><p className="eyebrow">Live metrics</p>{metrics ? <dl className="mt-4 grid grid-cols-2 gap-3">{Object.entries(metrics).map(([key, value]) => <div key={key} className="rounded-lg bg-[#09111b] p-3"><dt className="text-[10px] uppercase text-slate-500">{key}</dt><dd className="mt-1 font-mono text-lg font-bold text-sky-200">{String(value)}</dd></div>)}</dl> : <p className="mt-4 text-sm text-slate-500">Metrics sẽ xuất hiện khi runner gửi callback.</p>}</section>
          <section className="panel rounded-2xl p-5"><p className="eyebrow">Live events</p><div className="mt-4 max-h-80 space-y-3 overflow-auto">{events.slice().reverse().map((event) => <div key={event.id} className="border-l border-slate-700 pl-3"><p className="text-xs font-semibold text-slate-300">{event.message}</p><p className="mt-1 font-mono text-[10px] text-slate-600">{new Date(event.created_at).toLocaleString()}</p></div>)}</div></section>
        </aside>
      </div>
    </div>
  );
}
