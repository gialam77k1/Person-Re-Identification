"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { StatusPill } from "@/components/status-pill";
import type { InboxBatch, InboxPolicy, InboxSource } from "@/lib/types";

const API = (process.env.NEXT_PUBLIC_CONTROL_API_URL ?? "http://localhost:8090").replace(/\/$/, "");

type InboxResponse = { batches: InboxBatch[]; policy: InboxPolicy; sources: InboxSource[] };

function message(detail: unknown, fallback: string) {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return fallback;
}

export function DataInbox() {
  const [batches, setBatches] = useState<InboxBatch[]>([]);
  const [policy, setPolicy] = useState<InboxPolicy | null>(null);
  const [sources, setSources] = useState<InboxSource[]>([]);
  const [sourcePath, setSourcePath] = useState("dukemtmc-reid/bounding_box_train");
  const [dataset, setDataset] = useState("dukemtmc-reid");
  const [actor, setActor] = useState("gialam77k1");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const response = await fetch(`${API}/control/inbox`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Data Inbox returned ${response.status}`);
    const payload = (await response.json()) as InboxResponse;
    setBatches(payload.batches);
    setPolicy(payload.policy);
    setSources(payload.sources);
    if (
      payload.sources.length &&
      !payload.sources.some((source) => source.source_path === sourcePath)
    ) {
      setSourcePath(payload.sources[0].source_path);
      if (payload.sources[0].dataset) setDataset(payload.sources[0].dataset);
    }
  }, [sourcePath]);

  useEffect(() => {
    const initial = window.setTimeout(() => refresh().catch((reason) => setError(String(reason))), 0);
    const timer = window.setInterval(() => refresh().catch(() => undefined), 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refresh]);

  async function importBatch() {
    if (!window.confirm(`Import ${sourcePath} và tự khởi động lifecycle nếu batch hợp lệ?`)) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/control/inbox/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor, dataset, source_path: sourcePath }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(message(payload.detail, `Import failed (${response.status})`));
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-7 space-y-6">
      <section className="panel rounded-2xl p-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <p className="eyebrow">Immutable local batch</p>
            <h2 className="mt-2 text-xl font-bold">Versioned ReID batches</h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
              Mỗi batch mới nằm trong datasets/inbox/&lt;batch_id&gt; và có manifest.json.
              Query và test luôn giữ riêng cho evaluation.
              Hệ thống kiểm tra ảnh, đọc person/camera ID, tạo SHA256 toàn batch và chống dispatch trùng.
            </p>
          </div>
          <div className="grid min-w-[22rem] gap-3">
            <label className="text-xs font-bold text-slate-400">Detected batch
              <select value={sourcePath} onChange={(event) => {
                const selected = sources.find((item) => item.source_path === event.target.value);
                setSourcePath(event.target.value);
                if (selected?.dataset) setDataset(selected.dataset);
              }} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-3 text-sm text-white">
                {!sources.length && <option value="">No manifest batch detected</option>}
                {sources.map((source) => <option key={source.source_path} value={source.source_path}>{source.batch_id} · {source.source_path}</option>)}
              </select>
            </label>
            <div className="grid gap-3 sm:grid-cols-[1fr_1fr]">
              <label className="text-xs font-bold text-slate-400">Dataset
                <select value={dataset} onChange={(event) => setDataset(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-3 text-sm text-white">
                  {(policy?.allowed_datasets ?? ["dukemtmc-reid"]).map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
            <label className="text-xs font-bold text-slate-400">Operator
              <input value={actor} onChange={(event) => setActor(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-[#09111b] px-3 py-3 text-sm text-white" />
            </label>
            </div>
            <button onClick={importBatch} disabled={busy || !actor.trim() || !sourcePath} className="self-end rounded-lg bg-teal-300 px-5 py-3 text-sm font-black text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-40">
              {busy ? "VALIDATING…" : "IMPORT SELECTED BATCH"}
            </button>
          </div>
        </div>
        {policy && <dl className="mt-6 grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl bg-[#09111b] p-4"><dt className="text-xs text-slate-500">Minimum valid images</dt><dd className="mt-1 text-2xl font-black">{policy.min_valid_images}</dd></div>
          <div className="rounded-xl bg-[#09111b] p-4"><dt className="text-xs text-slate-500">Minimum identities</dt><dd className="mt-1 text-2xl font-black">{policy.min_identities}</dd></div>
          <div className="rounded-xl bg-[#09111b] p-4"><dt className="text-xs text-slate-500">Minimum cameras</dt><dd className="mt-1 text-2xl font-black">{policy.min_cameras}</dd></div>
        </dl>}
        {error && <p role="alert" className="mt-5 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">{error}</p>}
      </section>

      <section className="panel rounded-2xl p-6">
        <div><p className="eyebrow">Import history</p><h2 className="mt-2 text-xl font-bold">Batch audit trail</h2></div>
        {batches.length ? <div className="mt-6 space-y-4">{batches.map((batch) => (
          <article key={batch.id} className="rounded-xl border border-slate-800 bg-[#09111b]/70 p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div><p className="font-bold">{batch.dataset} · {batch.source_split}</p><p className="mt-1 font-mono text-[10px] text-slate-600">SHA256 {batch.fingerprint}</p></div>
              <StatusPill status={batch.status} />
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-5">
              <div><dt className="text-xs text-slate-500">Valid images</dt><dd className="mt-1 font-bold">{batch.valid_images}</dd></div>
              <div><dt className="text-xs text-slate-500">Identities</dt><dd className="mt-1 font-bold">{batch.identities}</dd></div>
              <div><dt className="text-xs text-slate-500">Cameras</dt><dd className="mt-1 font-bold">{batch.cameras}</dd></div>
              <div><dt className="text-xs text-slate-500">Rejected</dt><dd className="mt-1 font-bold">{batch.rejected_images}</dd></div>
              <div><dt className="text-xs text-slate-500">Duplicates</dt><dd className="mt-1 font-bold">{batch.duplicate_images}</dd></div>
            </dl>
            {batch.summary.validation_failures?.length ? <p className="mt-4 text-xs text-rose-300">{batch.summary.validation_failures.join(" · ")}</p> : null}
            {batch.error && <p className="mt-4 text-xs text-rose-300">{batch.error}</p>}
            <div className="mt-4 flex items-center justify-between text-xs text-slate-500"><span>{new Date(batch.created_at).toLocaleString()} · {batch.actor}</span>{batch.cycle_id && <Link href={`/pipeline?cycle=${batch.cycle_id}`} className="font-bold text-teal-300 hover:text-teal-200">Open cycle →</Link>}</div>
          </article>
        ))}</div> : <p className="mt-6 text-sm text-slate-500">Chưa import batch nào. Không có lifecycle nào được tự chạy khi chỉ mở trang.</p>}
      </section>
    </div>
  );
}
