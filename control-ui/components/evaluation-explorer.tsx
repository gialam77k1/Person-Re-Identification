"use client";

import { useMemo, useState } from "react";
import type { EvaluationImage, EvaluationVersion, IdentityCameraGroup } from "@/lib/types";

const metricLabels = {
  rank1: "Rank-1",
  rank5: "Rank-5",
  rank10: "Rank-10",
  rank20: "Rank-20",
  mAP: "mAP",
  mINP: "mINP",
} as const;

type MetricName = keyof typeof metricLabels;
const metricNames = Object.keys(metricLabels) as MetricName[];

function imageUrl(dataset: string, imagePath: string) {
  const segments = [dataset, ...imagePath.split("/")].map(encodeURIComponent);
  return `/evaluation-image/${segments.join("/")}`;
}

function percentage(value?: number | null) {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
}

function DatasetImage({ dataset, image, className = "" }: { dataset: string; image: EvaluationImage; className?: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return <div className={`flex items-center justify-center bg-slate-950 p-2 text-center text-[10px] text-amber-300 ${className}`}>Preview cũ chưa có ảnh evidence</div>;
  }
  return (
    <img
      src={image.asset_url ?? imageUrl(dataset, image.image_path)}
      alt={`Person ${image.person_id ?? "unknown"}, camera ${image.camera_id ?? "unknown"}`}
      loading="lazy"
      className={`bg-slate-950 object-cover ${className}`}
      onError={() => setFailed(true)}
    />
  );
}

export function EvaluationExplorer({
  dataset,
  samples,
  identityCameraGroups,
  versions,
}: {
  dataset: string;
  samples: EvaluationImage[];
  identityCameraGroups: IdentityCameraGroup[];
  versions: EvaluationVersion[];
}) {
  const [selectedKey, setSelectedKey] = useState(
    versions.length ? `${versions[0].release_id}:${versions[0].domain}` : "",
  );
  const selected = useMemo(
    () => versions.find((item) => `${item.release_id}:${item.domain}` === selectedKey) ?? versions[0],
    [selectedKey, versions],
  );

  return (
    <div className="mt-7 space-y-6">
      <section className="panel rounded-2xl p-5 sm:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Dataset preview</p>
            <h2 className="mt-2 text-xl font-bold">Mẫu query thật từ {dataset}</h2>
            <p className="mt-2 text-sm text-slate-400">Mẫu được chọn cố định, có person ID và camera ID để kiểm tra dữ liệu đầu vào.</p>
          </div>
          <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400">{samples.length} samples</span>
        </div>
        {samples.length ? (
          <div className="mt-5 grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-12">
            {samples.map((sample) => (
              <figure key={sample.image_path} className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950/70">
                <DatasetImage dataset={dataset} image={sample} className="aspect-[1/2] w-full" />
                <figcaption className="p-2 font-mono text-[10px] leading-4 text-slate-400">ID {sample.person_id ?? "—"}<br />CAM {sample.camera_id ?? "—"}</figcaption>
              </figure>
            ))}
          </div>
        ) : <p className="mt-5 text-sm text-amber-300">Dataset chưa được mount vào Control API.</p>}
      </section>

      <section className="panel rounded-2xl p-5 sm:p-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Cross-camera identity view</p>
            <h2 className="mt-2 text-xl font-bold">Cùng một ID qua nhiều camera</h2>
            <p className="mt-2 text-sm text-slate-400">Mỗi hàng là một người thật trong dataset. Ảnh được lấy từ các camera khác nhau để kiểm tra biến thiên góc nhìn, ánh sáng và chất lượng ảnh.</p>
          </div>
          <span className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400">{identityCameraGroups.length} identities</span>
        </div>
        {identityCameraGroups.length ? (
          <div className="mt-5 space-y-4">
            {identityCameraGroups.map((group) => (
              <article key={group.person_id} className="rounded-2xl border border-slate-700 bg-slate-950/35 p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-black text-teal-200">Person ID {group.person_id}</p>
                    <p className="mt-1 text-xs text-slate-500">Xuất hiện trong {group.camera_count} camera</p>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {group.images.map((image) => <span key={image.camera_id} className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-[10px] text-slate-400">CAM {image.camera_id}</span>)}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
                  {group.images.map((image) => (
                    <figure key={`${image.camera_id}-${image.image_path}`} className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950/70">
                      <DatasetImage dataset={dataset} image={image} className="aspect-[1/2] w-full" />
                      <figcaption className="p-2 text-center font-mono text-[10px] font-bold text-slate-400">CAM {image.camera_id}</figcaption>
                    </figure>
                  ))}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="mt-5 rounded-xl border border-amber-300/25 bg-amber-300/5 p-4 text-sm text-amber-200">Chưa tìm thấy identity có ảnh từ ít nhất hai camera. Kiểm tra dataset mount và quy tắc tên file ReID.</p>
        )}
      </section>

      <section className="panel rounded-2xl p-5 sm:p-6">
        <p className="eyebrow">Version comparison</p>
        {versions.length ? (
          <div className="mt-4 overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full divide-y divide-slate-700 text-left text-xs">
              <thead className="bg-slate-950/70 text-slate-500">
                <tr>
                  <th className="px-4 py-3">Model / domain</th>
                  <th className="px-4 py-3">Protocol</th>
                  {metricNames.map((metric) => <th key={metric} className="px-4 py-3">{metricLabels[metric]}</th>)}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {versions.map((version) => (
                  <tr key={`${version.release_id}:${version.domain}:summary`} className="bg-[#09111b]/50">
                    <td className="px-4 py-3"><span className="font-black text-white">{version.model_version.toUpperCase()}</span><span className="ml-2 text-slate-500">{version.domain}</span></td>
                    <td className="px-4 py-3 font-mono text-teal-200">{version.primary_protocol}</td>
                    {metricNames.map((metric) => <td key={metric} className="px-4 py-3 font-mono text-slate-300">{percentage(version.metrics[metric])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-2">
          {versions.map((version) => {
            const key = `${version.release_id}:${version.domain}`;
            return (
              <button
                key={key}
                type="button"
                onClick={() => setSelectedKey(key)}
                className={`rounded-lg border px-3 py-2 text-left text-xs font-bold transition ${key === selectedKey ? "border-teal-300 bg-teal-300/10 text-teal-200" : "border-slate-700 text-slate-400 hover:border-slate-500"}`}
              >
                {version.model_version.toUpperCase()} · {version.domain}
              </button>
            );
          })}
        </div>

        {selected ? (
          <div className="mt-6">
            <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
              <div>
                <h2 className="break-all text-xl font-bold">{selected.release_id}</h2>
                <p className="mt-1 text-sm text-slate-500">Đánh giá trên {selected.domain} · Git {selected.git_commit?.slice(0, 8) ?? "—"} · DVC {selected.dataset_dvc_revision?.slice(0, 10) ?? "—"}</p>
              </div>
              <p className="text-xs text-slate-500">Query {selected.metrics.query_count ?? "—"} · Gallery {selected.metrics.gallery_count ?? "—"}</p>
            </div>
            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
              {metricNames.map((metric) => (
                <div key={metric} className="rounded-xl border border-slate-700 bg-slate-950/50 p-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-slate-500">{metricLabels[metric]}</p>
                  <p className="mt-2 text-2xl font-black">{percentage(selected.metrics[metric])}</p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-slate-500">Primary protocol: <span className="font-mono font-bold text-teal-200">{selected.primary_protocol}</span></p>

            {selected.protocols.reranked ? (
              <div className="mt-6 grid gap-4 lg:grid-cols-2">
                {(["base", "reranked"] as const).map((protocol) => (
                  <div key={protocol} className="rounded-xl border border-slate-700 bg-slate-950/35 p-4">
                    <p className="text-xs font-black uppercase tracking-wider text-slate-400">{protocol} metrics</p>
                    <dl className="mt-3 grid grid-cols-3 gap-3 text-xs">
                      {metricNames.map((metric) => <div key={metric}><dt className="text-slate-600">{metricLabels[metric]}</dt><dd className="mt-1 font-mono font-bold text-slate-200">{percentage(selected.protocols[protocol]?.[metric])}</dd></div>)}
                    </dl>
                  </div>
                ))}
              </div>
            ) : null}

            {selected.retrieval_preview.length ? (
              <div className="mt-7 space-y-5">
                {selected.retrieval_preview.map((preview, index) => (
                  <article key={`${preview.query.image_path}-${index}`} className="rounded-2xl border border-slate-700 bg-slate-950/35 p-4">
                    <div className="grid gap-4 md:grid-cols-[150px_1fr]">
                      <div>
                        <p className="mb-2 text-xs font-black uppercase tracking-wider text-teal-300">Query</p>
                        <DatasetImage dataset={selected.dataset} image={preview.query} className="aspect-[1/2] w-full rounded-lg border border-teal-300/30" />
                        <p className="mt-2 font-mono text-xs text-slate-400">ID {preview.query.person_id} · CAM {preview.query.camera_id}</p>
                      </div>
                      <div>
                        <div className="mb-2 flex items-center justify-between">
                          <p className="text-xs font-black uppercase tracking-wider text-slate-400">Top gallery predictions</p>
                          <span className={`rounded-full px-2 py-1 text-[10px] font-black ${preview.top1_correct ? "bg-teal-300/10 text-teal-300" : "bg-rose-400/10 text-rose-300"}`}>TOP-1 {preview.top1_correct ? "CORRECT" : "WRONG"}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
                          {preview.predictions.map((prediction) => (
                            <figure key={`${prediction.rank}-${prediction.image_path}`} className={`overflow-hidden rounded-xl border ${prediction.correct ? "border-teal-300/50" : "border-slate-700"}`}>
                              <DatasetImage dataset={selected.dataset} image={prediction} className="aspect-[1/2] w-full" />
                              <figcaption className="space-y-1 p-2 text-[10px] text-slate-400">
                                <div className="flex justify-between font-black"><span>#{prediction.rank}</span><span className={prediction.correct ? "text-teal-300" : "text-rose-300"}>{prediction.correct ? "MATCH" : "MISS"}</span></div>
                                <p className="font-mono">ID {prediction.person_id} · C{prediction.camera_id}</p>
                                <p>Similarity {(prediction.similarity * 100).toFixed(2)}%</p>
                              </figcaption>
                            </figure>
                          ))}
                        </div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="mt-6 rounded-xl border border-amber-300/25 bg-amber-300/5 p-4 text-sm text-amber-200">
                Version này chưa có retrieval preview. Hãy chạy evaluation bằng pipeline mới; UI không tạo dự đoán giả để lấp dữ liệu.
              </div>
            )}
          </div>
        ) : (
          <div className="mt-5 rounded-xl border border-amber-300/25 bg-amber-300/5 p-4 text-sm text-amber-200">Chưa có report evaluation. Sau khi lifecycle đánh giá source champion và candidate kế tiếp, kết quả thật sẽ tự xuất hiện tại đây.</div>
        )}
      </section>
    </div>
  );
}
