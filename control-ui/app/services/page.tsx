import { PageHeader } from "@/components/page-header";
import { StatusPill } from "@/components/status-pill";
import { getServices } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function ServicesPage() {
  const services = await getServices();
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="Connected systems" title="Services" description="Health tổng hợp và đường dẫn tới từng giao diện chuyên biệt trong local stack." />
      <section className="mt-7 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {services.length ? services.map((service) => (
          <a key={service.name} href={service.url} target="_blank" rel="noreferrer" className="panel group rounded-2xl p-5 transition hover:-translate-y-0.5 hover:border-teal-300/35">
            <div className="flex items-start justify-between gap-4"><div><p className="text-lg font-black">{service.name}</p><p className="mt-1 max-w-[16rem] truncate text-xs text-slate-500">{service.url}</p></div><StatusPill status={service.status} /></div>
            <p className="mt-7 text-xs font-bold text-teal-300 opacity-70 group-hover:opacity-100">OPEN SERVICE ↗</p>
          </a>
        )) : <p className="text-sm text-slate-500">Control API chưa trả về danh sách service.</p>}
      </section>
    </div>
  );
}
