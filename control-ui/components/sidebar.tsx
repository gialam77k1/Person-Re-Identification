"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/", "Overview", "01"],
  ["/inbox", "Data Inbox", "02"],
  ["/pipeline", "Pipeline", "03"],
  ["/models", "Models", "04"],
  ["/evaluation", "Evaluation", "05"],
  ["/monitoring", "Monitoring", "06"],
  ["/services", "Services", "07"],
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="border-b border-[#26384a] bg-[#0b1520]/95 px-5 py-5 lg:fixed lg:inset-y-0 lg:left-0 lg:w-64 lg:border-b-0 lg:border-r lg:px-6 lg:py-8">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl border border-teal-300/30 bg-teal-300/10 font-black text-teal-300">R</div>
        <div>
          <p className="text-sm font-black tracking-wide">REID CONTROL</p>
          <p className="text-xs text-slate-500">MLOps command center</p>
        </div>
      </div>
      <nav className="mt-6 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:mt-12 lg:grid-cols-1" aria-label="Main navigation">
        {links.map(([href, label, number]) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`group flex items-center gap-3 rounded-lg border px-3 py-3 text-sm transition ${
                active
                  ? "border-teal-300/40 bg-teal-300/10 text-teal-200"
                  : "border-transparent text-slate-400 hover:border-slate-700 hover:bg-slate-800/50 hover:text-white"
              }`}
            >
              <span className="font-mono text-[10px] text-slate-600 group-hover:text-teal-400">{number}</span>
              <span className="font-semibold">{label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-10 hidden rounded-xl border border-slate-800 bg-[#09111b] p-4 lg:block">
        <div className="mb-2 flex items-center gap-2 text-xs font-bold text-slate-300">
          <span className="h-2 w-2 rounded-full bg-teal-400 shadow-[0_0_12px_#2dd4bf]" />
          LOCAL STACK
        </div>
        <p className="text-xs leading-5 text-slate-500">Control plane connected to the self-hosted GPU runner.</p>
      </div>
    </aside>
  );
}
