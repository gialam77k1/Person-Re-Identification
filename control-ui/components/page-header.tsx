export function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <header className="border-b border-slate-800 pb-7">
      <p className="eyebrow">{eyebrow}</p>
      <h1 className="mt-3 text-3xl font-black tracking-tight sm:text-4xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">{description}</p>
    </header>
  );
}
