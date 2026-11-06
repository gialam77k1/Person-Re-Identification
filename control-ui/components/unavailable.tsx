export function Unavailable({ message }: { message: string }) {
  return (
    <div className="mt-6 rounded-xl border border-amber-300/25 bg-amber-300/5 px-4 py-3 text-sm text-amber-100">
      <strong>Control API chưa sẵn sàng.</strong> {message}
    </div>
  );
}
