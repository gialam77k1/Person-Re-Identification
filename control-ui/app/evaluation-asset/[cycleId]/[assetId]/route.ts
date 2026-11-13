import { CONTROL_API_URL } from "@/lib/api";

export async function GET(
  _request: Request,
  context: { params: Promise<{ cycleId: string; assetId: string }> },
) {
  const { cycleId, assetId } = await context.params;
  const upstream = await fetch(
    `${CONTROL_API_URL}/control/evaluation-assets/${encodeURIComponent(cycleId)}/${encodeURIComponent(assetId)}`,
    { cache: "force-cache" },
  );
  if (!upstream.ok) return new Response("Image not found", { status: 404 });
  return new Response(await upstream.arrayBuffer(), {
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "image/jpeg",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
