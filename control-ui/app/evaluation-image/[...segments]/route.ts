import { CONTROL_API_URL } from "@/lib/api";

export async function GET(
  _request: Request,
  context: { params: Promise<{ segments: string[] }> },
) {
  const { segments } = await context.params;
  const path = segments.map(encodeURIComponent).join("/");
  const upstream = await fetch(`${CONTROL_API_URL}/control/evaluation-images/${path}`, {
    cache: "force-cache",
  });
  if (!upstream.ok) return new Response("Image not found", { status: 404 });
  return new Response(await upstream.arrayBuffer(), {
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "image/jpeg",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
