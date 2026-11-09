import type { Cycle, Overview, ServiceStatus } from "@/lib/types";

export const CONTROL_API_URL = (
  process.env.CONTROL_API_URL ?? "http://localhost:8090"
).replace(/\/$/, "");

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${CONTROL_API_URL}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Control API ${path} returned ${response.status}`);
  return response.json() as Promise<T>;
}

export async function getOverview(): Promise<Overview> {
  try {
    return await request<Overview>("/control/overview");
  } catch (error) {
    return {
      unavailable: true,
      error: error instanceof Error ? error.message : "Control API unavailable",
      services: { ready: 0, checked: 0, items: [] },
      drift: { status: "UNAVAILABLE", detected: null },
      latest_cycle: null,
      allowed_datasets: ["dukemtmc-reid"],
    };
  }
}

export async function getServices(): Promise<ServiceStatus[]> {
  try {
    return (await request<{ services: ServiceStatus[] }>("/control/services")).services;
  } catch {
    return [];
  }
}

export async function getCycles(): Promise<Cycle[]> {
  try {
    return (await request<{ cycles: Cycle[] }>("/control/cycles")).cycles;
  } catch {
    return [];
  }
}
