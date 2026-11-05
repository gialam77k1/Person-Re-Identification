export type Stage = {
  name: string;
  position: number;
  status: string;
  progress: number;
  detail?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

export type Cycle = {
  id: string;
  dataset: string;
  source_release: string;
  target_release: string;
  status: string;
  workflow_run_id?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
  stages: Stage[];
};

export type ServiceStatus = {
  name: string;
  status: string;
  url: string;
  detail?: unknown;
  error?: string;
};

export type Overview = {
  services?: { ready: number; checked: number; items?: ServiceStatus[] };
  model?: Record<string, unknown>;
  collection?: Record<string, unknown>;
  drift?: { status?: string; detected?: boolean | null; labels?: Record<string, string> };
  latest_cycle?: Cycle | null;
  allowed_datasets?: string[];
  unavailable?: boolean;
  error?: string;
};

export type CycleEvent = {
  id: number;
  cycle_id: string;
  kind: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
};
