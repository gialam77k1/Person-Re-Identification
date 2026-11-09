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

export type EvaluationImage = {
  image_path: string;
  person_id?: number | null;
  camera_id?: number | null;
};

export type RetrievalPrediction = EvaluationImage & {
  rank: number;
  similarity: number;
  distance: number;
  correct: boolean;
};

export type RetrievalPreview = {
  query: EvaluationImage;
  predictions: RetrievalPrediction[];
  top1_correct: boolean;
};

export type EvaluationVersion = {
  release_id: string;
  model_version: string;
  domain: string;
  dataset: string;
  cycle_id?: string | null;
  metrics: {
    rank1?: number | null;
    rank5?: number | null;
    mAP?: number | null;
    mINP?: number | null;
    query_count?: number | null;
    gallery_count?: number | null;
  };
  git_commit?: string | null;
  dataset_dvc_revision?: string | null;
  generated_at?: string | null;
  retrieval_preview: RetrievalPreview[];
};

export type EvaluationOverview = {
  dataset: string;
  dataset_samples: EvaluationImage[];
  versions: EvaluationVersion[];
};
