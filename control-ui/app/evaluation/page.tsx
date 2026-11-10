import { EvaluationExplorer } from "@/components/evaluation-explorer";
import { PageHeader } from "@/components/page-header";
import { getEvaluations } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function EvaluationPage() {
  const evaluation = await getEvaluations();
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        eyebrow="Visual evaluation"
        title="Dataset & prediction explorer"
        description="Xem dữ liệu đầu vào, metric tổng hợp và top-k retrieval thật của từng model version."
      />
      <EvaluationExplorer
        dataset={evaluation.dataset}
        samples={evaluation.dataset_samples}
        identityCameraGroups={evaluation.identity_camera_groups}
        versions={evaluation.versions}
      />
    </div>
  );
}
