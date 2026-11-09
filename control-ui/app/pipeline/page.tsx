import { PageHeader } from "@/components/page-header";
import { PipelineControl } from "@/components/pipeline-control";

export default function PipelinePage() {
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="Lifecycle orchestration" title="Pipeline control" description="Theo dõi cycle do Data Inbox tạo và phê duyệt retraining mà không cần mở terminal." />
      <PipelineControl />
    </div>
  );
}
