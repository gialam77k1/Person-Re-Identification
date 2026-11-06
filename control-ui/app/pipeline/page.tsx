import { PageHeader } from "@/components/page-header";
import { PipelineControl } from "@/components/pipeline-control";

export default function PipelinePage() {
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader eyebrow="Lifecycle orchestration" title="Pipeline control" description="Khởi động, theo dõi và phê duyệt toàn bộ vòng đời model mà không cần mở terminal." />
      <PipelineControl />
    </div>
  );
}
