import { DataInbox } from "@/components/data-inbox";
import { PageHeader } from "@/components/page-header";

export default function InboxPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader
        eyebrow="Dataset ingestion"
        title="Data Inbox"
        description="Phát hiện và import batch ReID có nhãn; batch hợp lệ tự tạo cycle drift/evaluation rồi dừng ở approval gate."
      />
      <DataInbox />
    </div>
  );
}
