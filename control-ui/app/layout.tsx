import type { Metadata } from "next";
import { Sidebar } from "@/components/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReID MLOps Control Center",
  description: "Operate iterative Person Re-Identification model lifecycles.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>
        <div className="grid-noise min-h-screen">
          <Sidebar />
          <main className="px-5 py-7 sm:px-8 lg:ml-64 lg:px-10 lg:py-9">{children}</main>
        </div>
      </body>
    </html>
  );
}
