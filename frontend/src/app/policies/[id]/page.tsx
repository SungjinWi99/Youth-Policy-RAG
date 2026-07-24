import { PolicyDetailClient } from "@/components/policy-detail-client";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";

export default function PolicyDetailPage() {
  return (
    <>
      <SiteHeader compact />
      <main className="policy-detail-page">
        <div className="container">
          <PolicyDetailClient />
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
