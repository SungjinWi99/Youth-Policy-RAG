import { Suspense } from "react";

import { ChatClient } from "@/components/chat-client";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";

export default function ChatPage() {
  return (
    <>
      <SiteHeader compact />
      <main className="chat-page">
        <div className="container">
          <Suspense
            fallback={
              <div className="chat-loading" role="status">
                <span className="loading-spinner" aria-hidden="true" />
                <strong>상담 화면을 준비하고 있습니다</strong>
              </div>
            }
          >
            <ChatClient />
          </Suspense>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
