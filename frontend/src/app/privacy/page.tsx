import Link from "next/link";

import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";

export default function PrivacyPage() {
  return (
    <>
      <SiteHeader compact />
      <main className="content-page">
        <div className="container content-narrow">
          <nav className="breadcrumb" aria-label="현재 위치">
            <Link href="/">홈</Link>
            <span aria-hidden="true">/</span>
            <span>개인정보 및 이용 안내</span>
          </nav>

          <header className="content-header">
            <p className="eyebrow">이용 안내</p>
            <h1>개인정보 및 상담 기록 안내</h1>
            <p>
              청년정책 상담 챗봇이 어떤 정보를 저장하고 어떻게 삭제하는지
              안내합니다.
            </p>
          </header>

          <div className="content-sections">
            <section>
              <h2>저장하는 정보</h2>
              <p>
                사용자가 선택적으로 입력한 나이, 지역, 직업, 소득, 성별과
                상담 질문·답변을 저장합니다. 브라우저에는 세션을 식별하기
                위한 보안 쿠키가 저장됩니다.
              </p>
            </section>
            <section>
              <h2>이용 목적</h2>
              <p>
                입력한 프로필은 관련 정책 검색과 상담 답변 생성에 사용됩니다.
                상담 기록은 후속 질문의 맥락을 유지하고 같은 브라우저에서
                상담을 이어가기 위해 사용됩니다.
              </p>
            </section>
            <section>
              <h2>보존 기간</h2>
              <p>
                프로필과 상담 기록은 마지막 이용일로부터 30일 동안
                보존합니다. 기간 안에 다시 방문하면 보존 기간이 갱신됩니다.
              </p>
            </section>
            <section>
              <h2>직접 삭제</h2>
              <p>
                상담 화면 하단의 ‘내 프로필과 상담 기록 모두 삭제’를 선택하면
                저장된 프로필, 대화 기록, 익명 세션을 즉시 삭제할 수 있습니다.
              </p>
            </section>
            <section>
              <h2>정책 안내의 한계</h2>
              <p>
                이 서비스는 정책 탐색을 돕는 안내 도구입니다. 답변이나 추천
                결과가 신청 자격과 선정 여부를 보장하지 않으며, 최종 조건은
                반드시 운영기관의 공식 안내에서 확인해야 합니다.
              </p>
            </section>
          </div>

          <div className="content-actions">
            <Link className="button button-primary" href="/chat">
              정책 상담 시작하기
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
