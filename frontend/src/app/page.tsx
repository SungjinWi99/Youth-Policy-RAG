import Link from "next/link";

import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";

const exampleQuestions = [
  "서울에서 취업 준비 중인데 받을 수 있는 지원이 있나요?",
  "월세나 주거비를 지원하는 청년정책을 알려주세요.",
  "창업을 준비하는 청년이 신청할 수 있는 사업이 궁금해요.",
];

export default function Home() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="hero">
          <div className="container hero-grid">
            <div className="hero-copy">
              <p className="eyebrow">공공데이터 기반 청년정책 안내</p>
              <h1>
                내 상황에 맞는 청년정책,
                <br />
                대화로 쉽게 찾아보세요
              </h1>
              <p className="hero-description">
                나이, 지역, 직업 등 간단한 정보를 바탕으로 관련 정책을
                찾아 지원 조건과 신청 방법을 이해하기 쉽게 안내합니다.
              </p>
              <div className="hero-actions">
                <Link className="button button-primary button-large" href="/chat">
                  상담 시작하기
                  <span aria-hidden="true">→</span>
                </Link>
                <a className="button button-secondary button-large" href="#how">
                  이용 방법 보기
                </a>
              </div>
              <p className="hero-note">
                회원가입 없이 이용할 수 있으며, 상담 기록은 마지막
                이용일로부터 30일간 보관됩니다.
              </p>
            </div>

            <div className="hero-panel" aria-label="상담 예시">
              <div className="hero-panel-header">
                <span className="status-dot" aria-hidden="true" />
                청년정책 상담
              </div>
              <div className="sample-message sample-message-user">
                서울에서 취업 준비 중인데 월세 지원을 받을 수 있을까요?
              </div>
              <div className="sample-message sample-message-bot">
                <span className="sample-label">상담 답변</span>
                현재 상황에서 확인해볼 만한 주거 지원 정책을 찾았습니다.
                신청 기간과 소득 조건을 함께 살펴볼게요.
              </div>
              <div className="sample-policy">
                <span className="badge badge-blue">현재 상담 정책</span>
                <strong>청년 주거비 지원사업</strong>
                <span>지원 조건과 신청 방법 확인하기</span>
              </div>
            </div>
          </div>
        </section>

        <section className="section" id="how">
          <div className="container">
            <div className="section-heading">
              <p className="eyebrow">이용 방법</p>
              <h2>세 단계로 필요한 정책을 확인하세요</h2>
            </div>
            <ol className="steps-grid">
              <li className="step-card">
                <span className="step-number">1</span>
                <h3>내 조건 입력</h3>
                <p>나이, 지역, 직업 등 정책 추천에 필요한 정보만 입력합니다.</p>
              </li>
              <li className="step-card">
                <span className="step-number">2</span>
                <h3>자유롭게 질문</h3>
                <p>지원이 필요한 상황을 평소 말하듯 편하게 질문합니다.</p>
              </li>
              <li className="step-card">
                <span className="step-number">3</span>
                <h3>정책과 신청 방법 확인</h3>
                <p>답변에 사용된 정책의 조건과 공식 신청 경로를 확인합니다.</p>
              </li>
            </ol>
          </div>
        </section>

        <section className="section section-tint">
          <div className="container question-section">
            <div className="section-heading">
              <p className="eyebrow">이렇게 물어보세요</p>
              <h2>상황을 구체적으로 말할수록 도움이 됩니다</h2>
            </div>
            <div className="question-list">
              {exampleQuestions.map((question) => (
                <Link
                  className="question-link"
                  href={`/chat?question=${encodeURIComponent(question)}`}
                  key={question}
                >
                  <span>{question}</span>
                  <span aria-hidden="true">→</span>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="section">
          <div className="container trust-grid">
            <div>
              <p className="eyebrow">안내 원칙</p>
              <h2>근거 정책을 함께 보여드립니다</h2>
            </div>
            <div className="trust-points">
              <p>
                답변에 사용된 정책은 상담 화면의
                <strong> 현재 상담 중인 정책</strong> 영역에서 확인할 수
                있습니다.
              </p>
              <p>
                실제 신청 가능 여부는 개인별 세부 조건과 기관 심사에 따라
                달라질 수 있으므로 공식 페이지에서 마지막으로 확인해 주세요.
              </p>
            </div>
          </div>
        </section>

        <section className="home-cta">
          <div className="container home-cta-inner">
            <div>
              <h2>지금 내게 필요한 청년정책을 찾아보세요</h2>
              <p>복잡한 정책 용어는 줄이고, 다음 행동은 분명하게 안내합니다.</p>
            </div>
            <Link className="button button-inverse button-large" href="/chat">
              무료로 상담 시작
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
