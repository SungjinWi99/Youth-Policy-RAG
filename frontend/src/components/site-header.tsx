import Link from "next/link";

type SiteHeaderProps = {
  compact?: boolean;
};

export function SiteHeader({ compact = false }: SiteHeaderProps) {
  return (
    <>
      <div className="service-disclosure">
        <div className="container">
          온통청년 공공데이터를 활용한 정책 안내 서비스입니다.
        </div>
      </div>
      <header className={`site-header${compact ? " site-header-compact" : ""}`}>
        <div className="container site-header-inner">
          <Link className="brand" href="/" aria-label="청년정책 상담 챗봇 홈">
            <span className="brand-mark" aria-hidden="true">
              청년
            </span>
            <span>청년정책 상담 챗봇</span>
          </Link>
          <nav className="site-nav" aria-label="주요 메뉴">
            <Link href="/chat">정책 상담</Link>
            <Link href="/privacy">개인정보 안내</Link>
          </nav>
        </div>
      </header>
    </>
  );
}
