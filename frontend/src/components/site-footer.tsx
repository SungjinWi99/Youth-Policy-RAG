import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container footer-grid">
        <div>
          <strong>청년정책 상담 챗봇</strong>
          <p>
            이 서비스는 정책 탐색을 돕는 안내 도구이며 정부기관의 공식
            누리집이 아닙니다.
          </p>
        </div>
        <div className="footer-links">
          <Link href="/privacy">개인정보 및 이용 안내</Link>
          <span>공공데이터 기반 서비스</span>
        </div>
      </div>
    </footer>
  );
}
