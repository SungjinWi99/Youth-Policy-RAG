import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "청년정책 상담 챗봇",
    template: "%s | 청년정책 상담 챗봇",
  },
  description:
    "내 상황에 맞는 청년정책을 대화로 찾고 지원 조건과 신청 방법을 확인하세요.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
