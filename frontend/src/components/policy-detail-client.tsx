"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

import { apiJson } from "@/lib/api";
import type { PolicyDetail } from "@/lib/types";

function display(value?: string, fallback = "정보 없음") {
  return value?.trim() || fallback;
}

function externalUrl(value?: string) {
  if (!value) {
    return null;
  }
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

export function PolicyDetailClient() {
  const params = useParams<{ id: string }>();
  const [policy, setPolicy] = useState<PolicyDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadPolicy() {
      try {
        const result = await apiJson<PolicyDetail>(
          `/policies/${encodeURIComponent(params.id)}`,
        );
        setPolicy(result);
      } catch (caught) {
        setError(
          caught instanceof Error
            ? caught.message
            : "정책 정보를 불러오지 못했습니다.",
        );
      }
    }
    void loadPolicy();
  }, [params.id]);

  if (error) {
    return (
      <div className="policy-detail-error">
        <p className="eyebrow">정책 조회 오류</p>
        <h1>정책 정보를 불러오지 못했습니다</h1>
        <p>{error}</p>
        <Link className="button button-primary" href="/chat">
          상담으로 돌아가기
        </Link>
      </div>
    );
  }

  if (!policy) {
    return (
      <div className="chat-loading" role="status">
        <span className="loading-spinner" aria-hidden="true" />
        <strong>정책 정보를 불러오고 있습니다</strong>
      </div>
    );
  }

  const applicationUrl = externalUrl(policy.aplyUrlAddr);
  const referenceUrls = [
    externalUrl(policy.refUrlAddr1),
    externalUrl(policy.refUrlAddr2),
  ].filter((url): url is string => Boolean(url));

  return (
    <article className="policy-detail">
      <nav className="breadcrumb" aria-label="현재 위치">
        <Link href="/">홈</Link>
        <span aria-hidden="true">/</span>
        <Link href="/chat">정책 상담</Link>
        <span aria-hidden="true">/</span>
        <span>정책 상세</span>
      </nav>

      <header className="policy-detail-header">
        <div>
          <span className="badge badge-blue">
            {display(policy.mclsfNm, "청년정책")}
          </span>
          <h1>{policy.plcyNm}</h1>
          <p>
            {display(
              policy.operInstCdNm || policy.sprvsnInstCdNm,
              "운영기관 정보 없음",
            )}
          </p>
        </div>
        {applicationUrl ? (
          <a
            className="button button-primary button-large"
            href={applicationUrl}
            target="_blank"
            rel="noreferrer"
          >
            공식 신청 페이지
            <span aria-hidden="true">↗</span>
          </a>
        ) : null}
      </header>

      <div className="notice-box">
        <strong>신청 전 확인</strong>
        <p>
          이 화면은 정책 원문을 이해하기 쉽게 정리한 안내입니다. 실제 신청
          가능 여부와 최신 일정은 운영기관의 공식 페이지에서 확인해 주세요.
        </p>
      </div>

      <dl className="policy-facts">
        <div>
          <dt>신청 기간</dt>
          <dd>{display(policy.aplyYmd)}</dd>
        </div>
        <div>
          <dt>지원 연령</dt>
          <dd>
            {policy.sprtTrgtMinAge || policy.sprtTrgtMaxAge
              ? `${display(policy.sprtTrgtMinAge, "-")} ~ ${display(
                  policy.sprtTrgtMaxAge,
                  "-",
                )}세`
              : "정보 없음"}
          </dd>
        </div>
        <div>
          <dt>정책 분야</dt>
          <dd>
            {display(policy.lclsfNm)} · {display(policy.mclsfNm)}
          </dd>
        </div>
        <div>
          <dt>등록 기관</dt>
          <dd>{display(policy.rgtrInstCdNm)}</dd>
        </div>
      </dl>

      <div className="policy-detail-grid">
        <section className="detail-section">
          <h2>정책 소개</h2>
          <p>{display(policy.plcyExplnCn)}</p>
        </section>
        <section className="detail-section">
          <h2>지원 내용</h2>
          <p>{display(policy.plcySprtCn)}</p>
        </section>
        <section className="detail-section">
          <h2>지원 대상</h2>
          <dl className="detail-list">
            <div>
              <dt>참여 대상</dt>
              <dd>{display(policy.ptcpPrpTrgtCn)}</dd>
            </div>
            <div>
              <dt>추가 자격 조건</dt>
              <dd>{display(policy.addAplyQlfcCndCn)}</dd>
            </div>
            <div>
              <dt>소득 조건</dt>
              <dd>{display(policy.earnEtcCn)}</dd>
            </div>
          </dl>
        </section>
        <section className="detail-section">
          <h2>신청 안내</h2>
          <dl className="detail-list">
            <div>
              <dt>신청 방법</dt>
              <dd>{display(policy.plcyAplyMthdCn)}</dd>
            </div>
            <div>
              <dt>제출 서류</dt>
              <dd>{display(policy.sbmsnDcmntCn)}</dd>
            </div>
            <div>
              <dt>심사 방법</dt>
              <dd>{display(policy.srngMthdCn)}</dd>
            </div>
          </dl>
        </section>
      </div>

      {referenceUrls.length ? (
        <section className="reference-section">
          <h2>참고 링크</h2>
          <div>
            {referenceUrls.map((url, index) => (
              <a href={url} target="_blank" rel="noreferrer" key={url}>
                참고 페이지 {index + 1}
                <span aria-hidden="true">↗</span>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      <div className="policy-detail-actions">
        <Link className="button button-secondary" href="/chat">
          ← 상담으로 돌아가기
        </Link>
        {applicationUrl ? (
          <a
            className="button button-primary"
            href={applicationUrl}
            target="_blank"
            rel="noreferrer"
          >
            공식 신청 페이지
            <span aria-hidden="true">↗</span>
          </a>
        ) : null}
      </div>
    </article>
  );
}
