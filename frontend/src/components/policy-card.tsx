import Link from "next/link";

import type { PolicyDetail } from "@/lib/types";

function valueOrFallback(value?: string, fallback = "정보 확인 필요") {
  const normalized = value?.trim();
  return normalized || fallback;
}

function ageRange(policy: PolicyDetail) {
  const minimum = policy.sprtTrgtMinAge?.trim();
  const maximum = policy.sprtTrgtMaxAge?.trim();
  if (!minimum && !maximum) {
    return "연령 조건 확인 필요";
  }
  if (minimum && maximum) {
    return `${minimum}~${maximum}세`;
  }
  return minimum ? `${minimum}세 이상` : `${maximum}세 이하`;
}

type PolicyCardProps = {
  policy: PolicyDetail;
  compact?: boolean;
};

export function PolicyCard({ policy, compact = false }: PolicyCardProps) {
  return (
    <article className={`policy-card${compact ? " policy-card-compact" : ""}`}>
      <div className="policy-card-topline">
        <span className="badge badge-blue">현재 상담 정책</span>
        <span className="policy-category">
          {valueOrFallback(policy.mclsfNm, "청년정책")}
        </span>
      </div>
      <h3>{policy.plcyNm}</h3>
      <p className="policy-institution">
        {valueOrFallback(
          policy.operInstCdNm || policy.sprvsnInstCdNm,
          "운영기관 확인 필요",
        )}
      </p>
      <dl className="policy-summary">
        <div>
          <dt>신청 기간</dt>
          <dd>{valueOrFallback(policy.aplyYmd)}</dd>
        </div>
        <div>
          <dt>지원 연령</dt>
          <dd>{ageRange(policy)}</dd>
        </div>
      </dl>
      {!compact && policy.plcySprtCn ? (
        <p className="policy-support">{policy.plcySprtCn}</p>
      ) : null}
      <Link
        className="policy-detail-link"
        href={`/policies/${encodeURIComponent(policy.plcyNo)}`}
      >
        정책 상세 보기
        <span aria-hidden="true">→</span>
      </Link>
    </article>
  );
}
