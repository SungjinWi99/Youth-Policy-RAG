"use client";

import { FormEvent, useState } from "react";

import { apiJson } from "@/lib/api";

const NEGATIVE_REASONS = [
  { value: "policy-mismatch", label: "내 상황과 맞지 않는 정책이에요" },
  { value: "outdated-information", label: "정보가 오래됐거나 신청할 수 없어요" },
  { value: "missing-details", label: "신청 조건이나 방법이 부족해요" },
  { value: "unclear-answer", label: "설명이 어렵거나 이해하기 힘들어요" },
  { value: "other", label: "그 밖의 이유가 있어요" },
] as const;

type FeedbackReason = (typeof NEGATIVE_REASONS)[number]["value"];
type SubmitState = "idle" | "submitting" | "submitted";

type AnswerFeedbackProps = {
  traceId: string;
};

export function AnswerFeedback({ traceId }: AnswerFeedbackProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [reason, setReason] = useState<FeedbackReason | "">("");
  const [comment, setComment] = useState("");
  const [submitState, setSubmitState] = useState<SubmitState>("idle");
  const [error, setError] = useState("");

  async function submitFeedback(
    helpful: boolean,
    selectedReason?: FeedbackReason,
  ) {
    setSubmitState("submitting");
    setError("");
    try {
      await apiJson<{ message: string }>("/me/feedback", {
        method: "POST",
        body: JSON.stringify({
          trace_id: traceId,
          helpful,
          reason: selectedReason,
          comment: comment.trim() || undefined,
        }),
      });
      setSubmitState("submitted");
    } catch (cause) {
      setSubmitState("idle");
      setError(
        cause instanceof Error
          ? cause.message
          : "피드백을 저장하지 못했습니다.",
      );
    }
  }

  function submitNegativeFeedback(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!reason) {
      setError("아쉬운 점을 하나 선택해 주세요.");
      return;
    }
    void submitFeedback(false, reason);
  }

  if (submitState === "submitted") {
    return (
      <div className="answer-feedback answer-feedback-complete" role="status">
        <span aria-hidden="true">✓</span>
        <span>피드백 감사합니다. 서비스 개선에 반영할게요.</span>
      </div>
    );
  }

  return (
    <section className="answer-feedback" aria-label="답변 피드백">
      <div className="feedback-question">
        <span>이 답변이 도움이 됐나요?</span>
        <div className="feedback-actions">
          <button
            type="button"
            onClick={() => void submitFeedback(true)}
            disabled={submitState === "submitting"}
            aria-label="도움됐어요"
          >
            <span aria-hidden="true">👍</span>
            도움됐어요
          </button>
          <button
            type="button"
            onClick={() => {
              setShowDetails(true);
              setError("");
            }}
            disabled={submitState === "submitting"}
            aria-expanded={showDetails}
            aria-label="아쉬워요"
          >
            <span aria-hidden="true">👎</span>
            아쉬워요
          </button>
        </div>
      </div>

      {showDetails ? (
        <form className="feedback-details" onSubmit={submitNegativeFeedback}>
          <fieldset>
            <legend>가장 아쉬웠던 점을 선택해 주세요</legend>
            <div className="feedback-reasons">
              {NEGATIVE_REASONS.map((item) => (
                <label key={item.value}>
                  <input
                    type="radio"
                    name={`feedback-reason-${traceId}`}
                    value={item.value}
                    checked={reason === item.value}
                    onChange={() => {
                      setReason(item.value);
                      setError("");
                    }}
                    disabled={submitState === "submitting"}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <label className="feedback-comment">
            <span>
              추가 의견 <small>선택</small>
            </span>
            <textarea
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="어떤 점이 달랐는지 알려주시면 개선에 도움이 됩니다."
              maxLength={500}
              rows={3}
              disabled={submitState === "submitting"}
            />
            <small>
              개인정보나 연락처는 입력하지 마세요. {comment.length}/500
            </small>
          </label>

          {error ? (
            <p className="feedback-error" role="alert">
              {error}
            </p>
          ) : null}

          <div className="feedback-submit-actions">
            <button
              type="button"
              onClick={() => {
                setShowDetails(false);
                setReason("");
                setComment("");
                setError("");
              }}
              disabled={submitState === "submitting"}
            >
              취소
            </button>
            <button
              className="feedback-submit"
              type="submit"
              disabled={submitState === "submitting"}
            >
              {submitState === "submitting"
                ? "전송 중..."
                : "피드백 보내기"}
            </button>
          </div>
        </form>
      ) : error ? (
        <p className="feedback-error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
